import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.ensemble import GradientBoostingRegressor

# ==========================================
# 1. SURVIVAL ANALYSIS
# ==========================================
kmf = KaplanMeierFitter()
for val, grp in df.groupby('asp_active'):
    kmf.fit(grp['time_to_improvement'], event_observed=grp['event_observed'], label=f'ASP_{val}')

cox_cols = [
    'time_to_improvement', 'event_observed', 'hospital_id', 'asp_active',
    'age', 'comorbidity_score', 'immune_compromised', 'severity_score',
    'sex', 'ward', 'infection_type'
]
cox_df = pd.get_dummies(df[cox_cols], columns=['sex', 'ward', 'infection_type'], drop_first=True)

cph = CoxPHFitter()
cph.fit(
    cox_df,
    duration_col='time_to_improvement',
    event_col='event_observed',
    cluster_col='hospital_id',
    robust=True
)

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.standard_errors_['asp_active'])

ph_test = proportional_hazard_test(cph, cox_df, time_transform='rank')
ph_test_pvalues = {str(k): float(v) for k, v in ph_test.summary['p'].to_dict().items()}

# ==========================================
# 2. CAUSAL IMPACT (DiD)
# ==========================================
did_panel = df.groupby(['hospital_id', 'year'], as_index=False).agg(
    antibiotic_duration=('antibiotic_duration', 'mean'),
    asp_active=('asp_active', 'mean'),
    asp_adoption_year=('asp_adoption_year', 'first'),
    age=('age', 'mean'),
    comorbidity_score=('comorbidity_score', 'mean'),
    severity_score=('severity_score', 'mean')
)
did_panel['asp_active'] = (did_panel['asp_active'] > 0.5).astype(int)
did_panel['treated_ever'] = (did_panel['asp_adoption_year'] < 9999).astype(int)
did_panel['event_time'] = np.where(
    did_panel['treated_ever'] == 1,
    did_panel['year'] - did_panel['asp_adoption_year'],
    -1
)

# Plain TWFE on hospital-year panel
twfe_mod = smf.ols('antibiotic_duration ~ asp_active + C(hospital_id) + C(year)', data=did_panel).fit(
    cov_type='cluster', cov_kwds={'groups': did_panel['hospital_id']}
)
twfe_att_estimate = float(twfe_mod.params['asp_active'])

# Event study on hospital-year panel (reference period = -1)
es_panel = did_panel.copy()
event_times = sorted([int(et) for et in es_panel.loc[es_panel['treated_ever'] == 1, 'event_time'].unique() if et != -1])
es_terms = []
for et in event_times:
    col = f'et_m{abs(et)}' if et < 0 else f'et_p{et}'
    es_panel[col] = ((es_panel['treated_ever'] == 1) & (es_panel['event_time'] == et)).astype(int)
    es_terms.append((et, col))

es_formula = 'antibiotic_duration ~ ' + ' + '.join([c for _, c in es_terms]) + ' + C(hospital_id) + C(year)'
es_mod = smf.ols(es_formula, data=es_panel).fit(
    cov_type='cluster', cov_kwds={'groups': es_panel['hospital_id']}
)

es_ci = es_mod.conf_int()
es_rows = []
for et, col in es_terms:
    es_rows.append({
        'event_time': int(et),
        'coef': float(es_mod.params[col]),
        'ci_low': float(es_ci.loc[col, 0]),
        'ci_high': float(es_ci.loc[col, 1])
    })
event_study = pd.DataFrame(es_rows).sort_values('event_time').reset_index(drop=True)

# Pre-trend test (using event_time = -2 lead coefficient)
lead_col = 'et_m2' if 'et_m2' in es_mod.params else [c for et, c in es_terms if et < 0][-1]
pre_trend_coeff = float(es_mod.params[lead_col])
pre_trend_pvalue = float(es_mod.pvalues[lead_col])
pre_trend_ci = [float(es_ci.loc[lead_col, 0]), float(es_ci.loc[lead_col, 1])]

# Callaway-Sant'Anna style cohort-time ATT estimator using never-treated controls (9999)
never_treated = did_panel[did_panel['asp_adoption_year'] == 9999]
cohorts = sorted([c for c in did_panel['asp_adoption_year'].unique() if c < 9999])

cs_atts = []
cs_weights = []
for g in cohorts:
    base_year = g - 1
    g_panel = did_panel[did_panel['asp_adoption_year'] == g]
    y_g_base = g_panel.loc[g_panel['year'] == base_year, 'antibiotic_duration'].mean()
    y_c_base = never_treated.loc[never_treated['year'] == base_year, 'antibiotic_duration'].mean()
    
    post_years = sorted([y for y in did_panel['year'].unique() if y >= g])
    for t in post_years:
        y_g_t = g_panel.loc[g_panel['year'] == t, 'antibiotic_duration'].mean()
        y_c_t = never_treated.loc[never_treated['year'] == t, 'antibiotic_duration'].mean()
        att_gt = (y_g_t - y_g_base) - (y_c_t - y_c_base)
        
        n_gt = len(df[(df['asp_adoption_year'] == g) & (df['year'] == t)])
        cs_atts.append(att_gt)
        cs_weights.append(n_gt)

did_att_estimate = float(np.average(cs_atts, weights=cs_weights))

# ==========================================
# 3. TREATMENT HETEROGENEITY (T-Learner / X-Learner)
# ==========================================
feature_cols = [
    'age', 'comorbidity_score', 'immune_compromised', 'severity_score',
    'sex', 'ward', 'infection_type', 'hospital_size', 'region', 'hospital_id', 'year'
]
X_all = pd.get_dummies(
    df[feature_cols],
    columns=['sex', 'ward', 'infection_type', 'hospital_size', 'region', 'hospital_id', 'year'],
    drop_first=True
)
y_all = df['antibiotic_duration'].values
t_all = df['asp_active'].values

m0 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
m1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)

m0.fit(X_all[t_all == 0], y_all[t_all == 0])
m1.fit(X_all[t_all == 1], y_all[t_all == 1])

cate_estimates = np.asarray(m1.predict(X_all) - m0.predict(X_all), dtype=float)

elderly_idx = np.where(df['age'].values >= 65)[0]
young_idx = np.where(df['age'].values < 40)[0]

rng = np.random.default_rng(42)
boot_diffs = []
for _ in range(500):
    b_eld = rng.choice(elderly_idx, size=len(elderly_idx), replace=True)
    b_yng = rng.choice(young_idx, size=len(young_idx), replace=True)
    boot_diffs.append(np.mean(cate_estimates[b_eld]) - np.mean(cate_estimates[b_yng]))

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5))
]

# ==========================================
# 4. HIERARCHICAL MODELING
# ==========================================
fe_formula = (
    'antibiotic_duration ~ asp_active + age + comorbidity_score + '
    'immune_compromised + severity_score + C(sex) + C(ward) + C(infection_type) + C(year)'
)

mixed_mod = smf.mixedlm(fe_formula, data=df, groups=df['hospital_id']).fit(reml=False)
naive_ols = smf.ols(fe_formula, data=df).fit()

random_effect_variance = float(mixed_mod.cov_re.iloc[0, 0])
resid_variance = float(mixed_mod.scale)
icc_value = float(random_effect_variance / (random_effect_variance + resid_variance))

k_params = len(mixed_mod.fe_params) + 2
mixed_model_aic = float(mixed_mod.aic) if not np.isnan(mixed_mod.aic) else float(-2.0 * mixed_mod.llf + 2.0 * k_params)
naive_ols_aic = float(naive_ols.aic)
mixed_asp_coeff = float(mixed_mod.fe_params['asp_active'])