import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.ensemble import GradientBoostingRegressor

# ==========================================
# 1. SURVIVAL ANALYSIS
# ==========================================
kmf_asp = KaplanMeierFitter()
kmf_no_asp = KaplanMeierFitter()

mask_asp = df['asp_active'] == 1
kmf_asp.fit(df.loc[mask_asp, 'time_to_improvement'], event_observed=df.loc[mask_asp, 'event_observed'], label='ASP Active')
kmf_no_asp.fit(df.loc[~mask_asp, 'time_to_improvement'], event_observed=df.loc[~mask_asp, 'event_observed'], label='No ASP')

cox_cols = [
    'time_to_improvement', 'event_observed', 'hospital_id',
    'asp_active', 'age', 'comorbidity_score', 'immune_compromised',
    'severity_score', 'sex', 'ward', 'infection_type'
]
df_cox = pd.get_dummies(df[cox_cols], columns=['sex', 'ward', 'infection_type'], drop_first=True, dtype=float)

cph = CoxPHFitter()
cph.fit(
    df_cox,
    duration_col='time_to_improvement',
    event_col='event_observed',
    cluster_col='hospital_id'
)

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.standard_errors_['asp_active'])

ph_test = proportional_hazard_test(cph, df_cox, time_transform='rank')
ph_test_pvalues = {k: float(v) for k, v in ph_test.summary['p'].to_dict().items()}

# ==========================================
# 2. CAUSAL IMPACT (DiD)
# ==========================================
did_panel = df.groupby(['hospital_id', 'year'], as_index=False).agg(
    antibiotic_duration=('antibiotic_duration', 'mean'),
    asp_active=('asp_active', 'first'),
    asp_adoption_year=('asp_adoption_year', 'first'),
    age=('age', 'mean'),
    comorbidity_score=('comorbidity_score', 'mean'),
    severity_score=('severity_score', 'mean')
)

# Plain TWFE on hospital-year panel
twfe_mod = smf.ols('antibiotic_duration ~ asp_active + C(hospital_id) + C(year)', data=did_panel).fit(
    cov_type='cluster', cov_kwds={'groups': did_panel['hospital_id']}
)
twfe_att_estimate = float(twfe_mod.params['asp_active'])

# Event Study & Parallel Pre-Trends
did_panel['rel_time'] = np.where(
    did_panel['asp_adoption_year'] == 9999,
    -1,
    did_panel['year'] - did_panel['asp_adoption_year']
)

event_times = sorted([t for t in did_panel['rel_time'].unique() if t != -1])
es_terms = []
for t in event_times:
    col_name = f"evt_m{abs(t)}" if t < 0 else f"evt_p{t}"
    did_panel[col_name] = (did_panel['rel_time'] == t).astype(int)
    es_terms.append((t, col_name))

es_formula = 'antibiotic_duration ~ ' + ' + '.join([c for _, c in es_terms]) + ' + C(hospital_id) + C(year)'
es_mod = smf.ols(es_formula, data=did_panel).fit(
    cov_type='cluster', cov_kwds={'groups': did_panel['hospital_id']}
)

es_rows = []
for t, col_name in es_terms:
    coef = float(es_mod.params[col_name])
    ci_low, ci_high = es_mod.conf_int().loc[col_name]
    es_rows.append({
        'event_time': int(t),
        'coef': coef,
        'ci_low': float(ci_low),
        'ci_high': float(ci_high)
    })
event_study = pd.DataFrame(es_rows)

# Lead coefficient for pre-trend validation (use t = -2 as t = -1 is reference)
lead_col = 'evt_m2' if 'evt_m2' in [c for _, c in es_terms] else [c for t, c in es_terms if t < 0][-1]
pre_trend_coeff = float(es_mod.params[lead_col])
pre_trend_pvalue = float(es_mod.pvalues[lead_col])
pre_trend_ci = [float(x) for x in es_mod.conf_int().loc[lead_col]]

# Callaway-Sant'Anna style cohort-time ATT aggregation using never-treated controls (9999)
cohorts = sorted([c for c in did_panel['asp_adoption_year'].unique() if c != 9999])
cs_att_list = []
cs_weights = []

for g in cohorts:
    base_year = g - 1
    post_years = sorted([y for y in did_panel['year'].unique() if y >= g])
    
    ctrl_base = did_panel[(did_panel['asp_adoption_year'] == 9999) & (did_panel['year'] == base_year)]['antibiotic_duration'].mean()
    treat_base = did_panel[(did_panel['asp_adoption_year'] == g) & (did_panel['year'] == base_year)]['antibiotic_duration'].mean()
    
    for t in post_years:
        ctrl_t = did_panel[(did_panel['asp_adoption_year'] == 9999) & (did_panel['year'] == t)]['antibiotic_duration'].mean()
        treat_sub = did_panel[(did_panel['asp_adoption_year'] == g) & (did_panel['year'] == t)]
        treat_t = treat_sub['antibiotic_duration'].mean()
        
        att_gt = (treat_t - treat_base) - (ctrl_t - ctrl_base)
        cs_att_list.append(att_gt)
        cs_weights.append(len(treat_sub))

did_att_estimate = float(np.average(cs_att_list, weights=cs_weights))

# ==========================================
# 3. TREATMENT HETEROGENEITY (T-Learner)
# ==========================================
feature_cols = [
    'age', 'comorbidity_score', 'immune_compromised', 'severity_score',
    'sex', 'ward', 'infection_type', 'hospital_size', 'region', 'hospital_id', 'year'
]
X_all = pd.get_dummies(
    df[feature_cols],
    columns=['sex', 'ward', 'infection_type', 'hospital_size', 'region', 'hospital_id', 'year'],
    drop_first=True,
    dtype=float
)
y_all = df['antibiotic_duration'].values
t_all = df['asp_active'].values

m1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
m0 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)

m1.fit(X_all[t_all == 1], y_all[t_all == 1])
m0.fit(X_all[t_all == 0], y_all[t_all == 0])

cate_estimates = m1.predict(X_all) - m0.predict(X_all)

elderly_mask = (df['age'] >= 65).values
young_mask = (df['age'] < 40).values

cate_elderly = cate_estimates[elderly_mask]
cate_young = cate_estimates[young_mask]

rng = np.random.default_rng(42)
boot_diffs = []
for _ in range(1000):
    b_eld = rng.choice(cate_elderly, size=len(cate_elderly), replace=True).mean()
    b_yng = rng.choice(cate_young, size=len(cate_young), replace=True).mean()
    boot_diffs.append(b_eld - b_yng)

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
residual_variance = float(mixed_mod.scale)
icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))

mixed_model_aic = float(mixed_mod.aic)
naive_ols_aic = float(naive_ols.aic)
mixed_asp_coeff = float(mixed_mod.params['asp_active'])