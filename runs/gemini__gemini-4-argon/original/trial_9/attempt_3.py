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
for c in cox_df.columns:
    cox_df[c] = cox_df[c].astype(float)

cph = CoxPHFitter()
cph.fit(
    cox_df,
    duration_col='time_to_improvement',
    event_col='event_observed',
    cluster_col='hospital_id'
)

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.standard_errors_['asp_active'])

ph_res = proportional_hazard_test(cph, cox_df, time_transform='rank')
ph_test_pvalues = {str(k): float(v) for k, v in ph_res.summary['p'].to_dict().items()}


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
did_panel['mean_antibiotic_duration'] = did_panel['antibiotic_duration']
did_panel['ever_treated'] = (did_panel['asp_adoption_year'] != 9999).astype(int)
did_panel['event_time'] = np.where(
    did_panel['ever_treated'] == 1,
    did_panel['year'] - did_panel['asp_adoption_year'],
    -1
)

# Plain TWFE on hospital-year panel
twfe_mod = smf.ols(
    'antibiotic_duration ~ asp_active + C(hospital_id) + C(year)',
    data=did_panel
).fit(cov_type='cluster', cov_kwds={'groups': did_panel['hospital_id']})
twfe_att_estimate = float(twfe_mod.params['asp_active'])

# Event Study
es_df = did_panel.copy()
event_times = sorted([int(et) for et in es_df.loc[es_df['ever_treated'] == 1, 'event_time'].unique() if et != -1])
et_cols = []
for et in event_times:
    col_name = f"et_{'m' + str(abs(et)) if et < 0 else str(et)}"
    es_df[col_name] = ((es_df['ever_treated'] == 1) & (es_df['event_time'] == et)).astype(int)
    et_cols.append((et, col_name))

es_formula = 'antibiotic_duration ~ ' + ' + '.join([c for _, c in et_cols]) + ' + C(hospital_id) + C(year)'
es_mod = smf.ols(es_formula, data=es_df).fit(cov_type='cluster', cov_kwds={'groups': es_df['hospital_id']})

es_rows = []
es_ci = es_mod.conf_int()
for et, col_name in et_cols:
    es_rows.append({
        'event_time': int(et),
        'coef': float(es_mod.params[col_name]),
        'ci_low': float(es_ci.loc[col_name, 0]),
        'ci_high': float(es_ci.loc[col_name, 1])
    })
event_study = pd.DataFrame(es_rows).sort_values('event_time').reset_index(drop=True)

# Pre-trend test (using lead event_time = -2 from event study)
lead_col = 'et_m2' if 'et_m2' in es_mod.params else [c for et, c in et_cols if et < 0][-1]
pre_trend_coeff = float(es_mod.params[lead_col])
pre_trend_pvalue = float(es_mod.pvalues[lead_col])
pre_trend_ci = [float(es_ci.loc[lead_col, 0]), float(es_ci.loc[lead_col, 1])]

# Callaway-Sant'Anna / Sun-Abraham staggered DiD estimator using never-treated as controls
never_df = did_panel[did_panel['asp_adoption_year'] == 9999]
cohorts = sorted([int(c) for c in did_panel['asp_adoption_year'].unique() if c != 9999])
att_gt_list = []
weight_list = []

for g in cohorts:
    g_df = did_panel[did_panel['asp_adoption_year'] == g]
    n_g = len(g_df['hospital_id'].unique())
    base_year = g - 1
    y_g_base = g_df.loc[g_df['year'] == base_year, 'antibiotic_duration'].mean()
    y_c_base = never_df.loc[never_df['year'] == base_year, 'antibiotic_duration'].mean()
    
    post_years = sorted([int(y) for y in did_panel['year'].unique() if y >= g])
    for t in post_years:
        y_g_t = g_df.loc[g_df['year'] == t, 'antibiotic_duration'].mean()
        y_c_t = never_df.loc[never_df['year'] == t, 'antibiotic_duration'].mean()
        att_gt = (y_g_t - y_g_base) - (y_c_t - y_c_base)
        att_gt_list.append(att_gt)
        weight_list.append(n_g)

did_att_estimate = float(np.average(att_gt_list, weights=weight_list))


# ==========================================
# 3. TREATMENT HETEROGENEITY (T-Learner)
# ==========================================
feature_cols = [
    'age', 'comorbidity_score', 'immune_compromised', 'severity_score',
    'sex', 'ward', 'infection_type', 'hospital_size', 'region',
    'hospital_id', 'year'
]
X_all = pd.get_dummies(
    df[feature_cols],
    columns=['sex', 'ward', 'infection_type', 'hospital_size', 'region', 'hospital_id', 'year'],
    drop_first=True
).astype(float)
y_all = df['antibiotic_duration'].values
w_all = df['asp_active'].values

m0 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
m1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)

m0.fit(X_all[w_all == 0], y_all[w_all == 0])
m1.fit(X_all[w_all == 1], y_all[w_all == 1])

cate_estimates = np.asarray(m1.predict(X_all) - m0.predict(X_all), dtype=float)

# Bootstrap 95% CI for mean CATE(age>=65) - mean CATE(age<40)
elderly_mask = (df['age'] >= 65).values
young_mask = (df['age'] < 40).values
cate_elderly = cate_estimates[elderly_mask]
cate_young = cate_estimates[young_mask]

rng = np.random.default_rng(42)
boot_diffs = np.empty(1000, dtype=float)
n_e = len(cate_elderly)
n_y = len(cate_young)
for b in range(1000):
    e_sample = rng.choice(cate_elderly, size=n_e, replace=True)
    y_sample = rng.choice(cate_young, size=n_y, replace=True)
    boot_diffs[b] = np.mean(e_sample) - np.mean(y_sample)

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

random_effect_variance = float(np.asarray(mixed_mod.cov_re)[0, 0])
resid_variance = float(mixed_mod.scale)
icc_value = float(random_effect_variance / (random_effect_variance + resid_variance))

mixed_model_aic = float(mixed_mod.aic)
naive_ols_aic = float(naive_ols.aic)
mixed_asp_coeff = float(mixed_mod.fe_params['asp_active'])