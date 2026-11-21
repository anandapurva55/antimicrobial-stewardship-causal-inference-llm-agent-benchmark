import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.ensemble import GradientBoostingRegressor

# ==============================================================================
# 1. SURVIVAL ANALYSIS
# ==============================================================================
kmf = KaplanMeierFitter()
for asp_val, group in df.groupby('asp_active'):
    kmf.fit(group['time_to_improvement'], event_observed=group['event_observed'], label=f'ASP={asp_val}')

cox_cols = [
    'time_to_improvement', 'event_observed', 'hospital_id', 'asp_active',
    'age', 'comorbidity_score', 'immune_compromised', 'severity_score',
    'sex', 'ward', 'infection_type'
]
cox_df = pd.get_dummies(df[cox_cols], columns=['sex', 'ward', 'infection_type'], drop_first=True)
for col in cox_df.columns:
    if cox_df[col].dtype == bool:
        cox_df[col] = cox_df[col].astype(int)

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
ph_test_pvalues = {k: float(v) for k, v in ph_res.summary['p'].to_dict().items()}

# ==============================================================================
# 2. CAUSAL IMPACT (DiD)
# ==============================================================================
did_panel = df.groupby(['hospital_id', 'year'], as_index=False).agg(
    antibiotic_duration=('antibiotic_duration', 'mean'),
    asp_active=('asp_active', 'mean'),
    asp_adoption_year=('asp_adoption_year', 'first'),
    age=('age', 'mean'),
    comorbidity_score=('comorbidity_score', 'mean'),
    severity_score=('severity_score', 'mean'),
    n_patients=('patient_id', 'count')
)
did_panel['asp_active'] = (did_panel['asp_active'] > 0.5).astype(int)
did_panel['ever_treated'] = (did_panel['asp_adoption_year'] != 9999).astype(int)
did_panel['event_time'] = np.where(
    did_panel['ever_treated'] == 1,
    did_panel['year'] - did_panel['asp_adoption_year'],
    -1
)

# Plain TWFE on hospital-year panel
twfe_mod = smf.ols('antibiotic_duration ~ asp_active + C(hospital_id) + C(year)', data=did_panel).fit(
    cov_type='cluster', cov_kwds={'groups': did_panel['hospital_id']}
)
twfe_att_estimate = float(twfe_mod.params['asp_active'])

# Event Study
event_times = sorted([et for et in did_panel['event_time'].unique() if et != -1])
es_panel = did_panel.copy()
et_col_map = {}
for et in event_times:
    col_name = f"et_{'m' + str(abs(et)) if et < 0 else 'p' + str(et)}"
    es_panel[col_name] = ((es_panel['ever_treated'] == 1) & (es_panel['event_time'] == et)).astype(int)
    et_col_map[et] = col_name

es_formula = 'antibiotic_duration ~ ' + ' + '.join(et_col_map.values()) + ' + C(hospital_id) + C(year)'
es_mod = smf.ols(es_formula, data=es_panel).fit(
    cov_type='cluster', cov_kwds={'groups': es_panel['hospital_id']}
)

es_rows = []
es_conf = es_mod.conf_int()
for et in event_times:
    c = et_col_map[et]
    es_rows.append({
        'event_time': int(et),
        'coef': float(es_mod.params[c]),
        'ci_low': float(es_conf.loc[c, 0]),
        'ci_high': float(es_conf.loc[c, 1])
    })
event_study = pd.DataFrame(es_rows).sort_values('event_time').reset_index(drop=True)

# Pre-trend test (using lead t = -2 relative to reference t = -1)
lead_col = et_col_map[-2]
pre_trend_coeff = float(es_mod.params[lead_col])
pre_trend_pvalue = float(es_mod.pvalues[lead_col])
pre_trend_ci = [float(es_conf.loc[lead_col, 0]), float(es_conf.loc[lead_col, 1])]

# Callaway-Sant'Anna / Sun-Abraham cohort-time ATT aggregation using never-treated controls
never_treated = did_panel[did_panel['asp_adoption_year'] == 9999]
treated_cohorts = sorted([g for g in did_panel['asp_adoption_year'].unique() if g != 9999])

att_gt_list = []
weights_list = []
for g in treated_cohorts:
    base_year = g - 1
    cohort_df = did_panel[did_panel['asp_adoption_year'] == g]
    y_treat_base = cohort_df[cohort_df['year'] == base_year]['antibiotic_duration'].mean()
    y_ctrl_base = never_treated[never_treated['year'] == base_year]['antibiotic_duration'].mean()
    
    post_years = sorted([y for y in did_panel['year'].unique() if y >= g])
    for t in post_years:
        y_treat_t = cohort_df[cohort_df['year'] == t]['antibiotic_duration'].mean()
        y_ctrl_t = never_treated[never_treated['year'] == t]['antibiotic_duration'].mean()
        att_gt = (y_treat_t - y_treat_base) - (y_ctrl_t - y_ctrl_base)
        n_treat_gt = len(cohort_df[cohort_df['year'] == t])
        att_gt_list.append(att_gt)
        weights_list.append(n_treat_gt)

did_att_estimate = float(np.average(att_gt_list, weights=weights_list))

# ==============================================================================
# 3. TREATMENT HETEROGENEITY (T-Learner)
# ==============================================================================
feature_cols = [
    'age', 'comorbidity_score', 'immune_compromised', 'severity_score',
    'sex', 'ward', 'infection_type', 'hospital_size', 'region', 'hospital_id', 'year'
]
X_df = pd.get_dummies(
    df[feature_cols],
    columns=['sex', 'ward', 'infection_type', 'hospital_size', 'region', 'hospital_id', 'year'],
    drop_first=False
)
X_mat = X_df.values.astype(float)
y_vec = df['antibiotic_duration'].values.astype(float)
t_vec = df['asp_active'].values.astype(int)

m0 = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
m1 = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)

m0.fit(X_mat[t_vec == 0], y_vec[t_vec == 0])
m1.fit(X_mat[t_vec == 1], y_vec[t_vec == 1])

cate_estimates = m1.predict(X_mat) - m0.predict(X_mat)

elderly_cates = cate_estimates[df['age'].values >= 65]
young_cates = cate_estimates[df['age'].values < 40]

rng = np.random.default_rng(42)
boot_diffs = np.empty(1000)
n_eld = len(elderly_cates)
n_yng = len(young_cates)
for b in range(1000):
    e_sample = rng.choice(elderly_cates, size=n_eld, replace=True)
    y_sample = rng.choice(young_cates, size=n_yng, replace=True)
    boot_diffs[b] = np.mean(e_sample) - np.mean(y_sample)

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5))
]

# ==============================================================================
# 4. HIERARCHICAL MODELING
# ==============================================================================
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
mixed_asp_coeff = float(mixed_mod.params['asp_active'])