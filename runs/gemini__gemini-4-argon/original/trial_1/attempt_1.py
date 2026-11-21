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
kmf = KaplanMeierFitter()
for asp_val, group in df.groupby('asp_active'):
    kmf.fit(group['time_to_improvement'], event_observed=group['event_observed'], label=f'ASP_{asp_val}')

cox_cols = [
    'time_to_improvement', 'event_observed', 'hospital_id', 'asp_active',
    'age', 'sex', 'comorbidity_score', 'immune_compromised', 'severity_score',
    'ward', 'infection_type', 'hospital_size'
]
df_cox = pd.get_dummies(
    df[cox_cols],
    columns=['sex', 'ward', 'infection_type', 'hospital_size'],
    drop_first=True,
    dtype=float
)

cph = CoxPHFitter()
cph.fit(
    df_cox,
    duration_col='time_to_improvement',
    event_col='event_observed',
    cluster_col='hospital_id'
)

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.standard_errors_['asp_active'])

ph_results = proportional_hazard_test(cph, df_cox, time_transform='rank')
ph_test_pvalues = {str(k): float(v) for k, v in ph_results.summary['p'].to_dict().items()}


# ==========================================
# 2. CAUSAL IMPACT (DiD)
# ==========================================
did_panel = df.groupby(['hospital_id', 'year'], as_index=False).agg(
    antibiotic_duration=('antibiotic_duration', 'mean'),
    asp_active=('asp_active', 'mean'),
    asp_adoption_year=('asp_adoption_year', 'first'),
    age=('age', 'mean'),
    comorbidity_score=('comorbidity_score', 'mean'),
    severity_score=('severity_score', 'mean'),
    time_to_improvement=('time_to_improvement', 'mean'),
    length_of_stay=('length_of_stay', 'mean'),
    n_patients=('patient_id', 'count')
)
did_panel['mean_antibiotic_duration'] = did_panel['antibiotic_duration']
did_panel['treated_ever'] = (did_panel['asp_adoption_year'] != 9999).astype(int)
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

# Event Study
es_times = sorted([int(t) for t in did_panel.loc[did_panel['treated_ever'] == 1, 'event_time'].unique() if t != -1])
es_term_map = {}
for t in es_times:
    col_name = f"et_{'m' + str(abs(t)) if t < 0 else 'p' + str(t)}"
    did_panel[col_name] = ((did_panel['treated_ever'] == 1) & (did_panel['event_time'] == t)).astype(int)
    es_term_map[t] = col_name

es_formula = 'antibiotic_duration ~ ' + ' + '.join(es_term_map.values()) + ' + C(hospital_id) + C(year)'
es_mod = smf.ols(es_formula, data=did_panel).fit(
    cov_type='cluster', cov_kwds={'groups': did_panel['hospital_id']}
)

es_ci = es_mod.conf_int()
es_rows = []
for t in es_times:
    col = es_term_map[t]
    es_rows.append({
        'event_time': int(t),
        'coef': float(es_mod.params[col]),
        'ci_low': float(es_ci.loc[col, 0]),
        'ci_high': float(es_ci.loc[col, 1])
    })
event_study = pd.DataFrame(es_rows).sort_values('event_time').reset_index(drop=True)

# Pre-trend test (using immediate pre-treatment lead t = -2 vs reference t = -1)
lead_col = es_term_map[-2] if -2 in es_term_map else es_term_map[min(es_times)]
pre_trend_coeff = float(es_mod.params[lead_col])
pre_trend_pvalue = float(es_mod.pvalues[lead_col])
pre_trend_ci = [float(es_ci.loc[lead_col, 0]), float(es_ci.loc[lead_col, 1])]

# Callaway-Sant'Anna / Sun-Abraham estimator robust to staggered adoption (never-treated controls)
cohorts = sorted([int(g) for g in did_panel['asp_adoption_year'].unique() if g != 9999])
never_treated_panel = did_panel[did_panel['asp_adoption_year'] == 9999]

cs_att_gt = []
cs_weights = []

for g in cohorts:
    cohort_panel = did_panel[did_panel['asp_adoption_year'] == g]
    n_hospitals_g = cohort_panel['hospital_id'].nunique()
    base_year = g - 1
    
    y_g_base = cohort_panel.loc[cohort_panel['year'] == base_year, 'antibiotic_duration'].mean()
    y_c_base = never_treated_panel.loc[never_treated_panel['year'] == base_year, 'antibiotic_duration'].mean()
    
    post_years = sorted([int(y) for y in did_panel['year'].unique() if y >= g])
    for t in post_years:
        y_g_t = cohort_panel.loc[cohort_panel['year'] == t, 'antibiotic_duration'].mean()
        y_c_t = never_treated_panel.loc[never_treated_panel['year'] == t, 'antibiotic_duration'].mean()
        
        att_gt = (y_g_t - y_g_base) - (y_c_t - y_c_base)
        cs_att_gt.append(att_gt)
        cs_weights.append(n_hospitals_g)

did_att_estimate = float(np.average(cs_att_gt, weights=cs_weights))


# ==========================================
# 3. TREATMENT HETEROGENEITY (T-LEARNER)
# ==========================================
feature_cols = [
    'age', 'comorbidity_score', 'immune_compromised', 'severity_score',
    'sex', 'ward', 'infection_type', 'hospital_size', 'region', 'hospital_id', 'year'
]
X_all = pd.get_dummies(
    df[feature_cols],
    columns=['sex', 'ward', 'infection_type', 'hospital_size', 'region', 'hospital_id', 'year'],
    drop_first=False,
    dtype=float
)
y_all = df['antibiotic_duration'].values
w_all = df['asp_active'].values

mask_treated = (w_all == 1)
mask_control = (w_all == 0)

model_1 = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
model_0 = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)

model_1.fit(X_all[mask_treated], y_all[mask_treated])
model_0.fit(X_all[mask_control], y_all[mask_control])

mu_1 = model_1.predict(X_all)
mu_0 = model_0.predict(X_all)
cate_estimates = np.asarray(mu_1 - mu_0, dtype=float)

# Bootstrap 95% CI for mean CATE(age >= 65) minus mean CATE(age < 40)
elderly_cates = cate_estimates[df['age'] >= 65]
young_cates = cate_estimates[df['age'] < 40]

rng = np.random.default_rng(42)
n_boot = 1000
boot_diffs = np.empty(n_boot, dtype=float)
for b in range(n_boot):
    e_sample = rng.choice(elderly_cates, size=len(elderly_cates), replace=True)
    y_sample = rng.choice(young_cates, size=len(young_cates), replace=True)
    boot_diffs[b] = np.mean(e_sample) - np.mean(y_sample)

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5))
]


# ==========================================
# 4. HIERARCHICAL MODELING
# ==========================================
fe_formula = (
    'antibiotic_duration ~ asp_active + age + C(sex) + comorbidity_score + '
    'immune_compromised + severity_score + C(infection_type) + C(ward) + C(year)'
)

mixed_mod = smf.mixedlm(fe_formula, data=df, groups=df['hospital_id']).fit(reml=False)
naive_ols = smf.ols(fe_formula, data=df).fit()

random_effect_variance = float(np.asarray(mixed_mod.cov_re)[0, 0])
residual_variance = float(mixed_mod.scale)
icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))

mixed_asp_coeff = float(mixed_mod.fe_params['asp_active'])

if not np.isnan(mixed_mod.aic):
    mixed_model_aic = float(mixed_mod.aic)
else:
    k_params = len(mixed_mod.fe_params) + 2
    mixed_model_aic = float(-2.0 * mixed_mod.llf + 2.0 * k_params)

naive_ols_aic = float(naive_ols.aic)