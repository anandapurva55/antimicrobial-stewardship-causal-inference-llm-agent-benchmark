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
kmf_active = KaplanMeierFitter()
kmf_inactive = KaplanMeierFitter()

mask_active = df['asp_active'] == 1
kmf_active.fit(df.loc[mask_active, 'time_to_improvement'], event_observed=df.loc[mask_active, 'event_observed'], label='ASP Active')
kmf_inactive.fit(df.loc[~mask_active, 'time_to_improvement'], event_observed=df.loc[~mask_active, 'event_observed'], label='ASP Inactive')

cox_df = df[['time_to_improvement', 'event_observed', 'hospital_id', 'asp_active', 
             'age', 'sex', 'comorbidity_score', 'immune_compromised', 'severity_score', 
             'ward', 'infection_type']].copy()
cox_df = pd.get_dummies(cox_df, columns=['sex', 'ward', 'infection_type'], drop_first=True, dtype=float)

cph = CoxPHFitter()
cph.fit(cox_df, duration_col='time_to_improvement', event_col='event_observed', cluster_col='hospital_id')

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.standard_errors_['asp_active'])

ph_test = proportional_hazard_test(cph, cox_df, time_transform='rank')
ph_test_pvalues = {str(k): float(v) for k, v in ph_test.summary['p'].to_dict().items()}

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
did_panel['treated_ever'] = (did_panel['asp_adoption_year'] != 9999).astype(int)
did_panel['rel_time'] = np.where(
    did_panel['treated_ever'] == 1,
    did_panel['year'] - did_panel['asp_adoption_year'],
    -1
)

# Plain TWFE on hospital-year panel
twfe_mod = smf.ols('antibiotic_duration ~ asp_active + C(hospital_id) + C(year)', data=did_panel).fit(
    cov_type='cluster', cov_kwds={'groups': did_panel['hospital_id']}
)
twfe_att_estimate = float(twfe_mod.params['asp_active'])

# Event study / dynamic TWFE
rel_times = sorted([int(rt) for rt in did_panel['rel_time'].unique() if rt != -1])
es_terms = []
for rt in rel_times:
    col_name = f"rt_m{abs(rt)}" if rt < 0 else f"rt_p{rt}"
    did_panel[col_name] = (did_panel['rel_time'] == rt).astype(int)
    es_terms.append((rt, col_name))

es_formula = 'antibiotic_duration ~ ' + ' + '.join([c for _, c in es_terms]) + ' + C(hospital_id) + C(year)'
es_mod = smf.ols(es_formula, data=did_panel).fit(
    cov_type='cluster', cov_kwds={'groups': did_panel['hospital_id']}
)

es_rows = []
for rt, col_name in es_terms:
    coef = float(es_mod.params[col_name])
    ci_l, ci_h = es_mod.conf_int().loc[col_name]
    es_rows.append({'event_time': int(rt), 'coef': coef, 'ci_low': float(ci_l), 'ci_high': float(ci_h)})
es_rows.append({'event_time': -1, 'coef': 0.0, 'ci_low': 0.0, 'ci_high': 0.0})
event_study = pd.DataFrame(es_rows).sort_values('event_time').reset_index(drop=True)

# Pre-trend test using the t = -2 lead coefficient
lead_col = 'rt_m2'
pre_trend_coeff = float(es_mod.params[lead_col])
pre_trend_pvalue = float(es_mod.pvalues[lead_col])
pre_trend_ci = [float(x) for x in es_mod.conf_int().loc[lead_col]]

# Staggered-robust DiD (Callaway-Sant'Anna / Sun-Abraham style cohort-time ATT aggregation)
cohorts = [2020, 2021]
never_treated = did_panel[did_panel['asp_adoption_year'] == 9999]

att_gt_list = []
weights_list = []

for g in cohorts:
    cohort_df = did_panel[did_panel['asp_adoption_year'] == g]
    base_year = g - 1
    y_treat_base = cohort_df.loc[cohort_df['year'] == base_year, 'antibiotic_duration'].mean()
    y_ctrl_base = never_treated.loc[never_treated['year'] == base_year, 'antibiotic_duration'].mean()
    
    post_years = [y for y in sorted(did_panel['year'].unique()) if y >= g]
    for t in post_years:
        y_treat_t = cohort_df.loc[cohort_df['year'] == t, 'antibiotic_duration'].mean()
        y_ctrl_t = never_treated.loc[never_treated['year'] == t, 'antibiotic_duration'].mean()
        
        att_gt = (y_treat_t - y_treat_base) - (y_ctrl_t - y_ctrl_base)
        n_gt = len(cohort_df[cohort_df['year'] == t])
        
        att_gt_list.append(att_gt)
        weights_list.append(n_gt)

did_att_estimate = float(np.average(att_gt_list, weights=weights_list))

# ==============================================================================
# 3. TREATMENT HETEROGENEITY (T-Learner)
# ==============================================================================
feature_cols = ['age', 'sex', 'comorbidity_score', 'immune_compromised', 
                'severity_score', 'ward', 'infection_type', 'hospital_size', 
                'region', 'hospital_id', 'year']

X_raw = pd.get_dummies(df[feature_cols], columns=['sex', 'ward', 'infection_type', 'hospital_size', 'region', 'hospital_id', 'year'], drop_first=True, dtype=float)
Y_raw = df['antibiotic_duration'].values
T_raw = df['asp_active'].values

m0 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
m1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)

m0.fit(X_raw[T_raw == 0], Y_raw[T_raw == 0])
m1.fit(X_raw[T_raw == 1], Y_raw[T_raw == 1])

cate_estimates = m1.predict(X_raw) - m0.predict(X_raw)

elderly_mask = (df['age'] >= 65).values
young_mask = (df['age'] < 40).values

cate_elderly = cate_estimates[elderly_mask]
cate_young = cate_estimates[young_mask]

rng = np.random.default_rng(42)
n_boot = 1000
boot_diffs = np.empty(n_boot)
for b in range(n_boot):
    e_sample = rng.choice(cate_elderly, size=len(cate_elderly), replace=True)
    y_sample = rng.choice(cate_young, size=len(cate_young), replace=True)
    boot_diffs[b] = np.mean(e_sample) - np.mean(y_sample)

cate_elderly_minus_young_ci = [float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))]

# ==============================================================================
# 4. HIERARCHICAL MODELING
# ==============================================================================
fe_formula = 'antibiotic_duration ~ asp_active + age + C(sex) + comorbidity_score + immune_compromised + severity_score + C(ward) + C(infection_type) + C(year)'

naive_ols = smf.ols(fe_formula, data=df).fit()
naive_ols_aic = float(naive_ols.aic)

mixed_mod = smf.mixedlm(fe_formula, data=df, groups=df['hospital_id']).fit(reml=False)
mixed_asp_coeff = float(mixed_mod.fe_params['asp_active'])
random_effect_variance = float(mixed_mod.cov_re.iloc[0, 0])
residual_variance = float(mixed_mod.scale)
icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))

if not np.isnan(mixed_mod.aic):
    mixed_model_aic = float(mixed_mod.aic)
else:
    k_params = len(mixed_mod.fe_params) + 2
    mixed_model_aic = float(-2.0 * mixed_mod.llf + 2.0 * k_params)