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
kmf_asp = KaplanMeierFitter()
kmf_no_asp = KaplanMeierFitter()

mask_asp = df['asp_active'] == 1
kmf_asp.fit(df.loc[mask_asp, 'time_to_improvement'], event_observed=df.loc[mask_asp, 'event_observed'], label='ASP Active')
kmf_no_asp.fit(df.loc[~mask_asp, 'time_to_improvement'], event_observed=df.loc[~mask_asp, 'event_observed'], label='No ASP')

cox_cols = ['time_to_improvement', 'event_observed', 'hospital_id', 'asp_active', 
            'age', 'comorbidity_score', 'immune_compromised', 'severity_score', 
            'sex', 'ward', 'infection_type']
df_cox = pd.get_dummies(df[cox_cols], columns=['sex', 'ward', 'infection_type'], drop_first=True, dtype=float)

cph = CoxPHFitter()
cph.fit(df_cox, duration_col='time_to_improvement', event_col='event_observed', cluster_col='hospital_id')

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.standard_errors_['asp_active'])

ph_test = proportional_hazard_test(cph, df_cox, time_transform='rank')
ph_test_pvalues = {str(k): float(v) for k, v in ph_test.summary['p'].to_dict().items()}


# ==============================================================================
# 2. CAUSAL IMPACT (DiD & Event Study)
# ==============================================================================
did_panel = df.groupby(['hospital_id', 'year'], as_index=False).agg(
    antibiotic_duration=('antibiotic_duration', 'mean'),
    asp_active=('asp_active', 'mean'),
    asp_adoption_year=('asp_adoption_year', 'first'),
    age=('age', 'mean'),
    comorbidity_score=('comorbidity_score', 'mean'),
    severity_score=('severity_score', 'mean')
)
did_panel['asp_active'] = (did_panel['asp_active'] > 0.5).astype(int)
did_panel['treated_ever'] = (did_panel['asp_adoption_year'] != 9999).astype(int)

# Two-Way Fixed Effects (TWFE) on hospital-year panel
twfe_mod = smf.ols('antibiotic_duration ~ asp_active + C(hospital_id) + C(year)', data=did_panel).fit(
    cov_type='cluster', cov_kwds={'groups': did_panel['hospital_id']}
)
twfe_att_estimate = float(twfe_mod.params['asp_active'])

# Event Study specification
did_panel['rel_time'] = np.where(
    did_panel['treated_ever'] == 1,
    did_panel['year'] - did_panel['asp_adoption_year'],
    -1
)

event_times = sorted([int(t) for t in did_panel.loc[did_panel['treated_ever'] == 1, 'rel_time'].unique() if t != -1])
es_terms = []
for t in event_times:
    col_name = f"ev_m{abs(t)}" if t < 0 else f"ev_p{t}"
    did_panel[col_name] = ((did_panel['treated_ever'] == 1) & (did_panel['rel_time'] == t)).astype(int)
    es_terms.append((t, col_name))

es_formula = 'antibiotic_duration ~ ' + ' + '.join([c for _, c in es_terms]) + ' + C(hospital_id) + C(year)'
es_mod = smf.ols(es_formula, data=did_panel).fit(
    cov_type='cluster', cov_kwds={'groups': did_panel['hospital_id']}
)

es_rows = [{'event_time': -1, 'coef': 0.0, 'ci_low': 0.0, 'ci_high': 0.0}]
es_ci = es_mod.conf_int()
for t, col_name in es_terms:
    es_rows.append({
        'event_time': int(t),
        'coef': float(es_mod.params[col_name]),
        'ci_low': float(es_ci.loc[col_name, 0]),
        'ci_high': float(es_ci.loc[col_name, 1])
    })
event_study = pd.DataFrame(es_rows).sort_values('event_time').reset_index(drop=True)

# Pre-trend test using immediate lead (t = -2 vs reference t = -1)
lead_col = 'ev_m2' if 'ev_m2' in [c for _, c in es_terms] else [c for t, c in es_terms if t < 0][-1]
pre_trend_coeff = float(es_mod.params[lead_col])
pre_trend_pvalue = float(es_mod.pvalues[lead_col])
pre_trend_ci = [float(es_ci.loc[lead_col, 0]), float(es_ci.loc[lead_col, 1])]

# Callaway-Sant'Anna / Sun-Abraham Cohort-Time ATT aggregation (using never-treated as controls)
never_treated = did_panel[did_panel['asp_adoption_year'] == 9999]
cohorts = sorted([c for c in did_panel['asp_adoption_year'].unique() if c != 9999])

att_gt_list = []
weight_list = []

for g in cohorts:
    cohort_df = did_panel[did_panel['asp_adoption_year'] == g]
    base_year = g - 1
    y_treat_base = cohort_df.loc[cohort_df['year'] == base_year, 'antibiotic_duration'].mean()
    y_ctrl_base = never_treated.loc[never_treated['year'] == base_year, 'antibiotic_duration'].mean()
    
    post_years = sorted([y for y in cohort_df['year'].unique() if y >= g])
    for t in post_years:
        y_treat_t = cohort_df.loc[cohort_df['year'] == t, 'antibiotic_duration'].mean()
        y_ctrl_t = never_treated.loc[never_treated['year'] == t, 'antibiotic_duration'].mean()
        
        att_gt = (y_treat_t - y_treat_base) - (y_ctrl_t - y_ctrl_base)
        n_gt = len(df[(df['asp_adoption_year'] == g) & (df['year'] == t)])
        
        att_gt_list.append(att_gt)
        weight_list.append(n_gt)

did_att_estimate = float(np.average(att_gt_list, weights=weight_list))


# ==============================================================================
# 3. TREATMENT HETEROGENEITY (T-Learner with Gradient Boosting)
# ==============================================================================
feature_cols = ['age', 'comorbidity_score', 'immune_compromised', 'severity_score', 
                'sex', 'ward', 'infection_type', 'hospital_size', 'region', 'hospital_id', 'year']
X_all = pd.get_dummies(df[feature_cols], columns=['sex', 'ward', 'infection_type', 'hospital_size', 'region', 'hospital_id', 'year'], drop_first=True, dtype=float)
y_all = df['antibiotic_duration'].values
t_all = df['asp_active'].values

model_0 = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
model_1 = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)

model_0.fit(X_all[t_all == 0], y_all[t_all == 0])
model_1.fit(X_all[t_all == 1], y_all[t_all == 1])

mu_0 = model_0.predict(X_all)
mu_1 = model_1.predict(X_all)
cate_estimates = np.asarray(mu_1 - mu_0, dtype=float)

# Bootstrap 95% CI for mean CATE(age >= 65) minus mean CATE(age < 40)
elderly_cates = cate_estimates[df['age'] >= 65]
young_cates = cate_estimates[df['age'] < 40]

rng = np.random.default_rng(42)
n_boot = 1000
boot_diffs = np.empty(n_boot, dtype=float)
n_eld = len(elderly_cates)
n_yng = len(young_cates)

for b in range(n_boot):
    eld_sample = rng.choice(elderly_cates, size=n_eld, replace=True)
    yng_sample = rng.choice(young_cates, size=n_yng, replace=True)
    boot_diffs[b] = np.mean(eld_sample) - np.mean(yng_sample)

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5))
]


# ==============================================================================
# 4. HIERARCHICAL MODELING (Mixed-Effects Linear Model)
# ==============================================================================
fe_formula = ('antibiotic_duration ~ asp_active + age + comorbidity_score + '
              'immune_compromised + severity_score + C(sex) + C(ward) + '
              'C(infection_type) + C(year)')

# Naive OLS
ols_mod = smf.ols(fe_formula, data=df).fit()
naive_ols_aic = float(ols_mod.aic)

# Mixed-Effects Model (ML fit for valid AIC comparison)
mixed_mod = smf.mixedlm(fe_formula, data=df, groups=df['hospital_id']).fit(reml=False)

random_effect_variance = float(mixed_mod.cov_re.iloc[0, 0])
residual_variance = float(mixed_mod.scale)
icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))

mixed_asp_coeff = float(mixed_mod.fe_params['asp_active'])
if not np.isnan(mixed_mod.aic):
    mixed_model_aic = float(mixed_mod.aic)
else:
    k_params = len(mixed_mod.fe_params) + 2
    mixed_model_aic = float(-2.0 * mixed_mod.llf + 2.0 * k_params)