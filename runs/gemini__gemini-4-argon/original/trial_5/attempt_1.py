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
kmf_active = KaplanMeierFitter()
kmf_inactive = KaplanMeierFitter()

mask_active = df['asp_active'] == 1
kmf_active.fit(df.loc[mask_active, 'time_to_improvement'], df.loc[mask_active, 'event_observed'], label='ASP Active')
kmf_inactive.fit(df.loc[~mask_active, 'time_to_improvement'], df.loc[~mask_active, 'event_observed'], label='ASP Inactive')

cox_df = df[['time_to_improvement', 'event_observed', 'asp_active', 'age', 'comorbidity_score', 
             'immune_compromised', 'severity_score', 'sex', 'ward', 'infection_type', 'hospital_id']].copy()
cox_df = pd.get_dummies(cox_df, columns=['sex', 'ward', 'infection_type'], drop_first=True, dtype=float)

cph = CoxPHFitter()
cph.fit(cox_df, duration_col='time_to_improvement', event_col='event_observed', cluster_col='hospital_id')

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.standard_errors_['asp_active'])

ph_test = proportional_hazard_test(cph, cox_df, time_transform='rank')
ph_test_pvalues = {str(k): float(v) for k, v in ph_test.summary['p'].to_dict().items()}

# ==========================================
# 2. CAUSAL IMPACT (DiD)
# ==========================================
did_panel = df.groupby(['hospital_id', 'year'], as_index=False).agg(
    antibiotic_duration=('antibiotic_duration', 'mean'),
    asp_active=('asp_active', 'max'),
    asp_adoption_year=('asp_adoption_year', 'first'),
    age=('age', 'mean'),
    comorbidity_score=('comorbidity_score', 'mean'),
    severity_score=('severity_score', 'mean')
)

# Plain Two-Way Fixed Effects (TWFE) on hospital-year panel
twfe_mod = smf.ols('antibiotic_duration ~ asp_active + C(hospital_id) + C(year)', data=did_panel).fit(
    cov_type='cluster', cov_kwds={'groups': did_panel['hospital_id']}
)
twfe_att_estimate = float(twfe_mod.params['asp_active'])

# Staggered adoption robust estimator (Callaway-Sant'Anna style cohort-time 2x2 aggregation + Event Study)
did_panel['ever_treated'] = (did_panel['asp_adoption_year'] != 9999).astype(int)
did_panel['event_time'] = np.where(
    did_panel['ever_treated'] == 1,
    did_panel['year'] - did_panel['asp_adoption_year'],
    -1
)

event_times = sorted([int(et) for et in did_panel.loc[did_panel['ever_treated'] == 1, 'event_time'].unique() if et != -1])
es_formula_terms = []
for et in event_times:
    col_name = f"et_m{abs(et)}" if et < 0 else f"et_p{et}"
    did_panel[col_name] = ((did_panel['ever_treated'] == 1) & (did_panel['event_time'] == et)).astype(int)
    es_formula_terms.append((et, col_name))

es_rhs = " + ".join([col for _, col in es_formula_terms])
es_mod = smf.ols(f"antibiotic_duration ~ {es_rhs} + C(hospital_id) + C(year)", data=did_panel).fit(
    cov_type='cluster', cov_kwds={'groups': did_panel['hospital_id']}
)

es_rows = []
for et, col in es_formula_terms:
    coef = float(es_mod.params[col])
    ci_low, ci_high = es_mod.conf_int().loc[col]
    es_rows.append({'event_time': int(et), 'coef': coef, 'ci_low': float(ci_low), 'ci_high': float(ci_high)})

event_study = pd.DataFrame(es_rows).sort_values('event_time').reset_index(drop=True)

# Extract pre-trend validation metrics from lead event_time = -2
lead_col = "et_m2"
pre_trend_coeff = float(es_mod.params[lead_col])
pre_trend_pvalue = float(es_mod.pvalues[lead_col])
pre_trend_ci = [float(es_mod.conf_int().loc[lead_col, 0]), float(es_mod.conf_int().loc[lead_col, 1])]

# Callaway-Sant'Anna cohort-time ATT(g, t) using never-treated (9999) as controls
cohorts = [2020, 2021]
never_treated = did_panel[did_panel['asp_adoption_year'] == 9999]
att_gt_list = []
weights_list = []

for g in cohorts:
    base_year = g - 1
    cohort_df = did_panel[did_panel['asp_adoption_year'] == g]
    n_hospitals_g = cohort_df['hospital_id'].nunique()
    
    y_treat_base = cohort_df.loc[cohort_df['year'] == base_year, 'antibiotic_duration'].mean()
    y_ctrl_base = never_treated.loc[never_treated['year'] == base_year, 'antibiotic_duration'].mean()
    
    for t in [y for y in sorted(did_panel['year'].unique()) if y >= g]:
        y_treat_t = cohort_df.loc[cohort_df['year'] == t, 'antibiotic_duration'].mean()
        y_ctrl_t = never_treated.loc[never_treated['year'] == t, 'antibiotic_duration'].mean()
        
        att_gt = (y_treat_t - y_treat_base) - (y_ctrl_t - y_ctrl_base)
        att_gt_list.append(att_gt)
        weights_list.append(n_hospitals_g)

did_att_estimate = float(np.average(att_gt_list, weights=weights_list))

# ==========================================
# 3. TREATMENT HETEROGENEITY (T-Learner)
# ==========================================
feature_cols = ['age', 'comorbidity_score', 'immune_compromised', 'severity_score', 
                'sex', 'ward', 'infection_type', 'hospital_id', 'year']
X_raw = df[feature_cols].copy()
X_encoded = pd.get_dummies(X_raw, columns=['sex', 'ward', 'infection_type', 'hospital_id', 'year'], drop_first=True, dtype=float)

Y = df['antibiotic_duration'].values
T = df['asp_active'].values

m0 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
m1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)

m0.fit(X_encoded[T == 0], Y[T == 0])
m1.fit(X_encoded[T == 1], Y[T == 1])

cate_estimates = m1.predict(X_encoded) - m0.predict(X_encoded)

# Bootstrap 95% CI for mean CATE(age>=65) minus mean CATE(age<40)
mask_elderly = (df['age'] >= 65).values
mask_young = (df['age'] < 40).values

cate_elderly = cate_estimates[mask_elderly]
cate_young = cate_estimates[mask_young]

rng = np.random.default_rng(42)
n_boot = 1000
boot_diffs = np.empty(n_boot)
for b in range(n_boot):
    b_eld = rng.choice(cate_elderly, size=len(cate_elderly), replace=True).mean()
    b_yng = rng.choice(cate_young, size=len(cate_young), replace=True).mean()
    boot_diffs[b] = b_eld - b_yng

cate_elderly_minus_young_ci = [float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))]

# ==========================================
# 4. HIERARCHICAL MODELING
# ==========================================
fe_formula = (
    "antibiotic_duration ~ asp_active + age + comorbidity_score + immune_compromised + "
    "severity_score + C(sex) + C(ward) + C(infection_type) + C(year)"
)

# Mixed-effects linear model with hospital_id as random intercept (ML fit for valid AIC comparison)
mixed_mod = smf.mixedlm(fe_formula, data=df, groups=df['hospital_id']).fit(reml=False)

random_effect_variance = float(mixed_mod.cov_re.iloc[0, 0])
residual_variance = float(mixed_mod.scale)
icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))

mixed_model_aic = float(mixed_mod.aic)
mixed_asp_coeff = float(mixed_mod.fe_params['asp_active'])

# Naive OLS with identical fixed effects
naive_ols = smf.ols(fe_formula, data=df).fit()
naive_ols_aic = float(naive_ols.aic)