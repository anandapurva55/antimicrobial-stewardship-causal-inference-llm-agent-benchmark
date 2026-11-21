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
             'immune_compromised', 'severity_score', 'hospital_id', 'sex', 'ward', 'infection_type']].copy()
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
    asp_active=('asp_active', 'mean'),
    asp_adoption_year=('asp_adoption_year', 'first'),
    age=('age', 'mean'),
    severity_score=('severity_score', 'mean'),
    comorbidity_score=('comorbidity_score', 'mean')
)
did_panel['asp_active'] = (did_panel['asp_active'] > 0.5).astype(int)

# Plain Two-Way Fixed Effects (TWFE) on hospital-year panel
twfe_mod = smf.ols('antibiotic_duration ~ asp_active + C(hospital_id) + C(year)', data=did_panel).fit(
    cov_type='cluster', cov_kwds={'groups': did_panel['hospital_id']}
)
twfe_att_estimate = float(twfe_mod.params['asp_active'])

# Event Study & Parallel Pre-Trends Test
did_panel['rel_time'] = np.where(
    did_panel['asp_adoption_year'] == 9999, 
    -1, 
    did_panel['year'] - did_panel['asp_adoption_year']
)

rel_times = sorted([int(t) for t in did_panel['rel_time'].unique() if t != -1])
es_terms = []
for t in rel_times:
    col_name = f"evt_m{abs(t)}" if t < 0 else f"evt_p{t}"
    did_panel[col_name] = (did_panel['rel_time'] == t).astype(int)
    es_terms.append((t, col_name))

es_formula = 'antibiotic_duration ~ ' + ' + '.join([c for _, c in es_terms]) + ' + C(hospital_id) + C(year)'
es_mod = smf.ols(es_formula, data=did_panel).fit(
    cov_type='cluster', cov_kwds={'groups': did_panel['hospital_id']}
)

es_rows = []
es_ci = es_mod.conf_int()
for t, col_name in es_terms:
    es_rows.append({
        'event_time': int(t),
        'coef': float(es_mod.params[col_name]),
        'ci_low': float(es_ci.loc[col_name, 0]),
        'ci_high': float(es_ci.loc[col_name, 1])
    })
es_rows.append({'event_time': -1, 'coef': 0.0, 'ci_low': 0.0, 'ci_high': 0.0})
event_study = pd.DataFrame(es_rows).sort_values('event_time').reset_index(drop=True)

lead_col = 'evt_m2' if 'evt_m2' in es_mod.params else [c for t, c in es_terms if t < -1][-1]
pre_trend_coeff = float(es_mod.params[lead_col])
pre_trend_pvalue = float(es_mod.pvalues[lead_col])
pre_trend_ci = [float(es_ci.loc[lead_col, 0]), float(es_ci.loc[lead_col, 1])]

# Staggered-Adoption Robust Estimator: Callaway-Sant'Anna (using never-treated as controls)
cohorts = [2020, 2021]
never_treated = did_panel[did_panel['asp_adoption_year'] == 9999]
cs_att_list = []
cs_weights = []

for g in cohorts:
    base_year = g - 1
    cohort_df = did_panel[did_panel['asp_adoption_year'] == g]
    n_hospitals_g = cohort_df['hospital_id'].nunique()
    
    y_treat_base = cohort_df[cohort_df['year'] == base_year]['antibiotic_duration'].mean()
    y_ctrl_base = never_treated[never_treated['year'] == base_year]['antibiotic_duration'].mean()
    
    post_years = [y for y in sorted(did_panel['year'].unique()) if y >= g]
    for t in post_years:
        y_treat_t = cohort_df[cohort_df['year'] == t]['antibiotic_duration'].mean()
        y_ctrl_t = never_treated[never_treated['year'] == t]['antibiotic_duration'].mean()
        
        att_gt = (y_treat_t - y_treat_base) - (y_ctrl_t - y_ctrl_base)
        cs_att_list.append(att_gt)
        cs_weights.append(n_hospitals_g)

did_att_estimate = float(np.average(cs_att_list, weights=cs_weights))

# ==========================================
# 3. TREATMENT HETEROGENEITY (T-Learner)
# ==========================================
feature_cols = ['age', 'comorbidity_score', 'immune_compromised', 'severity_score', 
                'sex', 'ward', 'infection_type', 'hospital_size', 'region', 'hospital_id', 'year']
X_raw = df[feature_cols].copy()
X_encoded = pd.get_dummies(
    X_raw, 
    columns=['sex', 'ward', 'infection_type', 'hospital_size', 'region', 'hospital_id', 'year'], 
    drop_first=True,
    dtype=float
)

y_outcome = df['antibiotic_duration'].values
w_treat = df['asp_active'].values

model_0 = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
model_1 = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)

model_0.fit(X_encoded[w_treat == 0], y_outcome[w_treat == 0])
model_1.fit(X_encoded[w_treat == 1], y_outcome[w_treat == 1])

mu_0 = model_0.predict(X_encoded)
mu_1 = model_1.predict(X_encoded)
cate_estimates = np.asarray(mu_1 - mu_0, dtype=float)

elderly_idx = np.where(df['age'] >= 65)[0]
young_idx = np.where(df['age'] < 40)[0]

rng = np.random.default_rng(42)
boot_diffs = []
for _ in range(1000):
    b_eld = rng.choice(elderly_idx, size=len(elderly_idx), replace=True)
    b_yng = rng.choice(young_idx, size=len(young_idx), replace=True)
    boot_diffs.append(np.mean(cate_estimates[b_eld]) - np.mean(cate_estimates[b_yng]))

cate_elderly_minus_young_ci = [float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))]

# ==========================================
# 4. HIERARCHICAL MODELING
# ==========================================
fe_formula = ('antibiotic_duration ~ asp_active + age + comorbidity_score + '
              'immune_compromised + severity_score + C(sex) + C(ward) + C(infection_type) + C(year)')

mixed_mod = smf.mixedlm(fe_formula, data=df, groups=df['hospital_id']).fit(reml=False)
naive_ols = smf.ols(fe_formula, data=df).fit()

random_effect_variance = float(mixed_mod.cov_re.iloc[0, 0])
residual_variance = float(mixed_mod.scale)
icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))

mixed_model_aic = float(mixed_mod.aic)
naive_ols_aic = float(naive_ols.aic)
mixed_asp_coeff = float(mixed_mod.params['asp_active'])