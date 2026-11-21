import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from linearmodels.panel import PanelOLS
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import norm

# --- 1. SURVIVAL ANALYSIS ---
# Define the formula for the Cox Proportional Hazards model.
# The lifelines formula interface handles categorical variables automatically.
formula_cox = "asp_active + age + C(sex) + comorbidity_score + immune_compromised + C(infection_type) + severity_score + C(hospital_size) + C(region)"

# Instantiate and fit the CoxPHFitter model, clustering standard errors by hospital.
cph = CoxPHFitter()
cph.fit(df,
        duration_col='time_to_improvement',
        event_col='event_observed',
        formula=formula_cox,
        cluster_col='hospital_id',
        robust=True)

# Extract the hazard ratio for asp_active and the standard error of its log-hazard coefficient.
cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.summary.loc['asp_active', 'se(coef)'])

# Test the proportional hazards assumption for all covariates.
# The check_assumptions method returns a list of DataFrames, one for each covariate.
ph_summary = cph.check_assumptions(df, show_plots=False, p_value_threshold=1.0)
ph_test_pvalues = {}
for result_df in ph_summary:
    covariate_name = result_df.index[0]
    p_value = result_df['p'].iloc[0]
    ph_test_pvalues[covariate_name] = float(p_value)


# --- 2. CAUSAL IMPACT (DiD) ---
# Create a hospital-year panel by averaging the outcome.
did_panel_full = df.groupby(['hospital_id', 'year'])['antibiotic_duration'].mean().reset_index()
hospital_info = df[['hospital_id', 'asp_adoption_year']].drop_duplicates()
did_panel_full = pd.merge(did_panel_full, hospital_info, on='hospital_id')

# Create the treatment variable for the panel.
did_panel_full['asp_active'] = (did_panel_full['year'] >= did_panel_full['asp_adoption_year']).astype(int)

# Define event_time relative to the adoption year. Use NaN for never-treated units.
did_panel_full['event_time'] = np.nan
treated_mask = did_panel_full['asp_adoption_year'] != 9999
did_panel_full.loc[treated_mask, 'event_time'] = did_panel_full.loc[treated_mask, 'year'] - did_panel_full.loc[treated_mask, 'asp_adoption_year']

# Create event time dummies, dropping t=-1 as the reference period.
event_dummies = pd.get_dummies(did_panel_full['event_time'], prefix='event', drop_first=False, dtype=int)
if 'event_-1.0' in event_dummies.columns:
    event_dummies = event_dummies.drop(columns=['event_-1.0'])

did_panel_reg = pd.concat([did_panel_full, event_dummies], axis=1)
did_panel_reg = did_panel_reg.set_index(['hospital_id', 'year'])

# Fit the event study model with two-way fixed effects (hospital and year).
exog_vars = [col for col in did_panel_reg.columns if col.startswith('event_')]
exog = sm.add_constant(did_panel_reg[exog_vars])
mod_es = PanelOLS(did_panel_reg['antibiotic_duration'], exog, entity_effects=True, time_effects=True)
res_es = mod_es.fit(cov_type='clustered', cluster_entity=True)

# Extract pre-trend test results for t=-2 (the period before the reference).
pre_trend_coeff = float(res_es.params.get('event_-2.0', np.nan))
pre_trend_pvalue = float(res_es.pvalues.get('event_-2.0', np.nan))
if 'event_-2.0' in res_es.conf_int().index:
    pre_trend_ci = res_es.conf_int().loc['event_-2.0'].tolist()
else:
    pre_trend_ci = [np.nan, np.nan]

# Create the event_study DataFrame for output.
event_study_res = pd.DataFrame({
    'coef': res_es.params,
    'ci_low': res_es.conf_int().iloc[:, 0],
    'ci_high': res_es.conf_int().iloc[:, 1]
}).reset_index()
event_study_res = event_study_res.rename(columns={'index': 'variable'})
event_study = event_study_res[event_study_res['variable'].str.startswith('event_')].copy()
event_study['event_time'] = event_study['variable'].str.replace('event_', '').astype(float).astype(int)
event_study = event_study[['event_time', 'coef', 'ci_low', 'ci_high']].sort_values('event_time').reset_index(drop=True)

# Fit the standard TWFE model.
mod_twfe = PanelOLS(did_panel_reg['antibiotic_duration'], sm.add_constant(did_panel_reg[['asp_active']]), entity_effects=True, time_effects=True)
res_twfe = mod_twfe.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = float(res_twfe.params['asp_active'])

# Estimate a robust ATT by averaging post-treatment event study coefficients,
# weighted by the number of treated units in each period.
post_treatment_periods = event_study[event_study['event_time'] >= 0]['event_time'].values
post_treatment_coeffs = event_study[event_study['event_time'] >= 0]['coef'].values
weights = did_panel_full['event_time'][did_panel_full['event_time'].isin(post_treatment_periods)].value_counts()
ordered_weights = [weights.get(t, 0) for t in post_treatment_periods]
if sum(ordered_weights) > 0:
    did_att_estimate = float(np.average(post_treatment_coeffs, weights=ordered_weights))
else:
    did_att_estimate = np.nan

# Finalize the panel DataFrame for output.
did_panel = did_panel_full[['hospital_id', 'year', 'antibiotic_duration']]


# --- 3. TREATMENT HETEROGENEITY (CATE) ---
# Prepare data with dummy variables for sklearn models.
df_cate = df.copy()
categorical_cols = ['hospital_size', 'region', 'ward', 'sex', 'age_group', 'infection_type']
df_cate = pd.get_dummies(df_cate, columns=categorical_cols, drop_first=True)

# Define features, treatment, and outcome.
y = df_cate['antibiotic_duration']
treatment = df_cate['asp_active']
covariates = [c for c in df_cate.columns if c not in ['patient_id', 'time_to_improvement', 'event_observed', 'antibiotic_duration', 'length_of_stay', 'readmission_30d', 'c_diff_infection', 'mortality', 'asp_adoption_year', 'asp_active']]
X = df_cate[covariates]

# T-Learner implementation with Gradient Boosting Regressors.
X_treat, y_treat = X[treatment == 1], y[treatment == 1]
X_control, y_control = X[treatment == 0], y[treatment == 0]

m1 = GradientBoostingRegressor(random_state=42, n_estimators=100)
m0 = GradientBoostingRegressor(random_state=42, n_estimators=100)
m1.fit(X_treat, y_treat)
m0.fit(X_control, y_control)

cate_estimates = m1.predict(X) - m0.predict(X)

# Bootstrap a 95% CI for the difference in CATE between elderly (>=65) and young (<40).
n_bootstraps = 200
diffs = []
original_indices = df.index
np.random.seed(42) # for reproducibility
for _ in range(n_bootstraps):
    boot_indices = np.random.choice(original_indices, size=len(df), replace=True)
    boot_cate = cate_estimates[boot_indices]
    boot_age = df['age'].iloc[boot_indices]
    
    elderly_mask = boot_age >= 65
    young_mask = boot_age < 40
    
    if elderly_mask.sum() > 0 and young_mask.sum() > 0:
        mean_elderly = np.mean(boot_cate[elderly_mask])
        mean_young = np.mean(boot_cate[young_mask])
        diffs.append(mean_elderly - mean_young)

cate_elderly_minus_young_ci = np.percentile(diffs, [2.5, 97.5]).tolist()


# --- 4. HIERARCHICAL MODELING ---
# Define a common formula for the fixed effects part of the models.
base_covariates = "age + C(sex) + comorbidity_score + immune_compromised + C(infection_type) + severity_score + C(hospital_size) + C(region) + C(ward) + C(year)"
formula = f"antibiotic_duration ~ asp_active + {base_covariates}"

# Fit the mixed-effects model with hospital as a random intercept, using ML for AIC comparison.
mixed_model = smf.mixedlm(formula, data=df, groups=df['hospital_id'])
mixed_results = mixed_model.fit(method='ML', reml=False)

# Extract results from the mixed model.
mixed_model_aic = float(mixed_results.aic)
mixed_asp_coeff = float(mixed_results.params['asp_active'])
random_effect_variance = float(mixed_results.cov_re.iloc[0, 0])
residual_variance = float(mixed_results.scale)
icc_value = random_effect_variance / (random_effect_variance + residual_variance)

# Fit the naive OLS model with the same fixed effects for AIC comparison.
ols_model = smf.ols(formula, data=df)
ols_results = ols_model.fit()
naive_ols_aic = float(ols_results.aic)