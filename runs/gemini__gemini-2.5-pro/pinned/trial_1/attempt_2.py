import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
from linearmodels.panel import PanelOLS
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import norm

# Objective 1: Survival Analysis

# Prepare data for Cox model
# Use a subset of covariates to avoid convergence issues and for interpretability
df_cox = df.copy()
# Dummify infection_type as it's a key clinical factor
df_cox = pd.get_dummies(df_cox, columns=['infection_type'], drop_first=True, dtype=float)
covariates_cox = ['asp_active', 'age', 'comorbidity_score', 'severity_score', 'immune_compromised'] + [col for col in df_cox.columns if 'infection_type_' in col]

# Fit the Cox Proportional Hazards model with clustered standard errors
cph = CoxPHFitter()
cph.fit(df_cox,
        duration_col='time_to_improvement',
        event_col='event_observed',
        formula=" + ".join(covariates_cox),
        cluster_col='hospital_id')

# Extract required outputs
cox_hr_asp = cph.hazard_ratios_['asp_active']
cox_se_asp = cph.summary.loc['asp_active', 'se(coef)']

# Test the Proportional Hazards assumption
ph_test_results = cph.check_assumptions(df_cox, p_value_threshold=0.05, show_plots=False)
# The check_assumptions method returns a list of dataframes, one for each variable.
ph_test_pvalues = {res.index[0]: res['p'].iloc[0] for res in ph_test_results}


# Objective 2: Causal Impact (DiD)

# Create hospital-year panel
did_panel = df.groupby(['hospital_id', 'year'])['antibiotic_duration'].mean().reset_index()
hospital_info = df[['hospital_id', 'asp_adoption_year']].drop_duplicates()
did_panel = pd.merge(did_panel, hospital_info, on='hospital_id')

# Prepare for event study
did_panel['treat'] = (did_panel['asp_adoption_year'] != 9999).astype(int)
did_panel['post'] = (did_panel['year'] >= did_panel['asp_adoption_year']).astype(int)
did_panel['event_time'] = np.where(did_panel['treat'] == 1, did_panel['year'] - did_panel['asp_adoption_year'], np.nan)

# Bin distant event times to have enough observations per period and avoid perfect collinearity
did_panel['event_time'] = did_panel['event_time'].apply(lambda x: -3 if x <= -3 else (2 if x >= 2 else x))

# Create dummy variables for event_time, with t=-1 as the reference period
event_dummies = pd.get_dummies(did_panel['event_time'], prefix='t', drop_first=False, dtype=float)
if 't_-1.0' in event_dummies.columns:
    event_dummies = event_dummies.drop(columns=['t_-1.0'])
did_panel_es = pd.concat([did_panel.set_index(['hospital_id', 'year']), event_dummies], axis=1)

# Run the event study regression using only the 'ever-treated' sample
# This is a common way to implement the Sun & Abraham (2021) estimator
ever_treated_panel = did_panel_es[did_panel_es['treat'] == 1]
dummy_cols = [col for col in ever_treated_panel.columns if col.startswith('t_')]
formula = 'antibiotic_duration ~ 1 + ' + ' + '.join(dummy_cols) + ' + EntityEffects + TimeEffects'
mod = PanelOLS.from_formula(formula, data=ever_treated_panel, drop_absorbed=True)
es_results = mod.fit(cov_type='clustered', cluster_entity=True)

# Extract event study results
event_study_params = es_results.params.filter(like='t_')
event_study_cis = es_results.conf_int().filter(like='t_', axis=0)
event_study = pd.DataFrame({
    'coef': event_study_params,
    'ci_low': event_study_cis['lower'],
    'ci_high': event_study_cis['upper']
}).reset_index()
event_study['event_time'] = event_study['index'].str.replace('t_', '').astype(float).astype(int)
event_study = event_study.drop(columns='index').sort_values('event_time')

# Test for parallel pre-trends
# Check the coefficient for a pre-treatment period, e.g., t=-2
pre_trend_coeff_name = 't_-2.0'
if pre_trend_coeff_name in es_results.params:
    pre_trend_coeff = es_results.params[pre_trend_coeff_name]
    pre_trend_pvalue = es_results.pvalues[pre_trend_coeff_name]
    pre_trend_ci = es_results.conf_int().loc[pre_trend_coeff_name].tolist()
else: # In case t=-2 is not in the model for some reason
    pre_trend_coeff = np.nan
    pre_trend_pvalue = np.nan
    pre_trend_ci = [np.nan, np.nan]


# Run classic TWFE model
did_panel_twfe = did_panel.copy().set_index(['hospital_id', 'year'])
did_panel_twfe['treated_post'] = did_panel_twfe['treat'] * did_panel_twfe['post']
twfe_mod = PanelOLS.from_formula('antibiotic_duration ~ 1 + treated_post + EntityEffects + TimeEffects', data=did_panel_twfe)
twfe_res = twfe_mod.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = twfe_res.params['treated_post']

# Calculate robust ATT from event study results (weighted average of post-treatment coefficients)
post_treatment_coeffs = event_study[event_study['event_time'] >= 0]
weights = ever_treated_panel['event_time'].value_counts()
weighted_sum = 0
total_weight = 0
for _, row in post_treatment_coeffs.iterrows():
    t = row['event_time']
    if t in weights.index:
        weight = weights[t]
        weighted_sum += row['coef'] * weight
        total_weight += weight
did_att_estimate = weighted_sum / total_weight if total_weight > 0 else np.nan


# Objective 3: Treatment Heterogeneity (CATE)

# Prepare data for CATE estimation
df_cate = df.copy()
categorical_features = ['hospital_size', 'region', 'ward', 'sex', 'age_group', 'infection_type']
df_cate = pd.get_dummies(df_cate, columns=categorical_features, drop_first=True, dtype=float)
# Include hospital_id and year as fixed effects by dummifying them
df_cate = pd.get_dummies(df_cate, columns=['hospital_id', 'year'], drop_first=True, dtype=float)

T = df_cate['asp_active']
Y = df_cate['antibiotic_duration']
X_cols = [col for col in df_cate.columns if col not in ['patient_id', 'asp_active', 'antibiotic_duration', 'asp_adoption_year', 'time_to_improvement', 'event_observed', 'length_of_stay', 'readmission_30d', 'c_diff_infection', 'mortality']]
X = df_cate[X_cols]

# X-Learner implementation
# Stage 1: Fit outcome models
model_y0 = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=42, n_jobs=-1)
model_y0.fit(X[T==0], Y[T==0])
model_y1 = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=42, n_jobs=-1)
model_y1.fit(X[T==1], Y[T==1])

# Stage 2: Impute treatment effects
D1 = Y[T==1] - model_y0.predict(X[T==1])
D0 = model_y1.predict(X[T==0]) - Y[T==0]

model_tau0 = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=42, n_jobs=-1)
model_tau0.fit(X[T==0], D0)
model_tau1 = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=42, n_jobs=-1)
model_tau1.fit(X[T==1], D1)

tau0_hat = model_tau0.predict(X)
tau1_hat = model_tau1.predict(X)

# Stage 3: Weighting by propensity score
model_g = RandomForestClassifier(n_estimators=50, min_samples_leaf=10, random_state=42, n_jobs=-1)
model_g.fit(X, T)
g_hat = model_g.predict_proba(X)[:, 1]

cate_estimates = g_hat * tau0_hat + (1 - g_hat) * tau1_hat

# Bootstrap CI for CATE difference between elderly and young
n_bootstraps = 100
bootstrap_diffs = []

for _ in range(n_bootstraps):
    # Resample CATE estimates, not the whole process, for speed.
    # This captures sampling variability of the mean CATEs.
    sample_indices = np.random.choice(df.index, size=len(df), replace=True)
    
    # Get the CATE estimates corresponding to the resampled indices
    resampled_cates = cate_estimates[sample_indices]
    
    # Get the original age groups for the resampled indices
    resampled_ages = df.loc[sample_indices, 'age']
    
    mean_cate_elderly = resampled_cates[resampled_ages >= 65].mean()
    mean_cate_young = resampled_cates[resampled_ages < 40].mean()
    
    bootstrap_diffs.append(mean_cate_elderly - mean_cate_young)

cate_elderly_minus_young_ci = [
    np.percentile(bootstrap_diffs, 2.5),
    np.percentile(bootstrap_diffs, 97.5)
]


# Objective 4: Hierarchical Modeling

# Fit mixed-effects model
# Using a parsimonious set of fixed effects for model stability and interpretability
mix_formula = 'antibiotic_duration ~ asp_active + age + comorbidity_score + severity_score + C(year) + C(region)'
mixed_model = smf.mixedlm(formula=mix_formula, data=df, groups=df['hospital_id'])
mixed_results = mixed_model.fit(method='ML') # Use ML for AIC comparison

# Extract results from mixed model
mixed_model_aic = mixed_results.aic
mixed_asp_coeff = mixed_results.params['asp_active']
# The variance of the random intercept for hospital_id
random_effect_variance = float(mixed_results.cov_re.iloc[0,0])
# The variance of the residuals
residual_variance = mixed_results.scale
icc_value = random_effect_variance / (random_effect_variance + residual_variance)

# Fit naive OLS model with the same fixed effects
ols_model = smf.ols(formula=mix_formula, data=df)
ols_results = ols_model.fit()
naive_ols_aic = ols_results.aic