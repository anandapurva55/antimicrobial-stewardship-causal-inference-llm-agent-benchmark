import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from linearmodels.panel import PanelOLS
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import norm

# Suppress convergence warnings for cleaner output
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)


# --- 1. SURVIVAL ANALYSIS ---

# Prepare data for lifelines.
df_cox = df.copy()

# Select patient-level covariates
covariates_survival = [
    'asp_active', 'age', 'comorbidity_score', 'severity_score', 'immune_compromised'
]

# Create dummies for categorical variables to use as fixed effects.
# This approach preserves the original 'hospital_id' column for clustering.
dummy_features = ['hospital_id', 'year', 'infection_type']
df_dummies = pd.get_dummies(df_cox[dummy_features].astype(str), columns=dummy_features, drop_first=True, dtype=int)

# Combine the original dataframe with the new dummy variables
df_cox = pd.concat([df_cox, df_dummies], axis=1)

# Get the full list of columns for the model formula
dummy_cols = list(df_dummies.columns)
model_cols = covariates_survival + dummy_cols

# Fit the Cox Proportional Hazards model with clustered standard errors
cph = CoxPHFitter()
cph.fit(df_cox,
        duration_col='time_to_improvement',
        event_col='event_observed',
        formula=" + ".join(model_cols),
        cluster_col='hospital_id')

# Extract results
cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.summary.loc['asp_active', 'se(coef)'])

# Test the Proportional Hazards assumption
ph_test_results = cph.check_assumptions(df_cox, p_value_threshold=0.05, show_plots=False)
ph_test_pvalues = {var.split('(')[0].strip(): pval for var, pval in ph_test_results.p_values.items()}


# --- 2. CAUSAL IMPACT (DiD) ---

# Build the hospital-year panel DataFrame
did_panel_agg = df.groupby(['hospital_id', 'year']).agg(
    antibiotic_duration=('antibiotic_duration', 'mean')
).reset_index()

# Add hospital-level adoption year info to the panel
hospital_info = df[['hospital_id', 'asp_adoption_year']].drop_duplicates()
did_panel = pd.merge(did_panel_agg, hospital_info, on='hospital_id')

# Set panel data index for linearmodels
did_panel = did_panel.set_index(['hospital_id', 'year'])

# --- Event Study (for parallel trends test and robust ATT) ---
# Create relative time to event variable
did_panel['event_time'] = did_panel.index.get_level_values('year') - did_panel['asp_adoption_year']
# Never-treated units have no defined event_time; set to a placeholder
did_panel.loc[did_panel['asp_adoption_year'] == 9999, 'event_time'] = -999

# Create event time dummies, omitting t=-1 as the reference period
event_dummies = pd.get_dummies(did_panel['event_time'], prefix='t', drop_first=False, dtype=int)
if 't_-1.0' in event_dummies.columns:
    event_dummies = event_dummies.drop(columns=['t_-1.0'])
if 't_-999.0' in event_dummies.columns:
    event_dummies = event_dummies.drop(columns=['t_-999.0'])

did_panel_es = pd.concat([did_panel, event_dummies], axis=1)

# Define formula for event study regression (Sun & Abraham style)
exog_vars = [col for col in did_panel_es.columns if col.startswith('t_')]
formula_es = f'antibiotic_duration ~ 1 + {" + ".join(exog_vars)} + EntityEffects + TimeEffects'

# Run the event study model with clustered standard errors
mod_es = PanelOLS.from_formula(formula_es, data=did_panel_es)
res_es = mod_es.fit(cov_type='clustered', cluster_entity=True)

# Extract pre-trend test results for t=-2 (the earliest identifiable pre-period)
pre_trend_coeff_key = 't_-2.0'
pre_trend_coeff = float(res_es.params.get(pre_trend_coeff_key, np.nan))
pre_trend_pvalue = float(res_es.pvalues.get(pre_trend_coeff_key, np.nan))
pre_trend_ci_series = res_es.conf_int().loc[pre_trend_coeff_key]
pre_trend_ci = [float(pre_trend_ci_series['lower']), float(pre_trend_ci_series['upper'])]

# Create event study results DataFrame
event_study_list = []
for var in exog_vars:
    event_time = int(float(var.replace('t_', '')))
    coef = res_es.params[var]
    ci = res_es.conf_int().loc[var]
    event_study_list.append({
        'event_time': event_time,
        'coef': coef,
        'ci_low': ci['lower'],
        'ci_high': ci['upper']
    })
event_study = pd.DataFrame(event_study_list).sort_values('event_time').reset_index(drop=True)

# --- TWFE ATT ---
did_panel['post'] = ((did_panel.index.get_level_values('year') >= did_panel['asp_adoption_year']) & (did_panel['asp_adoption_year'] != 9999)).astype(int)
mod_twfe = PanelOLS.from_formula('antibiotic_duration ~ 1 + post + EntityEffects + TimeEffects', data=did_panel)
res_twfe = mod_twfe.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = float(res_twfe.params['post'])

# --- Robust DiD ATT (Staggering-Robust Estimator) ---
# Calculate weights: share of treated units in each post-treatment period
post_treatment_periods = did_panel['event_time'] >= 0
weights = did_panel[post_treatment_periods]['event_time'].value_counts(normalize=True).sort_index()

# Get post-treatment coefficients from the event study model
post_treatment_coeffs = res_es.params[res_es.params.index.str.startswith('t_') & (res_es.params.index.str.replace('t_', '').astype(float) >= 0)]

# Calculate the weighted average of post-treatment coefficients
did_att_estimate = 0.0
for time_str, coeff in post_treatment_coeffs.items():
    time_val = float(time_str.replace('t_', ''))
    if time_val in weights.index:
        did_att_estimate += coeff * weights[time_val]

# Reset index on did_panel for final output
did_panel = did_panel_agg # Use the originally aggregated panel for the output variable


# --- 3. TREATMENT HETEROGENEITY ---

# Prepare data for scikit-learn T-learner
df_cate = df.copy()
outcome = 'antibiotic_duration'
treatment = 'asp_active'
categorical_features = ['hospital_id', 'year', 'sex', 'infection_type', 'hospital_size', 'region', 'ward']
numeric_features = ['age', 'comorbidity_score', 'severity_score', 'immune_compromised']
features = numeric_features + categorical_features

# Create a pre-processing pipeline for features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features)
    ])

# Define the Gradient Boosting models for the T-learner
model_params = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1, 'subsample': 0.7, 'random_state': 42}
model0 = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', GradientBoostingRegressor(**model_params))])
model1 = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', GradientBoostingRegressor(**model_params))])

# Split data into treatment and control groups and fit models
X = df_cate[features]
y = df_cate[outcome]
X_treat, y_treat = X[df_cate[treatment] == 1], y[df_cate[treatment] == 1]
X_control, y_control = X[df_cate[treatment] == 0], y[df_cate[treatment] == 0]

model1.fit(X_treat, y_treat)
model0.fit(X_control, y_control)

# Estimate individual-level CATEs for the entire dataset
cate_estimates = model1.predict(X) - model0.predict(X)

# Bootstrap a 95% CI for the difference in CATE between elderly (>=65) and young (<40)
n_bootstraps = 500
diff_estimates = []
n_samples = len(df)
indices = np.arange(n_samples)

for i in range(n_bootstraps):
    boot_indices = np.random.choice(indices, size=n_samples, replace=True)
    df_boot = df.iloc[boot_indices].reset_index(drop=True)
    
    X_boot, y_boot = df_boot[features], df_boot[outcome]
    
    X_treat_boot = X_boot[df_boot[treatment] == 1]
    y_treat_boot = y_boot[df_boot[treatment] == 1]
    X_control_boot = X_boot[df_boot[treatment] == 0]
    y_control_boot = y_boot[df_boot[treatment] == 0]
    
    if len(X_treat_boot) > 5 and len(X_control_boot) > 5:
        model1_boot = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', GradientBoostingRegressor(**model_params))])
        model0_boot = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', GradientBoostingRegressor(**model_params))])
        
        model1_boot.fit(X_treat_boot, y_treat_boot)
        model0_boot.fit(X_control_boot, y_control_boot)
        cate_boot = model1_boot.predict(X_boot) - model0_boot.predict(X_boot)
        
        elderly_mask_boot = (df_boot['age'] >= 65)
        young_mask_boot = (df_boot['age'] < 40)
        
        if elderly_mask_boot.sum() > 0 and young_mask_boot.sum() > 0:
            mean_cate_elderly = np.mean(cate_boot[elderly_mask_boot])
            mean_cate_young = np.mean(cate_boot[young_mask_boot])
            diff_estimates.append(mean_cate_elderly - mean_cate_young)

diff_estimates = np.array(diff_estimates)
cate_elderly_minus_young_ci = [float(np.percentile(diff_estimates, 2.5)), float(np.percentile(diff_estimates, 97.5))]


# --- 4. HIERARCHICAL MODELING ---

# Prepare data and formula for statsmodels mixed-effects model
df_h = df.copy()
# Define a comprehensive set of fixed effects, treating year as categorical
fixed_effects = [
    'asp_active', 'age', 'C(sex)', 'comorbidity_score', 'severity_score',
    'immune_compromised', 'C(infection_type)', 'C(year)'
]
formula_h = f'antibiotic_duration ~ {" + ".join(fixed_effects)}'

# Fit mixed-effects model with hospital as a random intercept
# Use ML (not REML) for valid AIC comparison with OLS
mixed_model = smf.mixedlm(formula_h, df_h, groups=df_h["hospital_id"])
mixed_model_fit = mixed_model.fit(reml=False)

# Fit a naive OLS model with the same fixed effects for comparison
ols_model = smf.ols(formula_h, data=df_h)
ols_model_fit = ols_model.fit()

# Extract results from the models
mixed_model_aic = float(mixed_model_fit.aic)
naive_ols_aic = float(ols_model_fit.aic)
mixed_asp_coeff = float(mixed_model_fit.params['asp_active'])

# Calculate Intraclass Correlation Coefficient (ICC)
random_effect_variance = float(mixed_model_fit.cov_re.iloc[0,0])
residual_variance = float(mixed_model_fit.scale)
icc_value = random_effect_variance / (random_effect_variance + residual_variance)