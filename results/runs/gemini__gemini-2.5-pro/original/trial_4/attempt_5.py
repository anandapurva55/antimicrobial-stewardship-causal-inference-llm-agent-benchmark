import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from linearmodels.panel import PanelOLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Ensure reproducibility
np.random.seed(42)

# --- 1. SURVIVAL ANALYSIS ---

# Prepare data for lifelines: one-hot encode categorical variables
survival_df = df.copy()
categorical_cols_survival = ['sex', 'infection_type', 'hospital_size', 'region']
survival_df = pd.get_dummies(survival_df, columns=categorical_cols_survival, drop_first=True)

# Define covariates for the Cox model
covariates_survival = [
    'asp_active', 'age', 'comorbidity_score', 'severity_score', 'immune_compromised'
] + [col for col in survival_df.columns if any(cat_col in col for cat_col in categorical_cols_survival)]

# Fit Cox Proportional Hazards model with clustered standard errors
cph = CoxPHFitter()
cph.fit(survival_df,
        duration_col='time_to_improvement',
        event_col='event_observed',
        formula=" + ".join(covariates_survival),
        cluster_col='hospital_id')

# Extract required outputs
cox_summary = cph.summary
cox_hr_asp = float(cox_summary.loc['asp_active', 'exp(coef)'])
cox_se_asp = float(cox_summary.loc['asp_active', 'se(coef)'])

# Test the proportional hazards assumption
ph_test_results = cph.check_assumptions(survival_df, p_value_threshold=1.0, show_plots=False)
ph_test_pvalues = {result.name[0]: result.p_value for result in ph_test_results}


# --- 2. CAUSAL IMPACT (DiD) ---

# Create hospital-year panel
did_panel = df.groupby(['hospital_id', 'year']).agg(
    antibiotic_duration=('antibiotic_duration', 'mean')
).reset_index()

# Add hospital-level time-invariant info (adoption year)
hospital_info = df[['hospital_id', 'asp_adoption_year']].drop_duplicates()
did_panel = pd.merge(did_panel, hospital_info, on='hospital_id')

# Set index for PanelOLS
did_panel = did_panel.set_index(['hospital_id', 'year'])

# a) TWFE Estimator
did_panel['treated'] = (did_panel.index.get_level_values('year') >= did_panel['asp_adoption_year']).astype(int)
twfe_model = PanelOLS.from_formula('antibiotic_duration ~ treated + EntityEffects + TimeEffects', data=did_panel)
twfe_results = twfe_model.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = float(twfe_results.params['treated'])

# b) Event Study for Parallel Trends Test
did_panel['event_time'] = did_panel.index.get_level_values('year') - did_panel['asp_adoption_year']
# For never-treated units, set event_time to a value outside the analysis range
did_panel.loc[did_panel['asp_adoption_year'] == 9999, 'event_time'] = -1000

# Create event time dummies, binning distant periods
did_panel['event_time'] = did_panel['event_time'].clip(lower=-4, upper=2)
event_dummies = pd.get_dummies(did_panel['event_time'], prefix='t', drop_first=False)
# Omit reference period t=-1, never-treated group, and one endpoint dummy to avoid collinearity with FEs
event_dummies = event_dummies.drop(columns=['t_-1.0', 't_-1000.0', 't_2.0'], errors='ignore')
did_panel_event = pd.concat([did_panel, event_dummies], axis=1)

# Run event study regression
# Quote variable names to handle special characters like '-'
quoted_dummies = [f'`{col}`' for col in event_dummies.columns]
# Do not include an explicit intercept '1 +' when using two-way fixed effects in linearmodels
formula_event = 'antibiotic_duration ~ ' + ' + '.join(quoted_dummies) + ' + EntityEffects + TimeEffects'
event_study_model = PanelOLS.from_formula(formula_event, data=did_panel_event)
event_study_results = event_study_model.fit(cov_type='clustered', cluster_entity=True)

# Extract pre-trend test results for t=-2
pre_trend_coeff = float(event_study_results.params['t_-2.0'])
pre_trend_pvalue = float(event_study_results.pvalues['t_-2.0'])
pre_trend_ci = event_study_results.conf_int().loc['t_-2.0'].tolist()

# Create event_study DataFrame
event_study_params = event_study_results.params.filter(like='t_')
event_study_ci = event_study_results.conf_int().filter(like='t_', axis=0)
event_study = pd.DataFrame({
    'coef': event_study_params,
    'ci_low': event_study_ci['lower'],
    'ci_high': event_study_ci['upper']
}).reset_index()
event_study['event_time'] = event_study['index'].str.replace('t_', '').astype(float).astype(int)
event_study = event_study.drop(columns='index').sort_values('event_time')

# c) Robust DiD Estimator (Treated vs. Never-Treated)
robust_panel = did_panel[did_panel['asp_adoption_year'] != 9999] # treated units
never_treated_panel = did_panel[did_panel['asp_adoption_year'] == 9999] # never-treated units
did_robust_data = pd.concat([robust_panel, never_treated_panel])

did_robust_model = PanelOLS.from_formula('antibiotic_duration ~ treated + EntityEffects + TimeEffects', data=did_robust_data)
did_robust_results = did_robust_model.fit(cov_type='clustered', cluster_entity=True)
did_att_estimate = float(did_robust_results.params['treated'])


# --- 3. TREATMENT HETEROGENEITY ---

# Prepare data for ML models
features = ['age', 'comorbidity_score', 'severity_score', 'immune_compromised',
            'sex', 'infection_type', 'hospital_id', 'year']
X = pd.get_dummies(df[features], columns=['sex', 'infection_type'], drop_first=True)
y = df['antibiotic_duration']
T = df['asp_active']

# T-Learner implementation
X_control = X[T == 0]
y_control = y[T == 0]
X_treated = X[T == 1]
y_treated = y[T == 1]

# Model for control group
m0 = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=10)
m0.fit(X_control, y_control)

# Model for treated group
m1 = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=10)
m1.fit(X_treated, y_treated)

# Estimate CATE
cate_estimates = m1.predict(X) - m0.predict(X)

# Bootstrap CI for difference in mean CATE (elderly vs. young)
n_bootstraps = 200
bootstrap_diffs = []
df_for_boot = df.copy()
df_for_boot['cate_idx'] = np.arange(len(df))

for _ in range(n_bootstraps):
    # Resample with replacement
    boot_df = df_for_boot.sample(n=len(df), replace=True)
    
    # Prepare bootstrapped data
    boot_X = pd.get_dummies(boot_df[features], columns=['sex', 'infection_type'], drop_first=True)
    boot_y = boot_df['antibiotic_duration']
    boot_T = boot_df['asp_active']

    # Align columns for prediction on original data
    boot_X = boot_X.reindex(columns=X.columns, fill_value=0)
    
    # T-Learner on bootstrap sample
    m0_boot = RandomForestRegressor(n_estimators=50, random_state=42, min_samples_leaf=10)
    m1_boot = RandomForestRegressor(n_estimators=50, random_state=42, min_samples_leaf=10)
    
    m0_boot.fit(boot_X[boot_T == 0], boot_y[boot_T == 0])
    m1_boot.fit(boot_X[boot_T == 1], boot_y[boot_T == 1])
    
    # Predict CATE on the original full feature set
    boot_cates = m1_boot.predict(X) - m0_boot.predict(X)
    
    # Calculate statistic of interest
    mean_cate_elderly = np.mean(boot_cates[df['age_group'] == '65+'])
    mean_cate_young = np.mean(boot_cates[df['age_group'] == '<40'])
    bootstrap_diffs.append(mean_cate_elderly - mean_cate_young)

# Calculate 95% CI
cate_elderly_minus_young_ci = [
    float(np.percentile(bootstrap_diffs, 2.5)),
    float(np.percentile(bootstrap_diffs, 97.5))
]


# --- 4. HIERARCHICAL MODELING ---

# Define formula for models
formula_hierarchical = 'antibiotic_duration ~ asp_active + age + comorbidity_score + severity_score + immune_compromised + C(sex) + C(infection_type) + C(hospital_size) + C(region)'

# Fit mixed-effects model
mixed_model = smf.mixedlm(formula_hierarchical, df, groups=df["hospital_id"])
mixed_model_fit = mixed_model.fit(reml=False) # Use ML for AIC comparison

# Extract mixed model results
mixed_model_aic = float(mixed_model_fit.aic)
mixed_asp_coeff = float(mixed_model_fit.params['asp_active'])
# Variance of the random intercept for hospital_id
random_effect_variance = float(mixed_model_fit.cov_re.iloc[0, 0])
# Residual variance
residual_variance = float(mixed_model_fit.scale)
icc_value = random_effect_variance / (random_effect_variance + residual_variance)

# Fit naive OLS model with hospital fixed effects for comparison
formula_ols = formula_hierarchical + ' + C(hospital_id)'
naive_ols_model = smf.ols(formula_ols, data=df)
naive_ols_fit = naive_ols_model.fit()

naive_ols_aic = float(naive_ols_fit.aic)