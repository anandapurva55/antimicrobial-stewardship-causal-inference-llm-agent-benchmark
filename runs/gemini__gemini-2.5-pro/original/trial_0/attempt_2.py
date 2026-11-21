import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from lifelines import CoxPHFitter
from linearmodels.panel import PanelOLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# --- Task 1: Survival Analysis ---

# Prepare data for lifelines, which requires all columns to be numeric
df_cox = df.copy()
categorical_cols_cox = ['hospital_size', 'region', 'ward', 'sex', 'infection_type']
df_cox = pd.get_dummies(df_cox, columns=categorical_cols_cox, drop_first=True)

# Define covariates for the model
covariates_cox = [col for col in df_cox.columns if col not in ['patient_id', 'hospital_id', 'year', 'age_group', 'asp_adoption_year', 'time_to_improvement', 'event_observed', 'antibiotic_duration', 'length_of_stay', 'readmission_30d', 'c_diff_infection', 'mortality']]

# Fit the Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(df_cox,
        duration_col='time_to_improvement',
        event_col='event_observed',
        formula=' + '.join(covariates_cox),
        cluster_col='hospital_id')

cox_hr_asp = cph.hazard_ratios_['asp_active']
cox_se_asp = cph.summary.loc['asp_active', 'se(coef)']

# Test the proportional hazards assumption
ph_test_results = cph.check_assumptions(df_cox, p_value_threshold=1.0, show_plots=False)
ph_test_pvalues = {}
for res_df in ph_test_results:
    var_name = res_df.index[0]
    p_val = res_df.loc[var_name, 'p']
    ph_test_pvalues[var_name] = p_val


# --- Task 2: Causal Impact (DiD) ---

# Create hospital-year panel
hospital_info = df[['hospital_id', 'asp_adoption_year']].drop_duplicates()
did_panel = df.groupby(['hospital_id', 'year']).agg(
    antibiotic_duration=('antibiotic_duration', 'mean')
).reset_index()
did_panel = pd.merge(did_panel, hospital_info, on='hospital_id')

# Set up panel data for linearmodels
did_panel_indexed = did_panel.set_index(['hospital_id', 'year'])
did_panel_indexed['asp_active'] = (did_panel_indexed.index.get_level_values('year') >= did_panel_indexed['asp_adoption_year']).astype(int)

# Run TWFE model
twfe_model = PanelOLS.from_formula(
    'antibiotic_duration ~ 1 + asp_active + EntityEffects + TimeEffects',
    data=did_panel_indexed
)
twfe_results = twfe_model.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = twfe_results.params['asp_active']

# Event Study and Parallel Trends Test
did_panel['event_time'] = did_panel['year'] - did_panel['asp_adoption_year']
# For never-treated units, set event_time to a large negative number so they are not included in dummies
did_panel.loc[did_panel['asp_adoption_year'] == 9999, 'event_time'] = -1000

# Cap event times to avoid collinearity and thin bins
min_event_time = int(did_panel[did_panel['asp_adoption_year'] != 9999]['event_time'].min())
max_event_time = int(did_panel[did_panel['asp_adoption_year'] != 9999]['event_time'].max())
event_times_to_model = [t for t in range(min_event_time, max_event_time + 1) if t != -1]

for t in event_times_to_model:
    did_panel_indexed[f'T{t}'] = (did_panel['event_time'] == t).astype(int).values

formula_event_study = 'antibiotic_duration ~ 1 + ' + ' + '.join([f'T{t}' for t in event_times_to_model]) + ' + EntityEffects + TimeEffects'

event_study_model = PanelOLS.from_formula(formula_event_study, data=did_panel_indexed)
event_study_results = event_study_model.fit(cov_type='clustered', cluster_entity=True)

# Extract pre-trend test results for t=-2
pre_trend_coeff = event_study_results.params['T-2']
pre_trend_pvalue = event_study_results.pvalues['T-2']
pre_trend_ci = event_study_results.conf_int().loc['T-2'].tolist()

# Create event study dataframe
event_study_list = []
for t in event_times_to_model:
    coef = event_study_results.params[f'T{t}']
    ci = event_study_results.conf_int().loc[f'T{t}']
    event_study_list.append({'event_time': t, 'coef': coef, 'ci_low': ci[0], 'ci_high': ci[1]})

event_study = pd.DataFrame(event_study_list).sort_values('event_time').reset_index(drop=True)

# Robust DiD estimator (using never-treated as controls)
did_panel_robust = did_panel.copy()
did_panel_robust['treated_group'] = (did_panel_robust['asp_adoption_year'] != 9999).astype(int)
did_panel_robust['post'] = (did_panel_robust['year'] >= did_panel_robust['asp_adoption_year']).astype(int)
did_panel_robust = did_panel_robust.set_index(['hospital_id', 'year'])

robust_did_model = PanelOLS.from_formula(
    'antibiotic_duration ~ 1 + treated_group:post + EntityEffects + TimeEffects',
    data=did_panel_robust
)
robust_did_results = robust_did_model.fit(cov_type='clustered', cluster_entity=True)
did_att_estimate = robust_did_results.params['treated_group:post']


# --- Task 3: Treatment Heterogeneity ---

# Prepare data for CATE estimation
features = [
    'age', 'sex', 'comorbidity_score', 'immune_compromised', 'severity_score',
    'hospital_size', 'region', 'ward', 'infection_type', 'hospital_id', 'year'
]
target = 'antibiotic_duration'
treatment = 'asp_active'

X = df[features]
y = df[target]
T = df[treatment]

# One-hot encode categorical features
categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=np.number).columns

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
X_encoded_cats = encoder.fit_transform(X[categorical_features])
X_encoded_cats_df = pd.DataFrame(X_encoded_cats, columns=encoder.get_feature_names_out(categorical_features), index=X.index)

X_processed = pd.concat([X[numerical_features], X_encoded_cats_df], axis=1)

# Split data into treatment and control groups
X_treat = X_processed[T == 1]
y_treat = y[T == 1]
X_control = X_processed[T == 0]
y_control = y[T == 0]

# T-Learner with RandomForestRegressor
m1 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1)
m0 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1)

m1.fit(X_treat, y_treat)
m0.fit(X_control, y_control)

# Estimate CATE for all individuals
y1_hat = m1.predict(X_processed)
y0_hat = m0.predict(X_processed)
cate_estimates = y1_hat - y0_hat

# Bootstrap CI for elderly vs. young difference
n_bootstraps = 500
np.random.seed(42)
diffs = []
elderly_mask = df['age'] >= 65
young_mask = df['age'] < 40
indices = np.arange(len(df))

for _ in range(n_bootstraps):
    bootstrap_indices = np.random.choice(indices, size=len(indices), replace=True)
    
    cate_sample = cate_estimates[bootstrap_indices]
    elderly_mask_sample = elderly_mask.iloc[bootstrap_indices].values
    young_mask_sample = young_mask.iloc[bootstrap_indices].values
    
    # Ensure there are samples in both groups to avoid NaN
    if np.sum(elderly_mask_sample) > 0 and np.sum(young_mask_sample) > 0:
        mean_elderly = np.mean(cate_sample[elderly_mask_sample])
        mean_young = np.mean(cate_sample[young_mask_sample])
        diffs.append(mean_elderly - mean_young)

cate_elderly_minus_young_ci = np.percentile(diffs, [2.5, 97.5]).tolist()


# --- Task 4: Hierarchical Modeling ---

# Fit mixed-effects model
# Using a subset of covariates for simplicity and to avoid convergence issues
formula_mixed = 'antibiotic_duration ~ asp_active + age + C(sex) + comorbidity_score + severity_score + C(year)'
mixed_model = smf.mixedlm(formula_mixed, df, groups=df["hospital_id"])
mixed_results = mixed_model.fit(method='ML', reml=False)

mixed_model_aic = mixed_results.aic
mixed_asp_coeff = mixed_results.params['asp_active']

# Calculate ICC
random_effect_variance = mixed_results.cov_re.iloc[0, 0]
residual_variance = mixed_results.scale
icc_value = random_effect_variance / (random_effect_variance + residual_variance)

# Fit naive OLS model for comparison
naive_ols_model = smf.ols(formula_mixed, data=df)
naive_ols_results = naive_ols_model.fit()
naive_ols_aic = naive_ols_results.aic