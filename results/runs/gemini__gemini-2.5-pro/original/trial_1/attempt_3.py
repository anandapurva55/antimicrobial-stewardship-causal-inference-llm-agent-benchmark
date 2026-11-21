import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from linearmodels.panel import PanelOLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Set a random seed for reproducibility
np.random.seed(42)

# --- Objective 1: Survival Analysis ---

# Prepare data for lifelines, creating dummy variables for categorical predictors
df_cox = df.copy()
df_cox = pd.get_dummies(df_cox, columns=['infection_type', 'ward'], drop_first=True)

# Define covariates for the Cox model
covariates_cox = [
    'asp_active', 'age', 'comorbidity_score', 'severity_score', 'immune_compromised'
] + [col for col in df_cox.columns if 'infection_type_' in col or 'ward_' in col]

# Fit Cox Proportional Hazards model with clustered standard errors
cph = CoxPHFitter()
cph.fit(df_cox,
        duration_col='time_to_improvement',
        event_col='event_observed',
        formula=" + ".join(covariates_cox),
        cluster_col='hospital_id')

# Extract required outputs from the fitted model
cox_hr_asp = cph.hazard_ratios_['asp_active']
cox_se_asp = cph.summary.loc['asp_active', 'se(coef)']

# Test the proportional hazards assumption using Schoenfeld residuals
ph_test_results = cph.check_assumptions(df_cox, show_plots=False, p_value_threshold=1.0) # Check all
ph_test_pvalues = {covar.split(':')[0].strip(): results.p for covar, results in ph_test_results}


# --- Objective 2: Causal Impact (DiD) ---

# 2a. Create hospital-year panel by aggregating patient data
did_panel = df.groupby(['hospital_id', 'year']).agg(
    antibiotic_duration=('antibiotic_duration', 'mean'),
    asp_active=('asp_active', 'first'),
    asp_adoption_year=('asp_adoption_year', 'first')
).reset_index()

# 2b. Standard Two-Way Fixed Effects (TWFE) Model
panel_data_twfe = did_panel.set_index(['hospital_id', 'year'])
mod_twfe = PanelOLS.from_formula('antibiotic_duration ~ asp_active + EntityEffects + TimeEffects', data=panel_data_twfe)
res_twfe = mod_twfe.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = res_twfe.params['asp_active']

# 2c. Event Study for Parallel Trends Test
event_df = did_panel.copy()
# Define event time relative to adoption year
event_df['event_time'] = np.where(event_df['asp_adoption_year'] != 9999,
                                  event_df['year'] - event_df['asp_adoption_year'],
                                  np.nan)

# Bin distant event times for model stability
event_df['event_time'] = event_df['event_time'].clip(lower=-3, upper=3)
event_dummies = pd.get_dummies(event_df['event_time'], prefix='event_time', dummy_na=False).astype(int)

# Omit the conventional reference period t-1
if 'event_time_-1.0' in event_dummies.columns:
    event_dummies = event_dummies.drop(columns=['event_time_-1.0'])

event_df = pd.concat([event_df, event_dummies], axis=1).set_index(['hospital_id', 'year'])

# Wrap column names in Q() for formulaic to parse names with special characters like '-' and '.'
dummy_cols_for_formula = [f'Q("{col}")' for col in event_dummies.columns]
formula_es = f"antibiotic_duration ~ 1 + {' + '.join(dummy_cols_for_formula)} + EntityEffects + TimeEffects"

mod_es = PanelOLS.from_formula(formula_es, data=event_df)
res_es = mod_es.fit(cov_type='clustered', cluster_entity=True)

# Extract pre-trend coefficient for t-2 to test parallel trends assumption
# The coefficient name in the results will be the original column name, not the Q-wrapped one.
pre_trend_coeff_name = 'event_time_-2.0'
if pre_trend_coeff_name in res_es.params:
    pre_trend_coeff = res_es.params[pre_trend_coeff_name]
    pre_trend_pvalue = res_es.pvalues[pre_trend_coeff_name]
    pre_trend_ci = res_es.conf_int().loc[pre_trend_coeff_name].tolist()
else:
    pre_trend_coeff = np.nan
    pre_trend_pvalue = np.nan
    pre_trend_ci = [np.nan, np.nan]


# Create a clean DataFrame for the event study results
event_study_results = pd.DataFrame({
    'coef': res_es.params,
    'ci_low': res_es.conf_int()['lower'],
    'ci_high': res_es.conf_int()['upper']
}).reset_index()

event_study = event_study_results[event_study_results['index'].str.startswith('event_time_')].copy()
event_study['event_time'] = event_study['index'].str.replace('event_time_', '').astype(float).astype(int)
event_study = event_study[['event_time', 'coef', 'ci_low', 'ci_high']].sort_values('event_time').reset_index(drop=True)

# 2d. Robust Staggered DiD Estimator (Callaway & Sant'Anna style with never-treated controls)
cohorts = sorted(df[df['asp_adoption_year'] != 9999]['asp_adoption_year'].unique())
panel_means = did_panel.groupby(['asp_adoption_year', 'year'])['antibiotic_duration'].mean()

att_gt_list = []
weights = []

for g in cohorts:
    treated_hospitals_g = df[df['asp_adoption_year'] == g]['hospital_id'].unique()
    num_treated = len(treated_hospitals_g)
    post_treatment_years = range(g, df['year'].max() + 1)

    for t in post_treatment_years:
        y_treat_post = panel_means.get((g, t), np.nan)
        y_treat_pre = panel_means.get((g, g - 1), np.nan)
        y_control_post = panel_means.get((9999, t), np.nan)
        y_control_pre = panel_means.get((9999, g - 1), np.nan)

        if not any(pd.isna([y_treat_post, y_treat_pre, y_control_post, y_control_pre])):
            did_treat = y_treat_post - y_treat_pre
            did_control = y_control_post - y_control_pre
            att_gt = did_treat - did_control
            att_gt_list.append(att_gt)
            weights.append(num_treated)

did_att_estimate = np.average(att_gt_list, weights=weights)


# --- Objective 3: Treatment Heterogeneity (T-Learner) ---

# Prepare data for machine learning models
df_cate = df.copy()
y = df_cate['antibiotic_duration']
T = df_cate['asp_active']
X = pd.get_dummies(df_cate[['age', 'sex', 'comorbidity_score', 'immune_compromised', 'infection_type', 'severity_score', 'hospital_id', 'year']],
                   columns=['sex', 'infection_type', 'hospital_id', 'year'], drop_first=True)

X_treat, y_treat = X[T == 1], y[T == 1]
X_control, y_control = X[T == 0], y[T == 0]

# T-Learner: Train separate models for treated and control groups
m1 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1)
m0 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1)
m1.fit(X_treat, y_treat)
m0.fit(X_control, y_control)

# Estimate CATE for all individuals
cate_estimates = m1.predict(X) - m0.predict(X)

# Bootstrap a 95% CI for the difference in CATE between elderly and young patients
bootstrap_diffs = []
n_bootstraps = 200 # A practical number of iterations
for i in range(n_bootstraps):
    sample_indices = df.sample(n=len(df), replace=True, random_state=i).index
    df_s = df.iloc[sample_indices].reset_index(drop=True)

    y_s = df_s['antibiotic_duration']
    T_s = df_s['asp_active']
    X_s = pd.get_dummies(df_s[['age', 'sex', 'comorbidity_score', 'immune_compromised', 'infection_type', 'severity_score', 'hospital_id', 'year']],
                         columns=['sex', 'infection_type', 'hospital_id', 'year'], drop_first=True)
    X_s = X_s.reindex(columns=X.columns, fill_value=0) # Align columns with original data

    X_treat_s, y_treat_s = X_s[T_s == 1], y_s[T_s == 1]
    X_control_s, y_control_s = X_s[T_s == 0], y_s[T_s == 0]

    if len(X_treat_s) < 5 or len(X_control_s) < 5: continue # Ensure enough data in each arm

    m1_s = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=42, n_jobs=-1).fit(X_treat_s, y_treat_s)
    m0_s = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=42, n_jobs=-1).fit(X_control_s, y_control_s)

    cate_s = m1_s.predict(X_s) - m0_s.predict(X_s)
    df_s['cate_s'] = cate_s

    mean_elderly = df_s[df_s['age_group'] == '65+']['cate_s'].mean()
    mean_young = df_s[df_s['age_group'] == '<40']['cate_s'].mean()

    if not (pd.isna(mean_elderly) or pd.isna(mean_young)):
        bootstrap_diffs.append(mean_elderly - mean_young)

cate_elderly_minus_young_ci = [np.percentile(bootstrap_diffs, 2.5), np.percentile(bootstrap_diffs, 97.5)]


# --- Objective 4: Hierarchical Modeling ---

# Define a common formula for fixed effects
formula_common = "antibiotic_duration ~ asp_active + age + C(sex) + comorbidity_score + immune_compromised + C(infection_type) + severity_score + C(year)"

# Fit a mixed-effects model with a random intercept for hospital
mixed_model = smf.mixedlm(formula_common, df, groups=df["hospital_id"])
mixed_results = mixed_model.fit(method='ML') # Use ML for AIC comparison

# Fit a naive OLS model (pooled data, ignoring hospital clustering) for comparison
naive_ols_model = smf.ols(formula_common, data=df)
naive_ols_results = naive_ols_model.fit()

# Extract required results from the models
mixed_model_aic = mixed_results.aic
naive_ols_aic = naive_ols_results.aic
mixed_asp_coeff = mixed_results.params['asp_active']

# Calculate the Intraclass Correlation Coefficient (ICC)
random_effect_variance = float(mixed_results.cov_re.iloc[0, 0])
residual_variance = mixed_results.scale
icc_value = random_effect_variance / (random_effect_variance + residual_variance)