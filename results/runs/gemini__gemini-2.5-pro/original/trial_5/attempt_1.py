import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from linearmodels.panel import PanelOLS
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import norm
import re

# Set seed for reproducibility
np.random.seed(42)

# --- Objective 1: Survival Analysis ---

# Define covariates for the Cox model
# Using formula API to handle categorical variables
formula_surv = (
    "age + comorbidity_score + immune_compromised + severity_score + "
    "C(sex) + C(infection_type) + C(hospital_size) + C(region) + C(year) + asp_active"
)

# Instantiate and fit the Cox Proportional Hazards model
# Cluster standard errors by hospital_id
cph = CoxPHFitter(cluster_col='hospital_id')
cph.fit(df, 'time_to_improvement', event_col='event_observed', formula=formula_surv)

# Extract hazard ratio and standard error for asp_active
cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.summary.loc['asp_active', 'se(coef)'])

# Test the proportional hazards assumption
# We check all variables by setting p_value_threshold to 1.0
ph_test_results = cph.check_assumptions(df, p_value_threshold=1.0, show_plots=False)
ph_test_pvalues = ph_test_results.set_index('covariate')['p'].to_dict()


# --- Objective 2: Causal Impact (DiD) ---

# Build a hospital-year panel
did_panel = df.groupby(['hospital_id', 'year']).agg(
    antibiotic_duration=('antibiotic_duration', 'mean'),
    asp_active=('asp_active', 'first'),
    asp_adoption_year=('asp_adoption_year', 'first')
).reset_index()

# Set panel data index for linearmodels
panel_data = did_panel.set_index(['hospital_id', 'year'])

# Fit a standard Two-Way Fixed Effects (TWFE) model
twfe_model = PanelOLS.from_formula(
    'antibiotic_duration ~ 1 + asp_active + EntityEffects + TimeEffects',
    data=panel_data
)
twfe_results = twfe_model.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = float(twfe_results.params['asp_active'])

# Event Study for parallel trends test
# Define event time relative to adoption year
did_panel['event_time'] = did_panel['year'] - did_panel['asp_adoption_year']
# For never-treated units, set event_time to a large negative number so they are always 'pre'
# and get binned into the leftmost category
did_panel.loc[did_panel['asp_adoption_year'] == 9999, 'event_time'] = -1000

# Bin event times to avoid collinearity and noisy estimates at the tails
min_event_time = -4
max_event_time = 3
did_panel['event_time_binned'] = did_panel['event_time'].clip(min_event_time, max_event_time).astype(int)

# Update panel_data with the new event time column
panel_data_es = did_panel.set_index(['hospital_id', 'year'])

# Fit the event study model, omitting t=-1 as the reference period
es_formula = f'antibiotic_duration ~ 1 + C(event_time_binned, ref=-1) + EntityEffects + TimeEffects'
es_model = PanelOLS.from_formula(es_formula, data=panel_data_es)
es_results = es_model.fit(cov_type='clustered', cluster_entity=True)

# Extract pre-trend test results for t=-2
pre_trend_param_name = 'C(event_time_binned, ref=-1)[T.-2]'
pre_trend_coeff = float(es_results.params[pre_trend_param_name])
pre_trend_pvalue = float(es_results.pvalues[pre_trend_param_name])
pre_trend_ci = es_results.conf_int().loc[pre_trend_param_name].tolist()

# Create the event study DataFrame
event_study_list = []
for param in es_results.params.index:
    if 'event_time_binned' in param:
        # Extract event time from parameter name like 'C(event_time_binned, ref=-1)[T.2]'
        time = int(float(re.search(r'\[T\.(-?\d+\.?\d*)\]', param).group(1)))
        coef = es_results.params[param]
        ci = es_results.conf_int().loc[param]
        event_study_list.append({
            'event_time': time,
            'coef': coef,
            'ci_low': ci['lower'],
            'ci_high': ci['upper']
        })
event_study = pd.DataFrame(event_study_list).sort_values('event_time').reset_index(drop=True)

# Estimate ATT with an estimator robust to staggered adoption (Callaway & Sant'Anna style)
# using never-treated as the only control group
g2020_hids = df[df['asp_adoption_year'] == 2020]['hospital_id'].unique()
g2021_hids = df[df['asp_adoption_year'] == 2021]['hospital_id'].unique()
g_never_hids = df[df['asp_adoption_year'] == 9999]['hospital_id'].unique()

# ATT for 2020 cohort
y_2020_post = did_panel[(did_panel.hospital_id.isin(g2020_hids)) & (did_panel.year >= 2020)]['antibiotic_duration'].mean()
y_2020_pre = did_panel[(did_panel.hospital_id.isin(g2020_hids)) & (did_panel.year < 2020)]['antibiotic_duration'].mean()
y_never_post_2020 = did_panel[(did_panel.hospital_id.isin(g_never_hids)) & (did_panel.year >= 2020)]['antibiotic_duration'].mean()
y_never_pre_2020 = did_panel[(did_panel.hospital_id.isin(g_never_hids)) & (did_panel.year < 2020)]['antibiotic_duration'].mean()
att_2020 = (y_2020_post - y_2020_pre) - (y_never_post_2020 - y_never_pre_2020)

# ATT for 2021 cohort
y_2021_post = did_panel[(did_panel.hospital_id.isin(g2021_hids)) & (did_panel.year >= 2021)]['antibiotic_duration'].mean()
y_2021_pre = did_panel[(did_panel.hospital_id.isin(g2021_hids)) & (did_panel.year < 2021)]['antibiotic_duration'].mean()
y_never_post_2021 = did_panel[(did_panel.hospital_id.isin(g_never_hids)) & (did_panel.year >= 2021)]['antibiotic_duration'].mean()
y_never_pre_2021 = did_panel[(did_panel.hospital_id.isin(g_never_hids)) & (did_panel.year < 2021)]['antibiotic_duration'].mean()
att_2021 = (y_2021_post - y_2021_pre) - (y_never_post_2021 - y_never_pre_2021)

# Weighted average of cohort ATTs
N_2020 = len(g2020_hids)
N_2021 = len(g2021_hids)
did_att_estimate = (att_2020 * N_2020 + att_2021 * N_2021) / (N_2020 + N_2021)


# --- Objective 3: Treatment Heterogeneity (CATE) ---

# Prepare data for T-Learner
# We use antibiotic_duration as the outcome, consistent with DiD
y = df['antibiotic_duration']
T = df['asp_active']

# Define features, including hospital and year as requested
features_to_encode = ['hospital_id', 'hospital_size', 'region', 'ward', 'sex', 'age_group', 'infection_type', 'year']
numerical_features = ['age', 'comorbidity_score', 'immune_compromised', 'severity_score']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(df[features_to_encode])
X = np.hstack([df[numerical_features].values, encoded_features])

# Split data into treatment and control groups
X_treat = X[T == 1]
y_treat = y[T == 1]
X_control = X[T == 0]
y_control = y[T == 0]

# Fit T-Learner models
gbr_params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_leaf': 20, 'learning_rate': 0.1, 'random_state': 42}
m1 = GradientBoostingRegressor(**gbr_params)
m0 = GradientBoostingRegressor(**gbr_params)

m1.fit(X_treat, y_treat)
m0.fit(X_control, y_control)

# Estimate CATE for all individuals
y1_hat = m1.predict(X)
y0_hat = m0.predict(X)
cate_estimates = y1_hat - y0_hat

# Bootstrap CI for the difference in mean CATE (elderly vs. young)
n_bootstraps = 200
diffs = []
age_col_idx = df.columns.get_loc('age')

for i in range(n_bootstraps):
    # Create bootstrap sample
    indices = np.random.choice(len(df), size=len(df), replace=True)
    df_boot = df.iloc[indices]
    X_boot = X[indices]
    y_boot = y.iloc[indices]
    T_boot = T.iloc[indices]

    # Split bootstrap sample
    X_treat_boot = X_boot[T_boot == 1]
    y_treat_boot = y_boot[T_boot == 1]
    X_control_boot = X_boot[T_boot == 0]
    y_control_boot = y_boot[T_boot == 0]

    # Re-fit models on bootstrap sample
    if len(X_treat_boot) > 0 and len(X_control_boot) > 0:
        m1_boot = GradientBoostingRegressor(**gbr_params).fit(X_treat_boot, y_treat_boot)
        m0_boot = GradientBoostingRegressor(**gbr_params).fit(X_control_boot, y_control_boot)

        # Predict CATEs on the bootstrap sample's data
        cate_boot = m1_boot.predict(X_boot) - m0_boot.predict(X_boot)
        df_boot['cate'] = cate_boot

        # Calculate difference in means
        mean_elderly = df_boot.loc[df_boot['age'] >= 65, 'cate'].mean()
        mean_young = df_boot.loc[df_boot['age'] < 40, 'cate'].mean()
        
        if pd.notna(mean_elderly) and pd.notna(mean_young):
            diffs.append(mean_elderly - mean_young)

# Calculate 95% CI from bootstrap distribution
cate_elderly_minus_young_ci = [np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)]


# --- Objective 4: Hierarchical Modeling ---

# Fit a mixed-effects linear model with hospital_id as a random intercept
# Outcome is antibiotic_duration to be consistent
formula_mixed = (
    'antibiotic_duration ~ asp_active + age + C(sex) + comorbidity_score + '
    'immune_compromised + C(infection_type) + severity_score + C(year)'
)

# Mixed model
mixed_model = smf.mixedlm(
    formula=formula_mixed,
    data=df,
    groups=df['hospital_id']
)
mixed_model_fit = mixed_model.fit(method="lbfgs") # Use LBFGS for better convergence

# Naive OLS model for comparison
ols_model = smf.ols(
    formula=formula_mixed,
    data=df
)
ols_model_fit = ols_model.fit()

# Extract required values from the mixed model
mixed_model_aic = float(mixed_model_fit.aic)
mixed_asp_coeff = float(mixed_model_fit.params['asp_active'])

# Calculate ICC
# Variance of the random intercept for 'hospital_id'
random_effect_variance = float(mixed_model_fit.cov_re.iloc[0, 0])
# Residual variance
residual_variance = float(mixed_model_fit.scale)
icc_value = random_effect_variance / (random_effect_variance + residual_variance)

# Extract AIC from the naive OLS model
naive_ols_aic = float(ols_model_fit.aic)