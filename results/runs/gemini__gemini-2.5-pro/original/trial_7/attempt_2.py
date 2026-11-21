import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from linearmodels.panel import PanelOLS
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. SURVIVAL ANALYSIS ---

# Prepare data for lifelines, creating dummy variables for categorical features
df_survival = df.copy()
categorical_cols_survival = ['age_group', 'sex', 'infection_type', 'hospital_size', 'region', 'ward']
df_survival = pd.get_dummies(df_survival, columns=categorical_cols_survival, drop_first=True)

# Define covariates for the Cox model
base_covariates_survival = ['comorbidity_score', 'immune_compromised', 'severity_score']
dummy_covariates_survival = [col for col in df_survival.columns if any(cat_col in col for cat_col in categorical_cols_survival)]
all_covariates_survival = ['asp_active'] + base_covariates_survival + dummy_covariates_survival

# Fit Cox Proportional Hazards model with standard errors clustered by hospital
cph = CoxPHFitter()
cph.fit(df_survival,
        duration_col='time_to_improvement',
        event_col='event_observed',
        # Quote column names to handle special characters like '<' in formula
        formula=" + ".join([f"`{col}`" for col in all_covariates_survival]),
        cluster_col='hospital_id')

# Extract hazard ratio and clustered standard error for asp_active
cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.summary.loc['asp_active', 'se(log(HR))'])

# Test the proportional hazards assumption for all covariates
# Suppress the printed output of check_assumptions to keep the output clean
from io import StringIO
import sys
old_stdout = sys.stdout
sys.stdout = StringIO()
try:
    # The fitted model stores the formula, so we just pass the dataframe
    cph.check_assumptions(df_survival, p_value_threshold=0.05, show_plots=False)
finally:
    sys.stdout = old_stdout

# Extract p-values from the Schoenfeld residuals test
ph_test_results = cph.proportional_hazard_test_
ph_test_pvalues = ph_test_results.summary['p'].to_dict()


# --- 2. CAUSAL IMPACT (DiD) ---

# Create a hospital-year panel dataset
did_panel = df.groupby(['hospital_id', 'year'])['antibiotic_duration'].mean().reset_index()
hospital_info = df[['hospital_id', 'asp_adoption_year']].drop_duplicates()
did_panel = pd.merge(did_panel, hospital_info, on='hospital_id')

# Set up for event study (staggered adoption)
# Define event_time relative to adoption year
did_panel['event_time'] = np.where(
    did_panel['asp_adoption_year'] != 9999,
    did_panel['year'] - did_panel['asp_adoption_year'],
    -np.inf  # Placeholder for never-treated
)

# Bin distant event times for a more stable estimate
min_event_time, max_event_time = -3, 3
did_panel['event_time_binned'] = did_panel['event_time'].copy()
did_panel.loc[did_panel['event_time'] <= min_event_time, 'event_time_binned'] = min_event_time
did_panel.loc[did_panel['event_time'] >= max_event_time, 'event_time_binned'] = max_event_time

# Create event time dummies, omitting t=-1 as the reference period
event_time_dummies = pd.get_dummies(did_panel['event_time_binned'], prefix='t').astype(int)
if 't_-1.0' in event_time_dummies.columns:
    event_time_dummies = event_time_dummies.drop(columns=['t_-1.0'])
if 't_-inf' in event_time_dummies.columns:
    event_time_dummies = event_time_dummies.drop(columns=['t_-inf'])
did_panel_event = pd.concat([did_panel, event_time_dummies], axis=1)
did_panel_event = did_panel_event.set_index(['hospital_id', 'year'])

# Run event study regression (TWFE with event time dummies)
exog_vars = [col for col in did_panel_event.columns if col.startswith('t_')]
exog_data = sm.add_constant(did_panel_event[exog_vars])
event_study_model = PanelOLS(did_panel_event.antibiotic_duration, exog_data, entity_effects=True, time_effects=True)
event_study_results = event_study_model.fit(cov_type='clustered', cluster_entity=True)

# Extract coefficients and CIs for the event study plot
event_study_params = event_study_results.params.filter(like='t_')
event_study_cis = event_study_results.conf_int().filter(like='t_', axis=0)
event_study = pd.DataFrame({
    'coef': event_study_params,
    'ci_low': event_study_cis.iloc[:, 0],
    'ci_high': event_study_cis.iloc[:, 1]
}).reset_index()
event_study['event_time'] = event_study['index'].str.replace('t_', '').astype(float).astype(int)
event_study = event_study[['event_time', 'coef', 'ci_low', 'ci_high']].sort_values('event_time').reset_index(drop=True)

# Test for parallel pre-trends using the coefficient for t=-2
if 't_-2.0' in event_study_results.params:
    pre_trend_coeff = float(event_study_results.params['t_-2.0'])
    pre_trend_pvalue = float(event_study_results.pvalues['t_-2.0'])
    ci = event_study_results.conf_int().loc['t_-2.0']
    pre_trend_ci = [ci['lower'], ci['upper']]
else:
    pre_trend_coeff, pre_trend_pvalue, pre_trend_ci = np.nan, np.nan, [np.nan, np.nan]

# Run standard TWFE model
did_panel['asp_active'] = (did_panel['year'] >= did_panel['asp_adoption_year']).astype(int)
panel_for_twfe = did_panel.set_index(['hospital_id', 'year'])
twfe_model = PanelOLS(panel_for_twfe.antibiotic_duration, sm.add_constant(panel_for_twfe.asp_active), entity_effects=True, time_effects=True)
twfe_results = twfe_model.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = float(twfe_results.params['asp_active'])

# Run robust DiD estimator (simplified Callaway-Sant'Anna using never-treated as controls)
cohorts = sorted(did_panel[did_panel.asp_adoption_year.isin([2020, 2021])].asp_adoption_year.unique())
control_hospitals = did_panel[did_panel.asp_adoption_year == 9999].hospital_id.unique()
atts, weights = [], []
panel_data_cs = df.groupby(['hospital_id', 'year', 'asp_adoption_year'])['antibiotic_duration'].mean().reset_index()

for g in cohorts:
    treated_hospitals = did_panel[did_panel.asp_adoption_year == g].hospital_id.unique()
    post_treatment_years = did_panel[did_panel.year >= g].year.unique()
    pre_treatment_year = g - 1

    for t in post_treatment_years:
        y_treat_post = panel_data_cs[(panel_data_cs.hospital_id.isin(treated_hospitals)) & (panel_data_cs.year == t)]['antibiotic_duration'].mean()
        y_treat_pre = panel_data_cs[(panel_data_cs.hospital_id.isin(treated_hospitals)) & (panel_data_cs.year == pre_treatment_year)]['antibiotic_duration'].mean()
        y_control_post = panel_data_cs[(panel_data_cs.hospital_id.isin(control_hospitals)) & (panel_data_cs.year == t)]['antibiotic_duration'].mean()
        y_control_pre = panel_data_cs[(panel_data_cs.hospital_id.isin(control_hospitals)) & (panel_data_cs.year == pre_treatment_year)]['antibiotic_duration'].mean()
        
        did_est = (y_treat_post - y_treat_pre) - (y_control_post - y_control_pre)
        
        if not np.isnan(did_est):
            atts.append(did_est)
            weights.append(len(treated_hospitals))

did_att_estimate = np.average(atts, weights=weights)


# --- 3. TREATMENT HETEROGENEITY (CATE) ---

# Prepare data for CATE estimation using a T-Learner
df_cate = df.copy()
cate_features_base = ['age', 'comorbidity_score', 'severity_score', 'immune_compromised']
cate_features_cat = ['sex', 'infection_type', 'ward', 'hospital_id', 'year']
X_cate = pd.get_dummies(df_cate[cate_features_base + cate_features_cat], columns=cate_features_cat, drop_first=True)
T_cate = df_cate['asp_active']
Y_cate = df_cate['antibiotic_duration']

# Align columns for prediction on the full dataset later
X_treat, Y_treat = X_cate[T_cate == 1], Y_cate[T_cate == 1]
X_control, Y_control = X_cate[T_cate == 0], Y_cate[T_cate == 0]
model_treat = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=20, n_jobs=-1)
model_control = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=20, n_jobs=-1)
model_treat.fit(X_treat, Y_treat)
model_control.fit(X_control, Y_control)

cate_estimates = model_treat.predict(X_cate) - model_control.predict(X_cate)

# Bootstrap a 95% CI for the difference in mean CATE (elderly vs. young)
n_bootstraps = 500
diffs = []
age_col = df_cate['age'].values
all_indices = np.arange(len(df_cate))
for _ in range(n_bootstraps):
    bootstrap_indices = np.random.choice(all_indices, size=len(all_indices), replace=True)
    boot_cates = cate_estimates[bootstrap_indices]
    boot_age = age_col[bootstrap_indices]
    
    mean_cate_elderly = np.nanmean(boot_cates[boot_age >= 65])
    mean_cate_young = np.nanmean(boot_cates[boot_age < 40])
    
    if not (np.isnan(mean_cate_elderly) or np.isnan(mean_cate_young)):
      diffs.append(mean_cate_elderly - mean_cate_young)
cate_elderly_minus_young_ci = list(np.percentile(diffs, [2.5, 97.5]))


# --- 4. HIERARCHICAL MODELING ---

# Define formula for mixed-effects and OLS models
formula_mixed = "antibiotic_duration ~ asp_active + severity_score + comorbidity_score + immune_compromised + C(age_group) + C(sex) + C(infection_type) + C(year)"

# Fit mixed-effects model with hospital as a random intercept
mixed_model = smf.mixedlm(formula_mixed, df, groups=df["hospital_id"])
mixed_model_fit = mixed_model.fit(method='ML', reml=False)

# Extract mixed model results
mixed_model_aic = float(mixed_model_fit.aic)
mixed_asp_coeff = float(mixed_model_fit.params['asp_active'])

# Calculate Intraclass Correlation Coefficient (ICC)
random_effect_variance = float(mixed_model_fit.cov_re.iloc[0, 0])
residual_variance = float(mixed_model_fit.scale)
icc_value = random_effect_variance / (random_effect_variance + residual_variance)

# Fit a corresponding naive OLS model with hospital as a fixed effect
ols_formula = formula_mixed + " + C(hospital_id)"
ols_model = smf.ols(ols_formula, data=df)
ols_model_fit = ols_model.fit()
naive_ols_aic = float(ols_model_fit.aic)