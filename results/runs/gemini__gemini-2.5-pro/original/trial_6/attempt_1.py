import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from linearmodels.panel import PanelOLS
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm

# --- Data Preparation ---
# Ensure correct dtypes for categorical variables used in formulas/dummies
for col in ['hospital_size', 'region', 'ward', 'sex', 'age_group', 'infection_type']:
    df[col] = df[col].astype('category')

# Define covariates for models
COVARIATES = [
    'age', 'sex', 'comorbidity_score', 'immune_compromised',
    'infection_type', 'severity_score', 'hospital_size', 'region', 'ward'
]

# --- 1. SURVIVAL ANALYSIS ---

# Prepare data for lifelines (needs numeric types for categorical vars)
df_cox = df[['time_to_improvement', 'event_observed', 'asp_active', 'hospital_id'] + COVARIATES].copy()
# Convert categorical covariates to dummies for the model
df_cox = pd.get_dummies(df_cox, columns=[c for c in COVARIATES if df_cox[c].dtype.name == 'category'], drop_first=True)

# Fit Cox Proportional Hazards Model
cph = CoxPHFitter()
# Construct formula string from dummy columns
formula_cox = "asp_active + " + " + ".join([c for c in df_cox.columns if c not in ['time_to_improvement', 'event_observed', 'hospital_id', 'asp_active']])
cph.fit(df_cox, 
        duration_col='time_to_improvement', 
        event_col='event_observed',
        cluster_col='hospital_id',
        formula=formula_cox)

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.summary.loc['asp_active', 'se(coef)'])

# Test Proportional Hazards Assumption
# The check_assumptions method returns a list of DataFrames. The first one contains the test results.
ph_test_results = cph.check_assumptions(df_cox, p_value_threshold=1, show_plots=False)[0]
ph_test_pvalues = ph_test_results['p'].to_dict()


# --- 2. CAUSAL IMPACT (DiD) ---

# Create hospital-year panel
did_panel = df.groupby(['hospital_id', 'year'])['antibiotic_duration'].mean().reset_index()
did_panel.rename(columns={'antibiotic_duration': 'mean_antibiotic_duration'}, inplace=True)
hospital_info = df[['hospital_id', 'asp_adoption_year']].drop_duplicates()
did_panel = pd.merge(did_panel, hospital_info, on='hospital_id')

# Event Study Setup
did_panel['year'] = did_panel['year'].astype(int)
did_panel['event_time'] = np.where(did_panel['asp_adoption_year'] != 9999, 
                                   did_panel['year'] - did_panel['asp_adoption_year'], 
                                   -np.inf)

# Bin distant event times to avoid sparsity and provide a stable baseline
# Range of event times is [-4, 2]. Bin <=-3 and >=2
did_panel['event_time'] = did_panel['event_time'].clip(lower=-3, upper=2)

# Set index for PanelOLS
did_panel_indexed = did_panel.set_index(['hospital_id', 'year'])

# Create event time dummies, omitting t=-1 (reference period) and t=-inf (never-treated)
event_dummies = pd.get_dummies(did_panel_indexed['event_time'], prefix='t', drop_first=False)
if 't_-1.0' in event_dummies.columns:
    event_dummies = event_dummies.drop(columns=['t_-1.0'])
if 't_-inf' in event_dummies.columns:
    event_dummies = event_dummies.drop(columns=['t_-inf']) 
did_panel_event = pd.concat([did_panel_indexed, event_dummies], axis=1)

# Run Event Study Regression
exog_vars = [col for col in did_panel_event.columns if col.startswith('t_')]
mod_event = PanelOLS.from_formula('mean_antibiotic_duration ~ 1 + ' + ' + '.join(exog_vars) + ' + EntityEffects + TimeEffects',
                                  data=did_panel_event,
                                  drop_absorbed=True)
res_event = mod_event.fit(cov_type='clustered', cluster_entity=True)

# Extract event study coefficients and CIs
event_study_results = pd.concat([res_event.params, res_event.conf_int()], axis=1)
event_study_results = event_study_results.loc[exog_vars]
event_study_results.reset_index(inplace=True)
event_study_results.rename(columns={'index': 'event_time', 'parameter': 'coef', 'lower': 'ci_low', 'upper': 'ci_high'}, inplace=True)
event_study_results['event_time'] = event_study_results['event_time'].str.replace('t_', '').astype(float).astype(int)
event_study = event_study_results.sort_values('event_time')

# Extract pre-trend test results for t=-2
target_pre_trend_col = 't_-2.0'
if target_pre_trend_col in res_event.params:
    pre_trend_coeff = float(res_event.params[target_pre_trend_col])
    pre_trend_pvalue = float(res_event.pvalues[target_pre_trend_col])
    pre_trend_ci = res_event.conf_int().loc[target_pre_trend_col].tolist()
else:
    pre_trend_coeff = np.nan
    pre_trend_pvalue = np.nan
    pre_trend_ci = [np.nan, np.nan]

# TWFE Model
panel_twfe = did_panel.copy()
panel_twfe['treat'] = ((panel_twfe['year'] >= panel_twfe['asp_adoption_year']) & (panel_twfe['asp_adoption_year'] != 9999)).astype(int)
panel_twfe = panel_twfe.set_index(['hospital_id', 'year'])

mod_twfe = PanelOLS.from_formula('mean_antibiotic_duration ~ 1 + treat + EntityEffects + TimeEffects', data=panel_twfe)
res_twfe = mod_twfe.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = float(res_twfe.params['treat'])

# Robust DiD (Callaway & Sant'Anna style, implemented via stacked regression)
att_estimates = []
cohort_sizes = []
cohorts = sorted(did_panel[did_panel['asp_adoption_year'] != 9999]['asp_adoption_year'].unique())
never_treated_hospitals = did_panel[did_panel['asp_adoption_year'] == 9999]['hospital_id'].unique()

for g in cohorts:
    cohort_hospitals = did_panel[did_panel['asp_adoption_year'] == g]['hospital_id'].unique()
    subset_panel = did_panel[did_panel['hospital_id'].isin(list(cohort_hospitals) + list(never_treated_hospitals))].copy()
    
    subset_panel['G'] = (subset_panel['asp_adoption_year'] == g).astype(int)
    subset_panel['Post'] = (subset_panel['year'] >= g).astype(int)
    
    subset_panel = subset_panel.set_index(['hospital_id', 'year'])

    mod_cs = PanelOLS.from_formula('mean_antibiotic_duration ~ 1 + G*Post + EntityEffects + TimeEffects', data=subset_panel)
    res_cs = mod_cs.fit(cov_type='clustered', cluster_entity=True)
    
    att_estimates.append(res_cs.params['G:Post'])
    cohort_sizes.append(len(cohort_hospitals))

did_att_estimate = float(np.average(att_estimates, weights=cohort_sizes))

# Finalize did_panel for output
did_panel = did_panel[['hospital_id', 'year', 'mean_antibiotic_duration']]


# --- 3. TREATMENT HETEROGENEITY (CATE) ---

# Prepare data for T-Learner
cate_outcome = 'antibiotic_duration'
cate_features = [
    'age', 'comorbidity_score', 'immune_compromised', 'severity_score',
    'sex', 'infection_type', 'ward', 'hospital_id', 'year'
]
df_cate = df[['asp_active', cate_outcome] + cate_features].copy()
X = pd.get_dummies(df_cate[cate_features], drop_first=True)
y = df_cate[cate_outcome]
T = df_cate['asp_active']

X_treat, y_treat = X[T == 1], y[T == 1]
X_control, y_control = X[T == 0], y[T == 0]

# T-Learner with RandomForestRegressor
M0 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1)
M1 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1)

M0.fit(X_control, y_control)
M1.fit(X_treat, y_treat)

cate_estimates = M1.predict(X) - M0.predict(X)

# Bootstrap CI for difference in CATE between elderly and young
n_bootstraps = 500
bootstrap_diffs = []
cate_df = pd.DataFrame({'cate': cate_estimates, 'age_group': df['age_group']})
elderly_cates = cate_df[cate_df['age_group'] == '65+']['cate']
young_cates = cate_df[cate_df['age_group'] == '<40']['cate']

np.random.seed(42) # for reproducibility
for _ in range(n_bootstraps):
    sample_elderly = elderly_cates.sample(len(elderly_cates), replace=True)
    sample_young = young_cates.sample(len(young_cates), replace=True)
    bootstrap_diffs.append(sample_elderly.mean() - sample_young.mean())

cate_elderly_minus_young_ci = [
    np.percentile(bootstrap_diffs, 2.5),
    np.percentile(bootstrap_diffs, 97.5)
]


# --- 4. HIERARCHICAL MODELING ---

# Define formula for mixed model and OLS
fixed_effects_formula = 'asp_active + age + C(sex) + comorbidity_score + immune_compromised + C(infection_type) + severity_score + C(hospital_size) + C(region) + C(ward)'
mixed_model_formula = f'antibiotic_duration ~ {fixed_effects_formula}'
ols_model_formula = f'antibiotic_duration ~ {fixed_effects_formula}'

# Fit Mixed-Effects Model
mixed_model = smf.mixedlm(mixed_model_formula, data=df, groups=df['hospital_id'])
mixed_results = mixed_model.fit(method='bfgs', reml=False) # Use ML for AIC comparison

# Extract results
mixed_model_aic = float(mixed_results.aic)
mixed_asp_coeff = float(mixed_results.params['asp_active'])
random_effect_variance = float(mixed_results.cov_re.iloc[0,0])
residual_variance = float(mixed_results.scale)
icc_value = random_effect_variance / (random_effect_variance + residual_variance)

# Fit Naive OLS Model
ols_model = smf.ols(ols_model_formula, data=df)
ols_results = ols_model.fit()
naive_ols_aic = float(ols_results.aic)