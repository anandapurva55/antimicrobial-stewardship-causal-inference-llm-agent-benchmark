import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
from sklearn.ensemble import GradientBoostingRegressor
import lifelines
from lifelines import CoxPHFitter, KaplanMeierFitter

# Ensure reproducibility for bootstrap
np.random.seed(42)

# --- Data Preparation for Models ---
# Create a dataframe with dummy variables for regression models
df_dummies = pd.get_dummies(df, columns=[
    'hospital_size', 'region', 'ward', 'sex', 'age_group', 'infection_type'
], drop_first=True)

# Define covariates for models
base_covariates = [
    'age', 'comorbidity_score', 'immune_compromised', 'severity_score'
]
categorical_covariates_dummies = [col for col in df_dummies.columns if
                                  any(prefix in col for prefix in
                                      ['hospital_size_', 'region_', 'ward_', 'sex_', 'infection_type_'])]
all_covariates = ['asp_active'] + base_covariates + categorical_covariates_dummies


# 1. SURVIVAL ANALYSIS
# ---
cox_df = df_dummies.copy()
cox_df['hospital_id'] = df['hospital_id'] # Ensure hospital_id is present for clustering

cph = CoxPHFitter()
cph.fit(cox_df,
        duration_col='time_to_improvement',
        event_col='event_observed',
        formula='asp_active + ' + ' + '.join(base_covariates + categorical_covariates_dummies),
        cluster_col='hospital_id')

cox_hr_asp = cph.hazard_ratios_['asp_active']
cox_se_asp = cph.summary.loc['asp_active', 'se(coef)']

# Test proportional hazards assumption
# The check_assumptions method can return an empty list, causing an IndexError.
# Using the underlying statistics function is more robust.
ph_test_results_obj = lifelines.statistics.proportional_hazard_test(cph, cox_df)
ph_test_pvalues = ph_test_results_obj.summary['p'].to_dict()


# 2. CAUSAL IMPACT (DiD)
# ---
# Build hospital-year panel
did_panel = df.groupby(['hospital_id', 'year']).agg(
    antibiotic_duration=('antibiotic_duration', 'mean')
).reset_index()

# Merge in adoption year info
hospital_info = df[['hospital_id', 'asp_adoption_year']].drop_duplicates()
did_panel = pd.merge(did_panel, hospital_info, on='hospital_id')

# --- Event Study and Parallel Trends Test ---
# Define event time relative to adoption year
did_panel['event_time'] = did_panel['year'] - did_panel['asp_adoption_year']
# Never-treated units don't have an event time
did_panel.loc[did_panel['asp_adoption_year'] == 9999, 'event_time'] = np.nan

# Create event time dummies, omitting t=-1 as reference
min_event_time = int(did_panel['event_time'].min())
max_event_time = int(did_panel['event_time'].max())
event_times = [t for t in range(min_event_time, max_event_time + 1) if t != -1 and t != (min_event_time - 1)] # Also avoid unidentifiable period

for t in event_times:
    did_panel[f'T{t}'] = (did_panel['event_time'] == t).astype(int)

# Set up PanelOLS
panel_data = did_panel.set_index(['hospital_id', 'year'])
exog_vars = [f'T{t}' for t in event_times]
exog = sm.add_constant(panel_data[exog_vars])

model_event = PanelOLS(panel_data['antibiotic_duration'], exog, entity_effects=True, time_effects=True)
results_event = model_event.fit(cov_type='clustered', cluster_entity=True)

# Extract pre-trend test results (earliest identifiable pre-period)
earliest_pre_period = min(t for t in event_times if t < -1)
pre_trend_coeff_col = f'T{earliest_pre_period}'
pre_trend_coeff = results_event.params[pre_trend_coeff_col]
pre_trend_pvalue = results_event.pvalues[pre_trend_coeff_col]
pre_trend_ci = results_event.conf_int().loc[pre_trend_coeff_col].tolist()

# Create event study results dataframe
event_study_list = []
for t in event_times:
    col = f'T{t}'
    coef = results_event.params[col]
    ci = results_event.conf_int().loc[col]
    event_study_list.append({'event_time': t, 'coef': coef, 'ci_low': ci['lower'], 'ci_high': ci['upper']})

event_study = pd.DataFrame(event_study_list).sort_values('event_time').reset_index(drop=True)

# --- TWFE Estimator ---
panel_data['asp_active'] = ((panel_data['year'] >= panel_data['asp_adoption_year']) & (panel_data['asp_adoption_year'] != 9999)).astype(int)
twfe_model = PanelOLS(panel_data['antibiotic_duration'], sm.add_constant(panel_data['asp_active']), entity_effects=True, time_effects=True)
twfe_results = twfe_model.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = twfe_results.params['asp_active']

# --- Robust DiD Estimator (Callaway-Sant'Anna / Stacked) ---
never_treated_hids = hospital_info[hospital_info['asp_adoption_year'] == 9999]['hospital_id'].unique()
cohorts = [2020, 2021]
att_estimates = []
cohort_sizes = []

for G in cohorts:
    cohort_hids = hospital_info[hospital_info['asp_adoption_year'] == G]['hospital_id'].unique()
    if len(cohort_hids) == 0:
        continue

    # Means for treated cohort
    y1_post = did_panel[(did_panel['hospital_id'].isin(cohort_hids)) & (did_panel['year'] >= G)]['antibiotic_duration'].mean()
    y1_pre = did_panel[(did_panel['hospital_id'].isin(cohort_hids)) & (did_panel['year'] < G)]['antibiotic_duration'].mean()

    # Means for control group (in the same time periods)
    y0_post = did_panel[(did_panel['hospital_id'].isin(never_treated_hids)) & (did_panel['year'] >= G)]['antibiotic_duration'].mean()
    y0_pre = did_panel[(did_panel['hospital_id'].isin(never_treated_hids)) & (did_panel['year'] < G)]['antibiotic_duration'].mean()

    att_g = (y1_post - y1_pre) - (y0_post - y0_pre)
    att_estimates.append(att_g)
    cohort_sizes.append(len(cohort_hids))

did_att_estimate = np.average(att_estimates, weights=cohort_sizes)


# 3. TREATMENT HETEROGENEITY (CATE)
# ---
# Prepare data for T-Learner
y = df['antibiotic_duration']
T = df['asp_active']
X_cols = base_covariates + [col for col in df_dummies.columns if any(p in col for p in ['hospital_size_', 'region_', 'ward_', 'sex_', 'infection_type_'])] + ['hospital_id', 'year']
X = df_dummies[X_cols]

# T-Learner Implementation
X_treat = X[T == 1]
y_treat = y[T == 1]
X_control = X[T == 0]
y_control = y[T == 0]

# Fit models for treated and control groups
model_treat = GradientBoostingRegressor(random_state=42)
model_treat.fit(X_treat, y_treat)

model_control = GradientBoostingRegressor(random_state=42)
model_control.fit(X_control, y_control)

# Estimate CATE for all individuals
y1_hat = model_treat.predict(X)
y0_hat = model_control.predict(X)
cate_estimates = y1_hat - y0_hat

# Bootstrap CI for difference in CATE between elderly and young
n_bootstraps = 500
diffs = []
age_65_plus_mask_orig = (df['age_group'] == '65+')
age_40_minus_mask_orig = (df['age_group'] == '<40')

for i in range(n_bootstraps):
    boot_indices = np.random.choice(df.index, size=len(df), replace=True)
    df_boot = df.iloc[boot_indices].reset_index(drop=True)
    df_dummies_boot = df_dummies.iloc[boot_indices].reset_index(drop=True)

    y_boot = df_boot['antibiotic_duration']
    T_boot = df_boot['asp_active']
    X_boot = df_dummies_boot[X_cols]
    
    X_treat_boot = X_boot[T_boot == 1]
    y_treat_boot = y_boot[T_boot == 1]
    X_control_boot = X_boot[T_boot == 0]
    y_control_boot = y_boot[T_boot == 0]

    if len(X_treat_boot) > 0 and len(X_control_boot) > 0:
        m1_boot = GradientBoostingRegressor(random_state=42).fit(X_treat_boot, y_treat_boot)
        m0_boot = GradientBoostingRegressor(random_state=42).fit(X_control_boot, y_control_boot)

        cate_boot = m1_boot.predict(X_boot) - m0_boot.predict(X_boot)

        age_65_plus_mask_boot = (df_boot['age_group'] == '65+')
        age_40_minus_mask_boot = (df_boot['age_group'] == '<40')
        
        if np.sum(age_65_plus_mask_boot) > 0 and np.sum(age_40_minus_mask_boot) > 0:
            mean_cate_elderly = np.mean(cate_boot[age_65_plus_mask_boot])
            mean_cate_young = np.mean(cate_boot[age_40_minus_mask_boot])
            diffs.append(mean_cate_elderly - mean_cate_young)

cate_elderly_minus_young_ci = [np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)]


# 4. HIERARCHICAL MODELING
# ---
# Using antibiotic_duration as the outcome for consistency
formula = (
    "antibiotic_duration ~ asp_active + age + C(sex) + comorbidity_score + "
    "immune_compromised + C(infection_type) + severity_score + C(hospital_size) + "
    "C(region) + C(ward) + C(year)"
)

# Mixed-effects model
mixed_model = smf.mixedlm(formula, data=df, groups=df["hospital_id"])
mixed_model_fit = mixed_model.fit(reml=False) # Use ML for AIC comparison

mixed_model_aic = mixed_model_fit.aic
mixed_asp_coeff = mixed_model_fit.params['asp_active']

# Extract variance components
random_effect_variance = float(mixed_model_fit.cov_re.iloc[0, 0])
residual_variance = mixed_model_fit.scale
icc_value = random_effect_variance / (random_effect_variance + residual_variance)

# Naive OLS model for comparison
ols_model = smf.ols(formula, data=df)
ols_model_fit = ols_model.fit()
naive_ols_aic = ols_model_fit.aic