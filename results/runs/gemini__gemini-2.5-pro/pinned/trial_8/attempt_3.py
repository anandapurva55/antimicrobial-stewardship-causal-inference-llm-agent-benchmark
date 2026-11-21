import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
from linearmodels.panel import PanelOLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# --- Objective 1: Survival Analysis ---

# Prepare data for lifelines, dummifying categorical variables
df_cox = df.copy()
categorical_cols_cox = ['sex', 'infection_type', 'hospital_size', 'region', 'ward', 'age_group']
df_cox = pd.get_dummies(df_cox, columns=categorical_cols_cox, drop_first=True, dtype=float)

# Define covariates for the Cox model
base_covariates = ['asp_active', 'age', 'comorbidity_score', 'immune_compromised', 'severity_score']
dummy_covariates = [col for col in df_cox.columns if any(col.startswith(cat + '_') for cat in categorical_cols_cox)]
all_covariates = base_covariates + dummy_covariates

# Create the dataframe for fitting with only necessary columns to avoid formula parsing issues
df_for_cph = df_cox[['time_to_improvement', 'event_observed', 'hospital_id'] + all_covariates]

# Fit the Cox Proportional Hazards model with clustered standard errors
cph = CoxPHFitter()
cph.fit(df_for_cph,
        duration_col='time_to_improvement',
        event_col='event_observed',
        cluster_col='hospital_id')

# Extract required outputs
cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.summary.loc['asp_active', 'se(coef)'])

# Test the Proportional Hazards assumption
ph_test_results = cph.check_assumptions(df_for_cph, p_value_threshold=1.0, show_plots=False)
ph_test_pvalues = {}
for result_df in ph_test_results:
    covariate_name = result_df.index[0]
    p_value = result_df['p'].iloc[0]
    ph_test_pvalues[covariate_name] = float(p_value)


# --- Objective 2: Causal Impact (DiD) ---

# 2.1: Build the hospital-year panel
did_panel = df.groupby(['hospital_id', 'year'])['antibiotic_duration'].mean().reset_index()
hospital_info = df[['hospital_id', 'asp_adoption_year']].drop_duplicates()
did_panel = pd.merge(did_panel, hospital_info, on='hospital_id', how='left')
did_panel = did_panel.sort_values(['hospital_id', 'year']).reset_index(drop=True)

# 2.2: Event Study for Parallel Trends
event_panel = did_panel.copy()
event_panel['event_time'] = event_panel['year'] - event_panel['asp_adoption_year']
event_panel.loc[event_panel['asp_adoption_year'] == 9999, 'event_time'] = -999 # Sentinel for never-treated

# Binning event times to ensure stability
event_panel['event_time'] = np.where(event_panel['event_time'] <= -4, -4, event_panel['event_time'])
event_panel['event_time'] = np.where(event_panel['event_time'] >= 2, 2, event_panel['event_time'])

# Create event time dummies, omitting t=-1 (reference period)
event_panel['event_time'] = event_panel['event_time'].astype(int)
event_dummies = pd.get_dummies(event_panel['event_time'], prefix='t', drop_first=False)
event_dummies.columns = event_dummies.columns.str.replace('-', 'neg')

if 't_neg1' in event_dummies.columns:
    event_dummies = event_dummies.drop(columns=['t_neg1'])
if 't_neg999' in event_dummies.columns:
    event_dummies = event_dummies.drop(columns=['t_neg999'])

event_panel = pd.concat([event_panel, event_dummies], axis=1)

# Run event study regression with two-way fixed effects
dummy_cols = list(event_dummies.columns)
formula_event = f"antibiotic_duration ~ {' + '.join(dummy_cols)} + C(hospital_id) + C(year)"
event_model = smf.ols(formula_event, data=event_panel).fit(cov_type='cluster', cov_kwds={'groups': event_panel['hospital_id']})

# Extract results for the event study plot
event_study_results = []
for col in dummy_cols:
    event_time_str = col.split('_')[1].replace('neg', '-')
    event_time = int(event_time_str)
    coef = event_model.params[col]
    ci = event_model.conf_int().loc[col].tolist()
    event_study_results.append({'event_time': event_time, 'coef': coef, 'ci_low': ci[0], 'ci_high': ci[1]})

# Add the reference period t=-1
event_study_results.append({'event_time': -1, 'coef': 0, 'ci_low': 0, 'ci_high': 0})
event_study = pd.DataFrame(event_study_results).sort_values('event_time').reset_index(drop=True)

# Test pre-trends (using t=-2 as representative pre-period)
pre_trend_coeff_name = 't_neg2'
pre_trend_coeff = float(event_model.params[pre_trend_coeff_name])
pre_trend_pvalue = float(event_model.pvalues[pre_trend_coeff_name])
pre_trend_ci = event_model.conf_int().loc[pre_trend_coeff_name].tolist()

# 2.3: Standard TWFE ATT
twfe_panel = did_panel.copy().set_index(['hospital_id', 'year'])
twfe_panel['asp_active'] = (twfe_panel.index.get_level_values('year') >= twfe_panel['asp_adoption_year']).astype(int)
twfe_model = PanelOLS.from_formula('antibiotic_duration ~ asp_active + EntityEffects + TimeEffects', data=twfe_panel)
twfe_results = twfe_model.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = float(twfe_results.params['asp_active'])

# 2.4: Robust Staggered Adoption ATT (Callaway & Sant'Anna style using never-treated)
cohort_atts = []
cohort_weights = []
adopter_cohorts = sorted(did_panel[did_panel['asp_adoption_year'] != 9999]['asp_adoption_year'].unique())
never_treated_hospitals = did_panel[did_panel['asp_adoption_year'] == 9999]['hospital_id'].unique()

for cohort_year in adopter_cohorts:
    cohort_hospitals = did_panel[did_panel['asp_adoption_year'] == cohort_year]['hospital_id'].unique()
    
    sub_panel = did_panel[did_panel['hospital_id'].isin(list(cohort_hospitals) + list(never_treated_hospitals))].copy()
    sub_panel['treat'] = sub_panel['hospital_id'].isin(cohort_hospitals).astype(int)
    sub_panel['post'] = (sub_panel['year'] >= cohort_year).astype(int)
    sub_panel['did'] = sub_panel['treat'] * sub_panel['post']
    
    sub_panel = sub_panel.set_index(['hospital_id', 'year'])
    
    model_cs = PanelOLS.from_formula('antibiotic_duration ~ did + EntityEffects + TimeEffects', data=sub_panel)
    res_cs = model_cs.fit(cov_type='clustered', cluster_entity=True)
    
    cohort_atts.append(res_cs.params['did'])
    cohort_weights.append(len(cohort_hospitals))

did_att_estimate = float(np.average(cohort_atts, weights=cohort_weights))


# --- Objective 3: Treatment Heterogeneity ---

# Prepare data for CATE estimation
df_cate = df.copy()
T = df_cate['asp_active']
y = df_cate['antibiotic_duration']

# Features: include patient characteristics + year and hospital as fixed effects
features_cate = ['age', 'comorbidity_score', 'immune_compromised', 'severity_score', 'year',
                 'sex', 'infection_type', 'ward', 'hospital_id']
X = pd.get_dummies(df_cate[features_cate], columns=['sex', 'infection_type', 'ward', 'hospital_id'], drop_first=True, dtype=float)

# T-Learner implementation
X_treat = X[T == 1]
y_treat = y[T == 1]
X_control = X[T == 0]
y_control = y[T == 0]

# Train model for treated group
m1 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1)
m1.fit(X_treat, y_treat)

# Train model for control group
m0 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1)
m0.fit(X_control, y_control)

# Estimate CATE for all individuals
cate_estimates = m1.predict(X) - m0.predict(X)

# Bootstrap CI for difference in CATE between elderly and young
elderly_mask = df['age_group'] == '65+'
young_mask = df['age_group'] == '<40'

cates_elderly = cate_estimates[elderly_mask]
cates_young = cate_estimates[young_mask]

n_bootstraps = 500
bootstrap_diffs = []
rng = np.random.RandomState(42)
for _ in range(n_bootstraps):
    sample_elderly = rng.choice(cates_elderly, size=len(cates_elderly), replace=True)
    sample_young = rng.choice(cates_young, size=len(cates_young), replace=True)
    bootstrap_diffs.append(np.mean(sample_elderly) - np.mean(sample_young))

cate_elderly_minus_young_ci = [
    float(np.percentile(bootstrap_diffs, 2.5)),
    float(np.percentile(bootstrap_diffs, 97.5))
]


# --- Objective 4: Hierarchical Modeling ---

# Fit mixed-effects model
formula_mixed = "antibiotic_duration ~ asp_active + age + C(sex) + comorbidity_score + immune_compromised + C(infection_type) + severity_score + C(year)"
mixed_model = smf.mixedlm(formula_mixed, df, groups=df['hospital_id'])
mixed_model_fit = mixed_model.fit(method='bfgs')

# Extract mixed model results
mixed_model_aic = float(mixed_model_fit.aic)
mixed_asp_coeff = float(mixed_model_fit.params['asp_active'])
random_effect_variance = float(mixed_model_fit.cov_re.iloc[0,0])
residual_variance = float(mixed_model_fit.scale)
icc_value = random_effect_variance / (random_effect_variance + residual_variance)

# Fit naive OLS model for comparison
# Note: same fixed effects formula, but no group structure
naive_ols_model = smf.ols(formula_mixed, data=df)
naive_ols_fit = naive_ols_model.fit()
naive_ols_aic = float(naive_ols_fit.aic)