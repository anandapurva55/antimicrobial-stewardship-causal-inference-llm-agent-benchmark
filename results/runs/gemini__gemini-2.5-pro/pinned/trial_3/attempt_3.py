import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from linearmodels.panel import PanelOLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import norm

# --- Objective 1: Survival Analysis ---

# Prepare data for lifelines
df_survival = df.copy()
# Dummify categorical variables for the model
df_survival = pd.get_dummies(df_survival, columns=['sex', 'infection_type', 'age_group'], drop_first=True)
covariates_survival = [
    'asp_active', 'age', 'comorbidity_score', 'severity_score', 'immune_compromised',
    'sex_M', 'infection_type_respiratory', 'infection_type_urinary', 'infection_type_wound'
]

# Fit Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(df_survival,
        duration_col='time_to_improvement',
        event_col='event_observed',
        formula=" + ".join(covariates_survival),
        cluster_col='hospital_id')

# Extract required outputs
cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.summary.loc['asp_active', 'se(coef)'])

# Test proportional hazards assumption
ph_test_results = cph.check_assumptions(df_survival, p_value_threshold=0.05, show_plots=False)
ph_test_pvalues = {result.name[0]: result.p_value for result in ph_test_results}


# --- Objective 2: Causal Impact (DiD) ---

# Build hospital-year panel
did_panel = df.groupby(['hospital_id', 'year']).agg(
    antibiotic_duration=('antibiotic_duration', 'mean'),
    asp_active=('asp_active', 'first') # Should be constant within hospital-year
).reset_index()

# Get adoption year for each hospital
hospital_adoption_year = df[['hospital_id', 'asp_adoption_year']].drop_duplicates().set_index('hospital_id')
did_panel = did_panel.merge(hospital_adoption_year, on='hospital_id', how='left')

# Run simple TWFE model
panel_data = did_panel.set_index(['hospital_id', 'year'])
twfe_model = PanelOLS.from_formula('antibiotic_duration ~ 1 + asp_active + EntityEffects + TimeEffects', data=panel_data)
twfe_results = twfe_model.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = float(twfe_results.params['asp_active'])

# Staggered adoption robust estimator (Sun & Abraham style event study)
# Define relative time to treatment
did_panel['relative_time'] = np.where(
    did_panel['asp_adoption_year'] != 9999,
    did_panel['year'] - did_panel['asp_adoption_year'],
    -999 # A placeholder for never-treated
)

# Create event time dummies, omitting t=-1 as the reference period
event_dummies = pd.get_dummies(did_panel['relative_time'], prefix='event', drop_first=False).astype(int)
if 'event_-1' in event_dummies.columns:
    event_dummies = event_dummies.drop(columns=['event_-1'])
if 'event_-999' in event_dummies.columns: # Don't need a dummy for never-treated
    event_dummies = event_dummies.drop(columns=['event_-999'])

did_panel_es = pd.concat([did_panel, event_dummies], axis=1)

# Build and run the event study model
formula_es = 'antibiotic_duration ~ 1 + C(hospital_id) + C(year) + ' + ' + '.join(event_dummies.columns)
es_model = smf.ols(formula_es, data=did_panel_es).fit(cov_type='cluster', cov_kwds={'groups': did_panel_es['hospital_id']})

# Extract event study coefficients and CIs
event_study_results = []
for col in event_dummies.columns:
    event_time = int(col.split('_')[1])
    coef = es_model.params[col]
    ci = es_model.conf_int().loc[col].tolist()
    event_study_results.append({'event_time': event_time, 'coef': coef, 'ci_low': ci[0], 'ci_high': ci[1]})

event_study = pd.DataFrame(event_study_results).sort_values('event_time').reset_index(drop=True)

# Test for pre-trends
pre_period_vars = [col for col in event_dummies.columns if int(col.split('_')[1]) < -1]
if pre_period_vars:
    # Test the coefficient of the earliest available pre-period
    earliest_pre_period = sorted(pre_period_vars, key=lambda x: int(x.split('_')[1]))[0]
    pre_trend_coeff = float(es_model.params[earliest_pre_period])
    pre_trend_pvalue = float(es_model.pvalues[earliest_pre_period])
    pre_trend_ci = es_model.conf_int().loc[earliest_pre_period].tolist()
else: # If no pre-periods beyond t=-2 are identifiable
    pre_trend_coeff = 0.0
    pre_trend_pvalue = 1.0
    pre_trend_ci = [0.0, 0.0]


# Calculate robust ATT (weighted average of post-treatment coefficients)
post_treatment_events = event_study[event_study['event_time'] >= 0]
weights = did_panel['relative_time'].value_counts()
weighted_sum = 0
total_weight = 0

for _, row in post_treatment_events.iterrows():
    time = row['event_time']
    if time in weights.index:
        weight = weights[time]
        weighted_sum += row['coef'] * weight
        total_weight += weight

did_att_estimate = weighted_sum / total_weight if total_weight > 0 else 0.0


# --- Objective 3: Treatment Heterogeneity (CATE) ---

# Prepare data for T-learner
df_cate = df.copy()
# Features: include patient characteristics + hospital and year dummies
categorical_features = ['sex', 'infection_type', 'hospital_id', 'year']
numeric_features = ['age', 'comorbidity_score', 'severity_score', 'immune_compromised']
outcome = 'antibiotic_duration'
treatment = 'asp_active'

X = pd.get_dummies(df_cate[numeric_features + categorical_features], drop_first=True, dtype=int)
T = df_cate[treatment]
Y = df_cate[outcome]

# T-Learner implementation
df_treated = df_cate[T == 1]
df_control = df_cate[T == 0]

X_treated = pd.get_dummies(df_treated[numeric_features + categorical_features], drop_first=True, dtype=int)
Y_treated = df_treated[outcome]

X_control = pd.get_dummies(df_control[numeric_features + categorical_features], drop_first=True, dtype=int)
Y_control = df_control[outcome]

# Align columns after dummification
X_all_cols = X.columns
X_treated = X_treated.reindex(columns=X_all_cols, fill_value=0)
X_control = X_control.reindex(columns=X_all_cols, fill_value=0)

# Train models
m1 = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=10, n_jobs=-1)
m0 = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=10, n_jobs=-1)
m1.fit(X_treated, Y_treated)
m0.fit(X_control, Y_control)

# Predict potential outcomes
y1_hat = m1.predict(X)
y0_hat = m0.predict(X)

# Estimate CATE
cate_estimates = y1_hat - y0_hat

# Bootstrap CI for CATE difference (elderly vs. young)
n_bootstraps = 250
bootstrap_diffs = []
df_cate['age_group_simple'] = 'middle'
df_cate.loc[df_cate['age'] >= 65, 'age_group_simple'] = 'elderly'
df_cate.loc[df_cate['age'] < 40, 'age_group_simple'] = 'young'
df_cate['cate'] = cate_estimates

for _ in range(n_bootstraps):
    sample_df = df_cate.sample(n=len(df_cate), replace=True, random_state=_)
    mean_cate_elderly = sample_df[sample_df['age_group_simple'] == 'elderly']['cate'].mean()
    mean_cate_young = sample_df[sample_df['age_group_simple'] == 'young']['cate'].mean()
    if pd.notna(mean_cate_elderly) and pd.notna(mean_cate_young):
        bootstrap_diffs.append(mean_cate_elderly - mean_cate_young)

cate_elderly_minus_young_ci = [
    float(np.percentile(bootstrap_diffs, 2.5)),
    float(np.percentile(bootstrap_diffs, 97.5))
]


# --- Objective 4: Hierarchical Modeling ---

# Fit mixed-effects model
formula_mixed = 'antibiotic_duration ~ asp_active + age + comorbidity_score + severity_score + C(sex) + C(infection_type) + C(year)'
mixed_model = smf.mixedlm(formula_mixed, df, groups=df["hospital_id"]).fit(method='ML')

mixed_model_aic = float(mixed_model.aic)
mixed_asp_coeff = float(mixed_model.params['asp_active'])

# Calculate ICC
# Variance of the random effect (intercept)
random_effect_variance = float(mixed_model.cov_re.iloc[0, 0])
# Residual variance
residual_variance = float(mixed_model.scale)
icc_value = random_effect_variance / (random_effect_variance + residual_variance)

# Fit naive OLS model for comparison
naive_ols_model = smf.ols(formula_mixed, data=df).fit()
naive_ols_aic = float(naive_ols_model.aic)