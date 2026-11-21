import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from linearmodels.panel import PanelOLS
from sklearn.ensemble import RandomForestRegressor

# --- Objective 1: Survival Analysis ---
df_surv = df.copy()
# Dummify categorical confounders for the model
df_surv = pd.get_dummies(df_surv, columns=['infection_type'], drop_first=True, dtype=int)
covariates_surv = [
    'asp_active', 'age', 'comorbidity_score', 'immune_compromised', 'severity_score'
] + [col for col in df_surv.columns if 'infection_type_' in col]

# Fit Cox Proportional Hazards model with clustered standard errors
cph = CoxPHFitter()
cph.fit(df_surv,
        duration_col='time_to_improvement',
        event_col='event_observed',
        formula=" + ".join(covariates_surv),
        cluster_col='hospital_id')

cox_hr_asp = cph.hazard_ratios_['asp_active']
cox_se_asp = cph.summary.loc['asp_active', 'se(coef)']

# Test the proportional hazards assumption
ph_test_results = cph.check_assumptions(df_surv, show_plots=False)
ph_test_pvalues = {covariate: results.p[0] for covariate, results in zip(cph.formula.split(' + '), ph_test_results) if not results.empty}


# --- Objective 2: Causal Impact (DiD) ---
# Create the hospital-year panel
hospital_info = df[['hospital_id', 'asp_adoption_year']].drop_duplicates()
did_panel = df.groupby(['hospital_id', 'year'])['antibiotic_duration'].mean().reset_index()
did_panel = pd.merge(did_panel, hospital_info, on='hospital_id')
did_panel['asp_active'] = (did_panel['year'] >= did_panel['asp_adoption_year']).astype(int)
did_panel = did_panel.set_index(['hospital_id', 'year'])

# 2.1: Standard TWFE model
twfe_model = PanelOLS.from_formula('antibiotic_duration ~ asp_active + EntityEffects + TimeEffects', data=did_panel)
twfe_results = twfe_model.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = float(twfe_results.params['asp_active'])

# 2.2: Event study for parallel pre-trends test
panel_es = did_panel.copy().reset_index()
panel_es['event_time'] = panel_es['year'] - panel_es['asp_adoption_year']
# Use a sentinel value for never-treated units
panel_es.loc[panel_es['asp_adoption_year'] == 9999, 'event_time'] = -999

# Create dummies for event time, omitting t = -1 (reference) and never-treated
event_dummies = pd.get_dummies(panel_es['event_time'], prefix='t', dtype=int)
event_times_to_include = [col for col in event_dummies.columns if col not in ['t_-1', 't_-999']]
panel_es = pd.concat([panel_es, event_dummies[event_times_to_include]], axis=1)
panel_es = panel_es.set_index(['hospital_id', 'year'])

formula_es = f"antibiotic_duration ~ 1 + {' + '.join(event_times_to_include)} + EntityEffects + TimeEffects"
es_model = PanelOLS.from_formula(formula_es, data=panel_es)
es_results = es_model.fit(cov_type='clustered', cluster_entity=True)

# Extract pre-trend test results for t=-2
pre_trend_period = 't_-2'
pre_trend_coeff = float(es_results.params[pre_trend_period])
pre_trend_pvalue = float(es_results.pvalues[pre_trend_period])
pre_trend_ci = es_results.conf_int().loc[pre_trend_period].tolist()

# Create event study DataFrame for plotting/reporting
event_study_data = []
for et_col in event_times_to_include:
    et = int(et_col.replace('t_', ''))
    coef = es_results.params[et_col]
    ci = es_results.conf_int().loc[et_col]
    event_study_data.append({'event_time': et, 'coef': coef, 'ci_low': ci['lower'], 'ci_high': ci['upper']})
event_study = pd.DataFrame(event_study_data).sort_values('event_time').reset_index(drop=True)

# 2.3: Robust DiD ATT (using never-treated as controls)
robust_did_data = did_panel.reset_index()
robust_did_data = robust_did_data[robust_did_data['asp_adoption_year'].isin(df[df.asp_adoption_year != 9999].asp_adoption_year.unique()) | (robust_did_data['asp_adoption_year'] == 9999)]
robust_did_data = robust_did_data.set_index(['hospital_id', 'year'])

robust_model = PanelOLS.from_formula('antibiotic_duration ~ asp_active + EntityEffects + TimeEffects', data=robust_did_data)
robust_results = robust_model.fit(cov_type='clustered', cluster_entity=True)
did_att_estimate = float(robust_results.params['asp_active'])


# --- Objective 3: Treatment Heterogeneity (CATE) ---
df_cate = df.copy()
# Treat hospital_id as a categorical feature
df_cate['hospital_id'] = df_cate['hospital_id'].astype('object')
categorical_cols = ['hospital_id', 'hospital_size', 'region', 'ward', 'sex', 'age_group', 'infection_type']
df_cate = pd.get_dummies(df_cate, columns=categorical_cols, drop_first=True, dtype=int)

treatment = 'asp_active'
outcome = 'antibiotic_duration'
# Define covariates, including year and hospital dummies as requested
covariates_cate = [col for col in df_cate.columns if col not in [
    'patient_id', 'asp_adoption_year', treatment, outcome,
    'time_to_improvement', 'event_observed', 'length_of_stay', 'readmission_30d',
    'c_diff_infection', 'mortality'
]]

X = df_cate[covariates_cate]
T = df_cate[treatment]
y = df_cate[outcome]

# T-Learner Implementation
X_treat, y_treat = X[T == 1], y[T == 1]
X_control, y_control = X[T == 0], y[T == 0]

# Use smaller, faster models for feasibility
rf_params = {'n_estimators': 50, 'max_depth': 10, 'min_samples_leaf': 10, 'random_state': 42, 'n_jobs': -1}
m1 = RandomForestRegressor(**rf_params)
m0 = RandomForestRegressor(**rf_params)

m1.fit(X_treat, y_treat)
m0.fit(X_control, y_control)

cate_estimates = m1.predict(X) - m0.predict(X)

# Bootstrap CI for difference in CATE between elderly and young
n_bootstraps = 100
diffs = []
original_indices = df.index
for _ in range(n_bootstraps):
    boot_indices = np.random.choice(original_indices, size=len(df), replace=True)
    boot_df = df.iloc[boot_indices]

    # Prepare bootstrap data for sklearn
    boot_df_cate = pd.get_dummies(boot_df, columns=categorical_cols, drop_first=True, dtype=int)
    X_boot = pd.DataFrame(columns=X.columns, index=boot_df_cate.index).fillna(0)
    common_cols = X.columns.intersection(boot_df_cate.columns)
    X_boot[common_cols] = boot_df_cate[common_cols]

    T_boot = boot_df_cate[treatment] if treatment in boot_df_cate else pd.Series(0, index=X_boot.index)
    y_boot = boot_df_cate[outcome]

    # Fit T-learner on bootstrap sample
    m1_boot = RandomForestRegressor(**rf_params).fit(X_boot[T_boot == 1], y_boot[T_boot == 1])
    m0_boot = RandomForestRegressor(**rf_params).fit(X_boot[T_boot == 0], y_boot[T_boot == 0])

    boot_cate = m1_boot.predict(X_boot) - m0_boot.predict(X_boot)
    
    # Calculate statistic of interest
    is_elderly = (boot_df['age'] >= 65)
    is_young = (boot_df['age'] < 40)
    
    if is_elderly.sum() > 0 and is_young.sum() > 0:
        mean_elderly = boot_cate[is_elderly.values].mean()
        mean_young = boot_cate[is_young.values].mean()
        diffs.append(mean_elderly - mean_young)

cate_elderly_minus_young_ci = np.percentile(diffs, [2.5, 97.5]).tolist()


# --- Objective 4: Hierarchical Modeling ---
formula_mixed = 'antibiotic_duration ~ asp_active + age + comorbidity_score + severity_score + immune_compromised + C(year)'
# Fit mixed-effects model with hospital as a random intercept
mixed_model = smf.mixedlm(formula_mixed, data=df, groups=df['hospital_id'])
mixed_results = mixed_model.fit(method='ML')

mixed_model_aic = float(mixed_results.aic)
mixed_asp_coeff = float(mixed_results.params['asp_active'])

# Calculate Intraclass Correlation Coefficient (ICC)
random_effect_variance = float(mixed_results.cov_re.iloc[0, 0])
residual_variance = float(mixed_results.scale)
icc_value = random_effect_variance / (random_effect_variance + residual_variance)

# Fit naive OLS model for AIC comparison
ols_model = smf.ols(formula_mixed, data=df)
ols_results = ols_model.fit()
naive_ols_aic = float(ols_results.aic)