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

# Prepare data for Cox model: one-hot encode categorical variables
df_cox = df.copy()
categorical_cols_cox = ['hospital_size', 'region', 'ward', 'sex', 'age_group', 'infection_type']
df_cox = pd.get_dummies(df_cox, columns=categorical_cols_cox, drop_first=True, dtype=float)

# Sanitize column names for formula compatibility by replacing special characters
df_cox.columns = df_cox.columns.str.replace('+', 'plus', regex=False)
df_cox.columns = df_cox.columns.str.replace('<', 'lt', regex=False)
df_cox.columns = df_cox.columns.str.replace('-', '_', regex=False)
df_cox.columns = df_cox.columns.str.replace('>=', 'gte', regex=False)


# Define covariates for the model
all_cols = df_cox.columns
base_covariates = ['age', 'comorbidity_score', 'immune_compromised', 'severity_score']
ohe_covariates = [col for col in all_cols if any(cat_col in col for cat_col in categorical_cols_cox)]
formula_cols = ['asp_active'] + base_covariates + ohe_covariates

# Fit Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(df_cox,
        duration_col='time_to_improvement',
        event_col='event_observed',
        formula=" + ".join(formula_cols),
        cluster_col='hospital_id')

# Extract required outputs
cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.summary.loc['asp_active', 'se(coef)'])

# Test proportional hazards assumption
ph_test_pvalues = {}
assumption_results_list = cph.check_assumptions(df_cox, p_value_threshold=0.05, show_plots=False)
for res_df in assumption_results_list:
    covariate_name = res_df.index.name
    p_value = res_df.loc['test_statistic', 'p']
    ph_test_pvalues[covariate_name] = float(p_value)


# --- Objective 2: Causal Impact (DiD) ---

# Build hospital-year panel for output
did_panel = df.groupby(['hospital_id', 'year']).agg(
    antibiotic_duration=('antibiotic_duration', 'mean')
).reset_index()

# Build panel for estimation
panel_for_est = df.groupby(['hospital_id', 'year']).agg(
    antibiotic_duration=('antibiotic_duration', 'mean'),
    asp_adoption_year=('asp_adoption_year', 'first')
).reset_index()

# Run plain TWFE model
panel_twfe = panel_for_est.copy()
panel_twfe['asp_active'] = ((panel_twfe['year'] >= panel_twfe['asp_adoption_year']) &
                            (panel_twfe['asp_adoption_year'] != 9999)).astype(int)
panel_twfe = panel_twfe.set_index(['hospital_id', 'year'])
twfe_mod = PanelOLS.from_formula('antibiotic_duration ~ 1 + asp_active + EntityEffects + TimeEffects', data=panel_twfe)
twfe_res = twfe_mod.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = float(twfe_res.params['asp_active'])

# Prepare for robust DiD (Sun-Abraham style with never-treated controls)
panel_robust = panel_for_est.copy()
panel_robust['cohort'] = panel_robust['asp_adoption_year'].replace(9999, np.inf)
panel_robust['event_time'] = panel_robust['year'] - panel_robust['cohort']

# Create event time dummies with valid names, omitting t-1 as the reference period
event_times = sorted([t for t in panel_robust['event_time'].unique() if np.isfinite(t)])
ref_period = -1
dummy_cols = []
col_to_time_map = {}

for t in event_times:
    if t != ref_period:
        t_int = int(t)
        if t_int < 0:
            col_name = f"et_neg{abs(t_int)}"
        else:
            col_name = f"et_{t_int}"
        
        panel_robust[col_name] = (panel_robust['event_time'] == t).astype(int)
        dummy_cols.append(col_name)
        col_to_time_map[col_name] = t_int

# Interact dummies with treatment indicator (being part of any treated cohort)
treated_indicator = (panel_robust['cohort'] != np.inf).astype(int)
for col in dummy_cols:
    panel_robust[col] = panel_robust[col] * treated_indicator

# Set up and run the regression
panel_robust = panel_robust.set_index(['hospital_id', 'year'])
formula = f'antibiotic_duration ~ 1 + {" + ".join(dummy_cols)} + EntityEffects + TimeEffects'
mod_robust = PanelOLS.from_formula(formula, data=panel_robust)
res_robust = mod_robust.fit(cov_type='clustered', cluster_entity=True)

# Extract event study results
event_study_list = []
for col, time in col_to_time_map.items():
    if col in res_robust.params:
        coef = res_robust.params[col]
        ci = res_robust.conf_int().loc[col]
        event_study_list.append({'event_time': time, 'coef': coef, 'ci_low': ci['lower'], 'ci_high': ci['upper']})
event_study = pd.DataFrame(event_study_list).sort_values('event_time').reset_index(drop=True)

# Extract pre-trend test results (using t=-2, if available)
pre_trend_coeff_name = 'et_neg2'
if pre_trend_coeff_name in res_robust.params:
    pre_trend_coeff = float(res_robust.params[pre_trend_coeff_name])
    pre_trend_pvalue = float(res_robust.pvalues[pre_trend_coeff_name])
    pre_trend_ci_series = res_robust.conf_int().loc[pre_trend_coeff_name]
    pre_trend_ci = [float(pre_trend_ci_series['lower']), float(pre_trend_ci_series['upper'])]
else:
    pre_trend_coeff, pre_trend_pvalue, pre_trend_ci = 0.0, 1.0, [0.0, 0.0]

# Calculate robust ATT (weighted average of post-treatment coefficients)
post_treatment_cols = [col for col, time in col_to_time_map.items() if time >= 0 and col in res_robust.params]
if post_treatment_cols:
    post_coeffs = res_robust.params[post_treatment_cols]
    weights = panel_robust[post_treatment_cols].sum()
    if weights.sum() > 0:
        did_att_estimate = float(np.average(post_coeffs, weights=weights))
    else:
        did_att_estimate = 0.0
else:
    did_att_estimate = 0.0


# --- Objective 3: Treatment Heterogeneity ---

# Prepare data for CATE estimation
df_cate = df.copy()
outcome, treatment = 'antibiotic_duration', 'asp_active'

features_base = ['age', 'comorbidity_score', 'immune_compromised', 'severity_score']
features_cat = ['sex', 'infection_type', 'ward', 'hospital_size', 'region', 'hospital_id', 'year']
X_df = pd.get_dummies(df_cate[features_base + features_cat], columns=features_cat, drop_first=True, dtype=float)
X, y, T = X_df.values, df_cate[outcome].values, df_cate[treatment].values

# T-Learner implementation
X_treat, y_treat = X[T == 1], y[T == 1]
X_control, y_control = X[T == 0], y[T == 0]

m1 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1).fit(X_treat, y_treat)
m0 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1).fit(X_control, y_control)

cate_estimates = m1.predict(X) - m0.predict(X)

# Bootstrap CI for difference in CATE (elderly vs. young)
n_bootstraps = 100
bootstrap_diffs = []
age_col = df_cate['age'].values
indices = np.arange(len(df_cate))

for _ in range(n_bootstraps):
    boot_indices = np.random.choice(indices, size=len(indices), replace=True)
    X_boot, y_boot, T_boot, age_boot = X[boot_indices], y[boot_indices], T[boot_indices], age_col[boot_indices]
    
    X_treat_boot, y_treat_boot = X_boot[T_boot == 1], y_boot[T_boot == 1]
    X_control_boot, y_control_boot = X_boot[T_boot == 0], y_boot[T_boot == 0]
    
    if len(X_treat_boot) < 2 or len(X_control_boot) < 2: continue
        
    m1_boot = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=42, n_jobs=-1).fit(X_treat_boot, y_treat_boot)
    m0_boot = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=42, n_jobs=-1).fit(X_control_boot, y_control_boot)
    
    cate_boot = m1_boot.predict(X_boot) - m0_boot.predict(X_boot)
    
    mask_elderly = age_boot >= 65
    mask_young = age_boot < 40
    
    if mask_elderly.sum() > 0 and mask_young.sum() > 0:
        mean_cate_elderly = cate_boot[mask_elderly].mean()
        mean_cate_young = cate_boot[mask_young].mean()
        bootstrap_diffs.append(mean_cate_elderly - mean_cate_young)

cate_elderly_minus_young_ci = [float(np.percentile(bootstrap_diffs, 2.5)), float(np.percentile(bootstrap_diffs, 97.5))]


# --- Objective 4: Hierarchical Modeling ---

# Define formula for mixed-effects and OLS models
fixed_effects = [
    'asp_active', 'age', 'comorbidity_score', 'immune_compromised', 'severity_score', 'year',
    'C(sex)', 'C(infection_type)', 'C(hospital_size)', 'C(region)', 'C(ward)'
]
formula_fixed = f'antibiotic_duration ~ {" + ".join(fixed_effects)}'

# Fit mixed-effects model
mixed_model = smf.mixedlm(f'{formula_fixed}', data=df, groups=df['hospital_id'])
mixed_model_fit = mixed_model.fit(method='ML', reml=False)

# Fit naive OLS model for comparison
ols_model = smf.ols(f'{formula_fixed}', data=df)
ols_model_fit = ols_model.fit()

# Extract required outputs
mixed_model_aic = float(mixed_model_fit.aic)
naive_ols_aic = float(ols_model_fit.aic)
mixed_asp_coeff = float(mixed_model_fit.params['asp_active'])

# Calculate ICC
random_effect_variance = float(mixed_model_fit.cov_re.iloc[0, 0])
residual_variance = float(mixed_model_fit.scale)
icc_value = random_effect_variance / (random_effect_variance + residual_variance)