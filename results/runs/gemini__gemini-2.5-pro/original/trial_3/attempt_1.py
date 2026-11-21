import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from linearmodels.panel import PanelOLS
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import norm

# --- 1. SURVIVAL ANALYSIS ---

# Prepare data for lifelines
df_cox = df.copy()
categorical_cols_cox = ['hospital_size', 'region', 'ward', 'sex', 'infection_type']
df_cox = pd.get_dummies(df_cox, columns=categorical_cols_cox, drop_first=True, dtype=float)

# Define covariates for the model
base_covariates_cox = ['age', 'comorbidity_score', 'immune_compromised', 'severity_score', 'asp_active']
dummy_cols_cox = [col for col in df_cox.columns if any(cat_col in col for cat_col in categorical_cols_cox)]
all_covariates_cox = base_covariates_cox + dummy_cols_cox

# Fit Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(df_cox,
        duration_col='time_to_improvement',
        event_col='event_observed',
        formula=" + ".join(all_covariates_cox),
        cluster_col='hospital_id')

# Extract required outputs
cox_hr_asp = cph.hazard_ratios_['asp_active']
cox_se_asp = cph.summary.loc['asp_active', 'se(coef)']

# Test proportional hazards assumption
ph_test_results = cph.check_assumptions(df_cox, show_plots=False, p_value_threshold=1)
ph_test_pvalues = {var.replace('var=', ''): p_val for var, p_val in ph_test_results.p_value.items()}


# --- 2. CAUSAL IMPACT (DiD) ---

# Build hospital-year panel
did_panel_agg = df.groupby(['hospital_id', 'year']).agg(
    antibiotic_duration=('antibiotic_duration', 'mean')
).reset_index()
hospital_info = df[['hospital_id', 'asp_adoption_year']].drop_duplicates()
did_panel = pd.merge(did_panel_agg, hospital_info, on='hospital_id')

# Set index for PanelOLS
did_panel = did_panel.set_index(['hospital_id', 'year'])

# Create event time variable
# For never-treated units, set event_time to a value that won't be used (e.g., -999)
did_panel['event_time'] = np.where(
    did_panel['asp_adoption_year'] == 9999,
    -999,
    did_panel.index.get_level_values('year') - did_panel['asp_adoption_year']
)

# Create event time dummies, omitting t=-1 (reference period)
# Binning endpoints for robustness, as not all cohorts have all leads/lags
min_event_time = int(did_panel[did_panel['asp_adoption_year'] != 9999]['event_time'].min())
max_event_time = int(did_panel[did_panel['asp_adoption_year'] != 9999]['event_time'].max())

# Cap event time for stable dummy creation
did_panel['event_time_capped'] = did_panel['event_time'].clip(lower=min_event_time, upper=max_event_time)

event_dummies = pd.get_dummies(did_panel['event_time_capped'], prefix='t', drop_first=False)
if 't_-1.0' in event_dummies.columns:
    event_dummies = event_dummies.drop(columns=['t_-1.0'])
if 't_-999.0' in event_dummies.columns:
    event_dummies = event_dummies.drop(columns=['t_-999.0'])

did_panel_es = pd.concat([did_panel, event_dummies], axis=1)

# Run event study regression
exog_vars_es = list(event_dummies.columns)
exog_es = sm.add_constant(did_panel_es[exog_vars_es])
es_model = PanelOLS(did_panel_es['antibiotic_duration'], exog_es, entity_effects=True, time_effects=True)
es_results = es_model.fit(cov_type='clustered', cluster_entity=True)

# Extract pre-trend test results (for t=-2)
pre_trend_col = 't_-2.0'
if pre_trend_col in es_results.params:
    pre_trend_coeff = es_results.params[pre_trend_col]
    pre_trend_pvalue = es_results.pvalues[pre_trend_col]
    ci = es_results.conf_int().loc[pre_trend_col]
    pre_trend_ci = [ci['lower'], ci['upper']]
else: # Fallback if t=-2 is not available
    pre_trend_coeff = np.nan
    pre_trend_pvalue = np.nan
    pre_trend_ci = [np.nan, np.nan]

# Create event study DataFrame
event_study_list = []
for t in sorted([float(c.replace('t_', '')) for c in exog_vars_es]):
    col = f"t_{t}"
    event_study_list.append({
        'event_time': int(t),
        'coef': es_results.params[col],
        'ci_low': es_results.conf_int().loc[col, 'lower'],
        'ci_high': es_results.conf_int().loc[col, 'upper']
    })
# Add reference period t=-1
event_study_list.append({'event_time': -1, 'coef': 0.0, 'ci_low': 0.0, 'ci_high': 0.0})
event_study = pd.DataFrame(event_study_list).sort_values('event_time').reset_index(drop=True)

# Run TWFE model
did_panel['asp_active'] = (did_panel.index.get_level_values('year') >= did_panel['asp_adoption_year']).astype(int)
exog_twfe = sm.add_constant(did_panel['asp_active'])
twfe_model = PanelOLS(did_panel['antibiotic_duration'], exog_twfe, entity_effects=True, time_effects=True)
twfe_results = twfe_model.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = twfe_results.params['asp_active']

# Estimate robust ATT (Sun & Abraham / interaction-weighted style)
post_treatment_rows = event_study[event_study['event_time'] >= 0]
weights = did_panel[did_panel['asp_active'] == 1]['event_time'].value_counts()
weighted_sum_coef = 0
total_weight = 0
for _, row in post_treatment_rows.iterrows():
    t = row['event_time']
    if t in weights.index:
        weight = weights[t]
        weighted_sum_coef += row['coef'] * weight
        total_weight += weight
did_att_estimate = weighted_sum_coef / total_weight if total_weight > 0 else 0.0


# --- 3. TREATMENT HETEROGENEITY ---

# Prepare data for T-learner
df_cate = df.copy()
y = df_cate['antibiotic_duration']
T = df_cate['asp_active']

# Define features, including one-hot encoded hospital and year
categorical_features = ['sex', 'infection_type', 'hospital_id', 'year']
numeric_features = ['age', 'comorbidity_score', 'severity_score', 'immune_compromised']
X_pre_dummy = df_cate[numeric_features + categorical_features]
X = pd.get_dummies(X_pre_dummy, columns=categorical_features, drop_first=True, dtype=float)

# T-Learner implementation
X_treat = X[T == 1]
y_treat = y[T == 1]
X_control = X[T == 0]
y_control = y[T == 0]

# Model for treated group
m1 = GradientBoostingRegressor(random_state=42)
m1.fit(X_treat, y_treat)

# Model for control group
m0 = GradientBoostingRegressor(random_state=42)
m0.fit(X_control, y_control)

# Estimate CATE for all individuals
y1_pred = m1.predict(X)
y0_pred = m0.predict(X)
cate_estimates = y1_pred - y0_pred

# Bootstrap CI for difference in CATEs
n_bootstraps = 200
diffs = []
df_boot_base = df.copy()
df_boot_base['original_index'] = df.index

for _ in range(n_bootstraps):
    # Resample with replacement
    boot_indices = np.random.choice(df.index, size=len(df), replace=True)
    df_boot = df.iloc[boot_indices].copy()

    # Prepare bootstrapped data for T-learner
    y_boot = df_boot['antibiotic_duration']
    T_boot = df_boot['asp_active']
    X_pre_dummy_boot = df_boot[numeric_features + categorical_features]
    X_boot = pd.get_dummies(X_pre_dummy_boot, columns=categorical_features, drop_first=True, dtype=float)
    
    # Align columns with original X
    X_boot = X_boot.reindex(columns=X.columns, fill_value=0)

    X_treat_boot = X_boot[T_boot == 1]
    y_treat_boot = y_boot[T_boot == 1]
    X_control_boot = X_boot[T_boot == 0]
    y_control_boot = y_boot[T_boot == 0]

    if len(X_treat_boot) > 0 and len(X_control_boot) > 0:
        m1_boot = GradientBoostingRegressor(random_state=42).fit(X_treat_boot, y_treat_boot)
        m0_boot = GradientBoostingRegressor(random_state=42).fit(X_control_boot, y_control_boot)

        cate_boot = m1_boot.predict(X_boot) - m0_boot.predict(X_boot)

        # Calculate difference in means for the age groups
        elderly_mask_boot = (df_boot['age'] >= 65)
        young_mask_boot = (df_boot['age'] < 40)
        
        if elderly_mask_boot.sum() > 0 and young_mask_boot.sum() > 0:
            mean_cate_elderly = np.mean(cate_boot[elderly_mask_boot])
            mean_cate_young = np.mean(cate_boot[young_mask_boot])
            diffs.append(mean_cate_elderly - mean_cate_young)

# Calculate 95% CI
cate_elderly_minus_young_ci = list(np.percentile(diffs, [2.5, 97.5]))


# --- 4. HIERARCHICAL MODELING ---

# Fit mixed-effects model
formula = 'antibiotic_duration ~ asp_active + age + C(sex) + comorbidity_score + immune_compromised + C(infection_type) + severity_score + C(year)'
mixed_model = smf.mixedlm(formula, data=df, groups=df["hospital_id"])
# Fit with ML for AIC comparison
mixed_results = mixed_model.fit(method='ml', reml=False)

# Extract mixed model outputs
mixed_model_aic = mixed_results.aic
mixed_asp_coeff = mixed_results.params['asp_active']
random_effect_variance = float(mixed_results.cov_re.iloc[0,0])
residual_variance = mixed_results.scale
icc_value = random_effect_variance / (random_effect_variance + residual_variance)

# Fit naive OLS model for comparison
ols_model = smf.ols(formula, data=df)
ols_results = ols_model.fit()
naive_ols_aic = ols_results.aic