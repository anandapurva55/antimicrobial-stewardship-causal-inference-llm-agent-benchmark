import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from lifelines import CoxPHFitter
from linearmodels.panel import PanelOLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

# Objective 1: Survival Analysis

# Prepare data for lifelines (requires one-hot encoding)
df_cox = df.copy()
categorical_cols_cox = ['sex', 'infection_type', 'hospital_size', 'region']
df_cox = pd.get_dummies(df_cox, columns=categorical_cols_cox, drop_first=True)

# Fit Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(df_cox,
        duration_col='time_to_improvement',
        event_col='event_observed',
        cluster_col='hospital_id')

cox_hr_asp = cph.hazard_ratios_['asp_active']
cox_se_asp = cph.summary.loc['asp_active', 'se(log(HR))']

# Test Proportional Hazards assumption
# The check_assumptions method returns a list of dataframes, the first one contains the test results
assumption_results = cph.check_assumptions(df_cox, show_plots=False, p_value_threshold=1.0)
ph_test_pvalues = {
    row.name: row['p'] for _, row in assumption_results[0].iterrows()
}


# Objective 2: Causal Impact (DiD)

# Create hospital-year panel
did_panel = df.groupby(['hospital_id', 'year']).agg(
    antibiotic_duration=('antibiotic_duration', 'mean')
).reset_index()

hospital_info = df[['hospital_id', 'asp_adoption_year']].drop_duplicates()
did_panel = pd.merge(did_panel, hospital_info, on='hospital_id', how='left')

# Define treatment status and relative time
did_panel['asp_active'] = (did_panel['year'] >= did_panel['asp_adoption_year']).astype(int)
did_panel['event_time'] = did_panel['year'] - did_panel['asp_adoption_year']
# For never-treated units, set event_time to a value that won't be used
did_panel.loc[did_panel['asp_adoption_year'] == 9999, 'event_time'] = -1000

# Event Study and Parallel Trends Test
# Use -1 as the reference period
# Bin event times outside a reasonable window for stability
event_times_actual = did_panel[did_panel['asp_adoption_year'] != 9999]['event_time']
min_event_time = int(event_times_actual.min()) if not event_times_actual.empty else -2
max_event_time = int(event_times_actual.max()) if not event_times_actual.empty else 2

did_panel['event_time_binned'] = did_panel['event_time'].copy()
did_panel.loc[did_panel['event_time'] <= min_event_time, 'event_time_binned'] = min_event_time
did_panel.loc[did_panel['event_time'] >= max_event_time, 'event_time_binned'] = max_event_time
did_panel.loc[did_panel['asp_adoption_year'] == 9999, 'event_time_binned'] = -1000

event_study_data = did_panel[did_panel['asp_adoption_year'] != 9999].copy()

# Fit the event study model with -1 as the reference category
formula = 'antibiotic_duration ~ C(hospital_id) + C(year) + C(event_time_binned, Treatment(reference=-1))'
event_study_model = smf.ols(formula, data=event_study_data).fit(
    cov_type='cluster', cov_kwds={'groups': event_study_data['hospital_id']}
)

# Extract results for the event study plot
event_study_results = []
for var in event_study_model.params.index:
    if 'event_time_binned' in var:
        event_time = int(float(var.split('.')[-1].replace(']', '')))
        coef = event_study_model.params[var]
        ci = event_study_model.conf_int().loc[var]
        event_study_results.append({
            'event_time': event_time,
            'coef': coef,
            'ci_low': ci[0],
            'ci_high': ci[1]
        })

event_study = pd.DataFrame(event_study_results).sort_values('event_time').reset_index(drop=True)

# Parallel pre-trend test (coefficient for t=-2)
pre_trend_time = -2
pre_trend_var_name = f'C(event_time_binned, Treatment(reference=-1))[T.{pre_trend_time:.1f}]'
if pre_trend_var_name in event_study_model.params.index:
    pre_trend_coeff = event_study_model.params[pre_trend_var_name]
    pre_trend_pvalue = event_study_model.pvalues[pre_trend_var_name]
    pre_trend_ci = event_study_model.conf_int().loc[pre_trend_var_name].tolist()
else:
    pre_trend_coeff = np.nan
    pre_trend_pvalue = np.nan
    pre_trend_ci = [np.nan, np.nan]

# TWFE ATT Estimate
panel_for_twfe = did_panel.set_index(['hospital_id', 'year'])
twfe_model = PanelOLS(panel_for_twfe.antibiotic_duration, sm.add_constant(panel_for_twfe['asp_active']),
                      entity_effects=True, time_effects=True)
twfe_results = twfe_model.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = twfe_results.params['asp_active']

# Robust Staggered DiD ATT (using never-treated as control group for each cohort)
g_2020 = hospital_info[hospital_info['asp_adoption_year'] == 2020]['hospital_id'].unique()
g_2021 = hospital_info[hospital_info['asp_adoption_year'] == 2021]['hospital_id'].unique()
g_never = hospital_info[hospital_info['asp_adoption_year'] == 9999]['hospital_id'].unique()

atts = []
weights = []

if len(g_2020) > 0 and len(g_never) > 0:
    df_2020 = did_panel[did_panel['hospital_id'].isin(np.append(g_2020, g_never))].copy()
    df_2020['treat'] = df_2020['hospital_id'].isin(g_2020).astype(int)
    df_2020['post'] = (df_2020['year'] >= 2020).astype(int)
    mod_2020 = smf.ols('antibiotic_duration ~ treat * post + C(hospital_id) + C(year)', data=df_2020).fit(
        cov_type='cluster', cov_kwds={'groups': df_2020['hospital_id']}
    )
    atts.append(mod_2020.params['treat:post'])
    weights.append(len(g_2020))

if len(g_2021) > 0 and len(g_never) > 0:
    df_2021 = did_panel[did_panel['hospital_id'].isin(np.append(g_2021, g_never))].copy()
    df_2021['treat'] = df_2021['hospital_id'].isin(g_2021).astype(int)
    df_2021['post'] = (df_2021['year'] >= 2021).astype(int)
    mod_2021 = smf.ols('antibiotic_duration ~ treat * post + C(hospital_id) + C(year)', data=df_2021).fit(
        cov_type='cluster', cov_kwds={'groups': df_2021['hospital_id']}
    )
    atts.append(mod_2021.params['treat:post'])
    weights.append(len(g_2021))

did_att_estimate = np.average(atts, weights=weights) if atts else np.nan


# Objective 3: Treatment Heterogeneity (CATE)

# Prepare data for CATE estimation
df_cate = df.copy()
Y = df_cate['antibiotic_duration'].values
T = df_cate['asp_active'].values
X_features = ['age', 'comorbidity_score', 'severity_score', 'year', 'sex',
              'immune_compromised', 'infection_type', 'hospital_size', 'region', 'hospital_id']
X = pd.get_dummies(df_cate[X_features], drop_first=True)

# X-Learner Implementation
model_params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 10, 'random_state': 42}
mu_0 = RandomForestRegressor(**model_params)
mu_1 = RandomForestRegressor(**model_params)
tau_0 = RandomForestRegressor(**model_params)
tau_1 = RandomForestRegressor(**model_params)
prop_model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)

# Stage 1
mu_0.fit(X[T==0], Y[T==0])
mu_1.fit(X[T==1], Y[T==1])

# Stage 2
D_tilde_0 = mu_1.predict(X[T==0]) - Y[T==0]
D_tilde_1 = Y[T==1] - mu_0.predict(X[T==1])
tau_0.fit(X[T==0], D_tilde_0)
tau_1.fit(X[T==1], D_tilde_1)

# Stage 3
prop_model.fit(X, T)
g = prop_model.predict_proba(X)[:, 1]
cate_estimates = g * tau_0.predict(X) + (1 - g) * tau_1.predict(X)

# Bootstrap CI for CATE difference
n_bootstraps = 200 # Reduced for reasonable execution time
cate_diffs = []

for _ in range(n_bootstraps):
    boot_indices = np.random.choice(df.index, size=len(df), replace=True)
    df_boot = df.iloc[boot_indices].reset_index(drop=True)

    Y_boot = df_boot['antibiotic_duration'].values
    T_boot = df_boot['asp_active'].values
    X_boot_raw = df_boot[X_features]
    X_boot = pd.get_dummies(X_boot_raw, drop_first=True).reindex(columns=X.columns, fill_value=0)
    
    T0_mask, T1_mask = (T_boot == 0), (T_boot == 1)
    if T0_mask.sum() < 2 or T1_mask.sum() < 2: continue

    mu_0_boot = RandomForestRegressor(**model_params).fit(X_boot[T0_mask], Y_boot[T0_mask])
    mu_1_boot = RandomForestRegressor(**model_params).fit(X_boot[T1_mask], Y_boot[T1_mask])
    
    D_tilde_0_boot = mu_1_boot.predict(X_boot[T0_mask]) - Y_boot[T0_mask]
    D_tilde_1_boot = Y_boot[T1_mask] - mu_0_boot.predict(X_boot[T1_mask])
    
    tau_0_boot = RandomForestRegressor(**model_params).fit(X_boot[T0_mask], D_tilde_0_boot)
    tau_1_boot = RandomForestRegressor(**model_params).fit(X_boot[T1_mask], D_tilde_1_boot)
    
    prop_model_boot = LogisticRegression(solver='liblinear', max_iter=1000).fit(X_boot, T_boot)
    g_boot = prop_model_boot.predict_proba(X_boot)[:, 1]
    
    cate_boot = g_boot * tau_0_boot.predict(X_boot) + (1 - g_boot) * tau_1_boot.predict(X_boot)
    
    mean_cate_elderly = cate_boot[df_boot['age'] >= 65].mean()
    mean_cate_young = cate_boot[df_boot['age'] < 40].mean()
    
    if not (np.isnan(mean_cate_elderly) or np.isnan(mean_cate_young)):
        cate_diffs.append(mean_cate_elderly - mean_cate_young)

cate_elderly_minus_young_ci = np.percentile(cate_diffs, [2.5, 97.5]).tolist() if cate_diffs else [np.nan, np.nan]


# Objective 4: Hierarchical Modeling

# Fit mixed-effects model
formula_mixed = 'antibiotic_duration ~ asp_active + age + C(sex) + comorbidity_score + immune_compromised + C(infection_type) + severity_score + C(year)'
mixed_model = smf.mixedlm(formula_mixed, df, groups=df['hospital_id'])
mixed_results = mixed_model.fit(method='ML')

# Extract results
mixed_model_aic = mixed_results.aic
mixed_asp_coeff = mixed_results.params['asp_active']

# ICC calculation
random_effect_variance = float(mixed_results.cov_re.iloc[0, 0])
residual_variance = mixed_results.scale
icc_value = random_effect_variance / (random_effect_variance + residual_variance)

# Fit naive OLS for comparison
formula_ols = formula_mixed + ' + C(hospital_id)'
ols_model = smf.ols(formula_ols, df).fit()
naive_ols_aic = ols_model.aic