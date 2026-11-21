import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from linearmodels.panel import PanelOLS
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm

# --- DATA PREPARATION for models ---
# Create a copy to avoid modifying the original df
df_models = df.copy()
# Set categorical types for statsmodels formula API
categorical_cols = ['hospital_size', 'region', 'ward', 'sex', 'age_group', 'infection_type', 'hospital_id', 'year']
for col in categorical_cols:
    if col in df_models.columns:
        df_models[col] = df_models[col].astype('category')


# === OBJECTIVE 1: SURVIVAL ANALYSIS ===

# Prepare data with dummies for lifelines
df_cox = pd.get_dummies(df, columns=['hospital_size', 'region', 'ward', 'sex', 'infection_type'], drop_first=True)
covariates_cox = [
    'asp_active', 'age', 'comorbidity_score', 'immune_compromised', 'severity_score'
] + [c for c in df_cox.columns if c.startswith(('hospital_size_', 'region_', 'ward_', 'sex_', 'infection_type_'))]

cph = CoxPHFitter()
cph.fit(df_cox, 
        duration_col='time_to_improvement', 
        event_col='event_observed', 
        formula=" + ".join(covariates_cox),
        cluster_col='hospital_id')

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
# The standard error is for the log-hazard coefficient
cox_se_asp = float(cph.summary.loc['asp_active', 'se(coef)'])

# Test Proportional Hazards assumption
# The check_assumptions method returns a list of StatisticalResult objects
ph_test_results = cph.check_assumptions(df_cox, show_plots=False, p_value_threshold=1.0)
ph_test_pvalues = {result.name[0]: float(result.p_value) for result in ph_test_results}


# === OBJECTIVE 2: CAUSAL IMPACT (DiD) ===

# Create hospital-year panel
did_panel = df.groupby(['hospital_id', 'year']).agg(
    antibiotic_duration=('antibiotic_duration', 'mean'),
    asp_adoption_year=('asp_adoption_year', 'first')
).reset_index()
did_panel['asp_active'] = (did_panel['year'] >= did_panel['asp_adoption_year']).astype(int)

# Naive TWFE model
panel_data_for_twfe = did_panel.set_index(['hospital_id', 'year'])
twfe_model = PanelOLS.from_formula('antibiotic_duration ~ asp_active + EntityEffects + TimeEffects', data=panel_data_for_twfe)
twfe_results = twfe_model.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = float(twfe_results.params['asp_active'])

# Event Study for Parallel Trends Test
did_panel['event_time'] = np.where(did_panel['asp_adoption_year'] < 9999, did_panel['year'] - did_panel['asp_adoption_year'], -999)

# Create event time dummies, omitting ref period t-1 and never-treated group
event_dummies = pd.get_dummies(did_panel['event_time'], prefix='event_time', drop_first=False)
event_dummies.columns = [c.replace('.0', '') for c in event_dummies.columns]
ref_period_col = 'event_time_-1'
never_treated_col = 'event_time_-999'
cols_to_drop = [c for c in [ref_period_col, never_treated_col] if c in event_dummies.columns]
if cols_to_drop:
    event_dummies.drop(columns=cols_to_drop, inplace=True)

reg_df = pd.concat([did_panel, event_dummies], axis=1)

# Quote column names with special characters for the patsy formula
quoted_event_cols = [f"Q('{col}')" for col in event_dummies.columns]
formula = f"antibiotic_duration ~ {' + '.join(quoted_event_cols)} + C(hospital_id) + C(year)"

event_study_model = smf.ols(formula, data=reg_df).fit(cov_type='cluster', cov_kwds={'groups': reg_df['hospital_id']})

# Extract pre-trend coefficient for t-2, using the quoted name
pre_trend_coeff_name_unquoted = 'event_time_-2'
pre_trend_coeff_name_quoted = f"Q('{pre_trend_coeff_name_unquoted}')"

if pre_trend_coeff_name_quoted in event_study_model.params:
    pre_trend_coeff = float(event_study_model.params[pre_trend_coeff_name_quoted])
    pre_trend_pvalue = float(event_study_model.pvalues[pre_trend_coeff_name_quoted])
    pre_trend_ci_raw = event_study_model.conf_int().loc[pre_trend_coeff_name_quoted]
    pre_trend_ci = [float(pre_trend_ci_raw.iloc[0]), float(pre_trend_ci_raw.iloc[1])]
else:
    pre_trend_coeff = np.nan
    pre_trend_pvalue = np.nan
    pre_trend_ci = [np.nan, np.nan]

# Create event study DataFrame
results_list = []
for var in event_dummies.columns:
    var_quoted = f"Q('{var}')"
    if var_quoted in event_study_model.params:
        time = int(var.split('_')[-1])
        coef = event_study_model.params[var_quoted]
        ci_low, ci_high = event_study_model.conf_int().loc[var_quoted]
        results_list.append({'event_time': time, 'coef': coef, 'ci_low': ci_low, 'ci_high': ci_high})

if ref_period_col.replace('event_time_', '') not in [r['event_time'] for r in results_list]:
    results_list.append({'event_time': -1, 'coef': 0.0, 'ci_low': 0.0, 'ci_high': 0.0})
event_study = pd.DataFrame(results_list).sort_values('event_time').reset_index(drop=True)

# Robust DiD estimator (using never-treated as control for each cohort)
hospitals = df[['hospital_id', 'asp_adoption_year']].drop_duplicates()
cohorts = sorted(hospitals[hospitals['asp_adoption_year'] < 9999]['asp_adoption_year'].unique())
never_treated_hospitals = hospitals[hospitals['asp_adoption_year'] == 9999]['hospital_id'].tolist()
att_estimates = []
cohort_sizes = []

for cohort_year in cohorts:
    treated_hospitals = hospitals[hospitals['asp_adoption_year'] == cohort_year]['hospital_id'].tolist()
    sub_panel = did_panel[did_panel['hospital_id'].isin(treated_hospitals + never_treated_hospitals)].copy()
    sub_panel['treat'] = sub_panel['hospital_id'].isin(treated_hospitals).astype(int)
    sub_panel['post'] = (sub_panel['year'] >= cohort_year).astype(int)
    
    formula_did = 'antibiotic_duration ~ treat * post + C(hospital_id) + C(year)'
    did_model = smf.ols(formula_did, data=sub_panel).fit(cov_type='cluster', cov_kwds={'groups': sub_panel['hospital_id']})
    
    att_g = did_model.params.get('treat:post', 0)
    att_estimates.append(att_g)
    cohort_sizes.append(len(treated_hospitals))

did_att_estimate = float(np.average(att_estimates, weights=cohort_sizes)) if sum(cohort_sizes) > 0 else np.nan


# === OBJECTIVE 3: TREATMENT HETEROGENEITY (CATE) ===

features = ['age', 'comorbidity_score', 'immune_compromised', 'severity_score']
categorical_features = ['hospital_size', 'region', 'ward', 'sex', 'infection_type', 'hospital_id', 'year']
X = pd.get_dummies(df[features + categorical_features], columns=categorical_features, drop_first=True)
y = df['antibiotic_duration']
T = df['asp_active']

# T-Learner implementation
model_t1 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1)
model_t0 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1)

model_t1.fit(X[T == 1], y[T == 1])
model_t0.fit(X[T == 0], y[T == 0])

cate_estimates = model_t1.predict(X) - model_t0.predict(X)

# Bootstrap CI for CATE difference
n_bootstraps = 200
bootstrap_diffs = []
for i in range(n_bootstraps):
    sample_idx = df.index.to_series().sample(n=len(df), replace=True, random_state=i)
    df_sample = df.loc[sample_idx]
    
    X_sample = pd.get_dummies(df_sample[features + categorical_features], columns=categorical_features, drop_first=True)
    y_sample = df_sample['antibiotic_duration']
    T_sample = df_sample['asp_active']
    
    X_sample = X_sample.reindex(columns=X.columns, fill_value=0)

    m1 = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=42, n_jobs=-1)
    m0 = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=42, n_jobs=-1)
    
    if T_sample.sum() > 0 and (1-T_sample).sum() > 0:
        m1.fit(X_sample[T_sample == 1], y_sample[T_sample == 1])
        m0.fit(X_sample[T_sample == 0], y_sample[T_sample == 0])
        cate_sample = m1.predict(X_sample) - m0.predict(X_sample)
        
        elderly_mask = df_sample['age_group'] == '65+'
        young_mask = df_sample['age_group'] == '<40'
        
        if elderly_mask.sum() > 0 and young_mask.sum() > 0:
            mean_cate_elderly = cate_sample[elderly_mask].mean()
            mean_cate_young = cate_sample[young_mask].mean()
            bootstrap_diffs.append(mean_cate_elderly - mean_cate_young)

cate_elderly_minus_young_ci = [float(np.percentile(bootstrap_diffs, 2.5)), float(np.percentile(bootstrap_diffs, 97.5))]


# === OBJECTIVE 4: HIERARCHICAL MODELING ===

fixed_effects_formula = "asp_active + age + comorbidity_score + immune_compromised + severity_score + C(hospital_size) + C(region) + C(ward) + C(sex) + C(infection_type) + C(year)"
outcome = "antibiotic_duration"

# Mixed-effects model
mixed_model = smf.mixedlm(f"{outcome} ~ {fixed_effects_formula}", 
                          data=df_models, 
                          groups=df_models["hospital_id"])
mixed_model_results = mixed_model.fit(method='ML')

mixed_model_aic = float(mixed_model_results.aic)
mixed_asp_coeff = float(mixed_model_results.params['asp_active'])
random_effect_variance = float(mixed_model_results.cov_re.iloc[0, 0])
residual_variance = float(mixed_model_results.scale)
icc_value = random_effect_variance / (random_effect_variance + residual_variance)

# Naive OLS model for comparison
ols_model = smf.ols(f"{outcome} ~ {fixed_effects_formula}", data=df_models)
ols_model_results = ols_model.fit()
naive_ols_aic = float(ols_model_results.aic)