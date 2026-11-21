import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from lifelines import KaplanMeierFitter, CoxPHFitter
from linearmodels.panel import PanelOLS
from sklearn.ensemble import RandomForestRegressor

# Ensure reproducibility
np.random.seed(42)

# --- Data Preparation ---
# Censor asp_adoption_year for never-adopters to NaN for easier calculations
df['asp_adoption_year_clean'] = df['asp_adoption_year'].replace(9999, np.nan)


# === OBJECTIVE 1: SURVIVAL ANALYSIS ===

# Prepare data for lifelines formula API
df_cox = df.copy()
# Use a formula to let lifelines handle dummy encoding
cox_formula = "asp_active + age + comorbidity_score + immune_compromised + severity_score + C(sex) + C(infection_type) + C(hospital_size) + C(region)"

cph = CoxPHFitter()
cph.fit(df_cox, 
        duration_col='time_to_improvement', 
        event_col='event_observed', 
        formula=cox_formula,
        cluster_col='hospital_id')

# Extract required values
cox_hr_asp = cph.hazard_ratios_['asp_active']
# SE of log-hazard is what is typically reported
cox_se_asp = cph.summary.loc['asp_active', 'se(coef)']

# Test Proportional Hazards assumption
# The check_assumptions method returns a summary DataFrame. We need to parse p-values.
assumption_results = cph.check_assumptions(df_cox, show_plots=False, p_value_threshold=1.0)
ph_test_pvalues = assumption_results['p'].to_dict()


# === OBJECTIVE 2: CAUSAL IMPACT (DiD) ===

# 1. Build hospital-year panel
did_panel_agg = df.groupby(['hospital_id', 'year'])['antibiotic_duration'].mean().reset_index()
hospital_info = df[['hospital_id', 'asp_adoption_year']].drop_duplicates()
did_panel_full = pd.merge(did_panel_agg, hospital_info, on='hospital_id')

# Set multi-index for PanelOLS
did_panel_full = did_panel_full.set_index(['hospital_id', 'year'])

# 2. Event Study and Parallel Trends Test
# Create relative time to treatment
did_panel_full['event_time'] = did_panel_full.index.get_level_values('year') - did_panel_full['asp_adoption_year']

# Bin distant periods to avoid thin data and multicollinearity
min_event_time = did_panel_full[did_panel_full.asp_adoption_year != 9999].event_time.min()
max_event_time = did_panel_full[did_panel_full.asp_adoption_year != 9999].event_time.max()

did_panel_full['event_time'] = did_panel_full['event_time'].apply(lambda x: min_event_time if x <= min_event_time else x)
did_panel_full['event_time'] = did_panel_full['event_time'].apply(lambda x: max_event_time if x >= max_event_time else x)

# Create event time dummies, omitting t=-1 (the reference period)
event_dummies = pd.get_dummies(did_panel_full['event_time'], prefix='t', drop_first=False)
if 't_-1.0' in event_dummies.columns:
    event_dummies = event_dummies.drop(columns=['t_-1.0'])
did_panel_dummies = pd.concat([did_panel_full, event_dummies], axis=1)

# Define formula for event study
dummy_cols = sorted([col for col in event_dummies.columns if col != 't_-1.0'])
formula_event_study = 'antibiotic_duration ~ 1 + ' + ' + '.join(dummy_cols) + ' + EntityEffects + TimeEffects'

# Run event study model
mod_event = PanelOLS.from_formula(formula_event_study, data=did_panel_dummies)
res_event = mod_event.fit(cov_type='clustered', cluster_entity=True)

# Extract pre-trend coefficient for t-2
pre_trend_coeff_col = 't_-2.0'
if pre_trend_coeff_col in res_event.params:
    pre_trend_coeff = res_event.params[pre_trend_coeff_col]
    pre_trend_pvalue = res_event.pvalues[pre_trend_coeff_col]
    pre_trend_ci = res_event.conf_int().loc[pre_trend_coeff_col].tolist()
else: 
    pre_trend_coeff = np.nan
    pre_trend_pvalue = np.nan
    pre_trend_ci = [np.nan, np.nan]

# Create event study plot data
event_study_results = []
for col in dummy_cols:
    event_time_val = float(col.split('_')[1])
    coef = res_event.params[col]
    ci = res_event.conf_int().loc[col]
    event_study_results.append({'event_time': event_time_val, 'coef': coef, 'ci_low': ci[0], 'ci_high': ci[1]})

# Add the reference period t=-1
event_study_results.append({'event_time': -1.0, 'coef': 0, 'ci_low': 0, 'ci_high': 0})
event_study = pd.DataFrame(event_study_results).sort_values('event_time').reset_index(drop=True)

# 3. Standard TWFE model
did_panel_full['asp_active'] = (did_panel_full.index.get_level_values('year') >= did_panel_full['asp_adoption_year']).astype(int)
mod_twfe = PanelOLS.from_formula('antibiotic_duration ~ 1 + asp_active + EntityEffects + TimeEffects', data=did_panel_full)
res_twfe = mod_twfe.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = res_twfe.params['asp_active']

# 4. Estimator robust to staggered adoption (Cohort DiD vs. never-treated)
did_panel_reset = did_panel_full.reset_index()
cohorts = sorted(did_panel_reset[did_panel_reset['asp_adoption_year'] != 9999]['asp_adoption_year'].unique())
control_hospitals = did_panel_reset[did_panel_reset['asp_adoption_year'] == 9999]['hospital_id'].unique()

cohort_atts = []
cohort_weights = []

for g in cohorts:
    treated_hospitals = did_panel_reset[did_panel_reset['asp_adoption_year'] == g]['hospital_id'].unique()
    
    if len(treated_hospitals) == 0 or len(control_hospitals) == 0:
        continue
        
    subset_hospitals = np.concatenate([treated_hospitals, control_hospitals])
    df_cohort = did_panel_reset[did_panel_reset['hospital_id'].isin(subset_hospitals)].copy()
    
    df_cohort['treat'] = df_cohort['hospital_id'].isin(treated_hospitals).astype(int)
    df_cohort['post'] = (df_cohort['year'] >= g).astype(int)
    
    model_cohort = smf.ols('antibiotic_duration ~ treat * post', data=df_cohort).fit(cov_type='cluster', cov_kwds={'groups': df_cohort['hospital_id']})
    
    att_g = model_cohort.params['treat:post']
    cohort_atts.append(att_g)
    cohort_weights.append(len(treated_hospitals))

if cohort_atts:
    did_att_estimate = np.average(cohort_atts, weights=cohort_weights)
else:
    did_att_estimate = np.nan

did_panel = did_panel_full.reset_index()[['hospital_id', 'year', 'antibiotic_duration']]


# === OBJECTIVE 3: TREATMENT HETEROGENEITY (CATE) ===

features = ['age', 'comorbidity_score', 'immune_compromised', 'severity_score', 'sex', 
            'infection_type', 'hospital_size', 'region', 'ward', 'year', 'hospital_id']
X = pd.get_dummies(df[features], columns=['sex', 'infection_type', 'hospital_size', 'region', 'ward', 'hospital_id'], drop_first=True)
y = df['antibiotic_duration']
T = df['asp_active']

X_treat = X[T == 1]
y_treat = y[T == 1]
X_control = X[T == 0]
y_control = y[T == 0]

m1 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1)
m0 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1)

m1.fit(X_treat, y_treat)
m0.fit(X_control, y_control)

y1_hat = m1.predict(X)
y0_hat = m0.predict(X)

cate_estimates = y1_hat - y0_hat

n_bootstraps = 200
bootstrap_diffs = []
elderly_idx = df.index[df['age_group'] == '65+']
young_idx = df.index[df['age_group'] == '<40']

for _ in range(n_bootstraps):
    sample_indices = np.random.choice(df.index, size=len(df), replace=True)
    df_sample = df.iloc[sample_indices]
    
    X_boot = pd.get_dummies(df_sample[features], columns=['sex', 'infection_type', 'hospital_size', 'region', 'ward', 'hospital_id'], drop_first=True)
    y_boot = df_sample['antibiotic_duration']
    T_boot = df_sample['asp_active']
    
    X_boot_aligned = X_boot.reindex(columns=X.columns, fill_value=0)
    
    X_treat_boot = X_boot_aligned[T_boot == 1]
    y_treat_boot = y_boot[T_boot == 1]
    X_control_boot = X_boot_aligned[T_boot == 0]
    y_control_boot = y_boot[T_boot == 0]

    if len(X_treat_boot) > 5 and len(X_control_boot) > 5:
        m1_boot = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=42, n_jobs=-1).fit(X_treat_boot, y_treat_boot)
        m0_boot = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=42, n_jobs=-1).fit(X_control_boot, y_control_boot)
        
        cate_boot = m1_boot.predict(X_boot_aligned) - m0_boot.predict(X_boot_aligned)
        
        original_indices = df_sample.index
        is_elderly_boot = np.isin(original_indices, elderly_idx)
        is_young_boot = np.isin(original_indices, young_idx)
        
        if np.sum(is_elderly_boot) > 0 and np.sum(is_young_boot) > 0:
            mean_cate_elderly = np.mean(cate_boot[is_elderly_boot])
            mean_cate_young = np.mean(cate_boot[is_young_boot])
            bootstrap_diffs.append(mean_cate_elderly - mean_cate_young)

cate_elderly_minus_young_ci = np.percentile(bootstrap_diffs, [2.5, 97.5]).tolist()


# === OBJECTIVE 4: HIERARCHICAL MODELING ===

mixed_formula = 'antibiotic_duration ~ asp_active + age + comorbidity_score + immune_compromised + severity_score + C(sex) + C(infection_type)'
mixed_model = smf.mixedlm(formula=mixed_formula, 
                          data=df, 
                          groups=df['hospital_id'])
mixed_model_fit = mixed_model.fit(reml=False)

mixed_model_aic = mixed_model_fit.aic
mixed_asp_coeff = mixed_model_fit.params['asp_active']
random_effect_variance = float(mixed_model_fit.cov_re)
residual_variance = mixed_model_fit.scale
icc_value = random_effect_variance / (random_effect_variance + residual_variance)

ols_model = smf.ols(formula=mixed_formula, data=df)
ols_model_fit = ols_model.fit()
naive_ols_aic = ols_model_fit.aic