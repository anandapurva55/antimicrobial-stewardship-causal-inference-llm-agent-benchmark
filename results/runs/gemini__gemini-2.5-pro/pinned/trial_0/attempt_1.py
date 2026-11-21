import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from lifelines import KaplanMeierFitter, CoxPHFitter
from linearmodels.panel import PanelOLS
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm

# --- PREPARATION ---
# Ensure correct dtypes for categorical variables to be used in models
df['hospital_id'] = df['hospital_id'].astype('category')
df['region'] = df['region'].astype('category')
df['ward'] = df['ward'].astype('category')
df['sex'] = df['sex'].astype('category')
df['age_group'] = df['age_group'].astype('category')
df['infection_type'] = df['infection_type'].astype('category')
df['hospital_size'] = df['hospital_size'].astype('category')
# Keep 'year' as int for calculations, convert to category within models where needed

# --- OBJECTIVE 1: SURVIVAL ANALYSIS ---

# Prepare data for lifelines (needs dummy variables)
df_survival = df.copy()
covariates_survival = ['asp_active', 'age', 'sex', 'comorbidity_score', 'severity_score', 'immune_compromised', 'infection_type', 'hospital_size']
df_survival_dummies = pd.get_dummies(df_survival[covariates_survival + ['time_to_improvement', 'event_observed', 'hospital_id']], drop_first=True, dtype=int)

# Fit Cox Proportional Hazards Model
cph = CoxPHFitter()
cph.fit(df_survival_dummies, 
        duration_col='time_to_improvement', 
        event_col='event_observed', 
        cluster_col='hospital_id',
        robust=True)

cox_summary = cph.summary
cox_hr_asp = float(cox_summary.loc['asp_active', 'exp(coef)'])
cox_se_asp = float(cox_summary.loc['asp_active', 'se(coef)'])

# Test Proportional Hazards Assumption
ph_test_results = cph.check_assumptions(df_survival_dummies, show_plots=False, p_value_threshold=0.05)
ph_test_pvalues = {col.replace("'", ""): p_val for col, p_val in zip(ph_test_results.index, ph_test_results.p) if col not in ['log(time)', 'identity']}


# --- OBJECTIVE 2: CAUSAL IMPACT (DiD) ---

# 1. Build hospital-year panel
did_panel = df.groupby(['hospital_id', 'year'])['antibiotic_duration'].mean().reset_index()
hospital_info = df[['hospital_id', 'asp_adoption_year']].drop_duplicates()
did_panel = pd.merge(did_panel, hospital_info, on='hospital_id')
did_panel['year'] = did_panel['year'].astype(int)
did_panel['asp_adoption_year'] = did_panel['asp_adoption_year'].astype(int)
did_panel['asp_active'] = (did_panel['year'] >= did_panel['asp_adoption_year']).astype(int)

# Set up panel data for linearmodels
did_panel = did_panel.set_index(['hospital_id', 'year'])

# 2. Run TWFE model
mod_twfe = PanelOLS.from_formula('antibiotic_duration ~ asp_active + EntityEffects + TimeEffects', data=did_panel)
res_twfe = mod_twfe.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = float(res_twfe.params['asp_active'])

# 3. Event Study and Parallel Trends Test
panel_es = did_panel.copy().reset_index()
panel_es['event_time'] = np.where(panel_es['asp_adoption_year'] != 9999, 
                                  panel_es['year'] - panel_es['asp_adoption_year'], 
                                  -999) # Placeholder for never-treated

# Binning event times for stability
panel_es['event_time'] = panel_es['event_time'].clip(lower=-3, upper=3) 

# Create dummies, drop reference period t=-1, and also drop the never-treated placeholder
event_dummies = pd.get_dummies(panel_es['event_time'], prefix='t', drop_first=False).astype(int)
if 't_-1' in event_dummies.columns:
    event_dummies = event_dummies.drop(columns=['t_-1'])
if 't_-999' in event_dummies.columns:
    event_dummies = event_dummies.drop(columns=['t_-999'])

panel_es = pd.concat([panel_es, event_dummies], axis=1)
panel_es = panel_es.set_index(['hospital_id', 'year'])

formula_es = 'antibiotic_duration ~ 1 + ' + ' + '.join(event_dummies.columns) + ' + EntityEffects + TimeEffects'
mod_es = PanelOLS.from_formula(formula_es, data=panel_es)
res_es = mod_es.fit(cov_type='clustered', cluster_entity=True)

# Extract pre-trend test results for t=-2
pre_trend_coeff = float(res_es.params['t_-2'])
pre_trend_pvalue = float(res_es.pvalues['t_-2'])
pre_trend_ci = res_es.conf_int().loc['t_-2'].tolist()

# Create event_study DataFrame
event_study_results = []
for col in event_dummies.columns:
    event_time = int(col.split('_')[1])
    coef = res_es.params[col]
    ci = res_es.conf_int().loc[col]
    event_study_results.append({'event_time': event_time, 'coef': coef, 'ci_low': ci['lower'], 'ci_high': ci['upper']})

event_study = pd.DataFrame(event_study_results).sort_values('event_time').reset_index(drop=True)

# 4. Robust DiD Estimator (Stacked Regression)
panel_robust = did_panel.copy().reset_index()
adoption_years = sorted([y for y in panel_robust['asp_adoption_year'].unique() if y != 9999])
never_treated_hospitals = panel_robust[panel_robust['asp_adoption_year'] == 9999]['hospital_id'].unique()

cohort_atts = []
cohort_weights = []

for g in adoption_years:
    treated_hospitals = panel_robust[panel_robust['asp_adoption_year'] == g]['hospital_id'].unique()
    
    subset_df = panel_robust[panel_robust['hospital_id'].isin(np.concatenate([treated_hospitals, never_treated_hospitals]))].copy()
    subset_df['treated'] = subset_df['hospital_id'].isin(treated_hospitals).astype(int)
    subset_df['post'] = (subset_df['year'] >= g).astype(int)
    subset_df['did'] = subset_df['treated'] * subset_df['post']
    
    subset_df = subset_df.set_index(['hospital_id', 'year'])
    
    mod_cohort = PanelOLS.from_formula('antibiotic_duration ~ did + EntityEffects + TimeEffects', data=subset_df)
    res_cohort = mod_cohort.fit(cov_type='clustered', cluster_entity=True)
    
    cohort_atts.append(res_cohort.params['did'])
    cohort_weights.append(len(treated_hospitals))

did_att_estimate = float(np.average(cohort_atts, weights=cohort_weights))


# --- OBJECTIVE 3: TREATMENT HETEROGENEITY (CATE) ---

# Prepare data for T-Learner
df_cate = df.copy()
df_cate['year'] = df_cate['year'].astype(str) # Treat year as categorical for one-hot
features = ['age', 'sex', 'comorbidity_score', 'immune_compromised', 'severity_score', 'infection_type', 'hospital_size', 'region', 'ward', 'hospital_id', 'year']
X = pd.get_dummies(df_cate[features], drop_first=True, dtype=int)
Y = df_cate['antibiotic_duration']
T = df_cate['asp_active']

# Split data into treated and control
X_treat, Y_treat = X[T == 1], Y[T == 1]
X_control, Y_control = X[T == 0], Y[T == 0]

# T-Learner with RandomForest
m1 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1)
m0 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1)

m1.fit(X_treat, Y_treat)
m0.fit(X_control, Y_control)

cate_estimates = m1.predict(X) - m0.predict(X)

# Bootstrap CI for CATE difference (elderly vs. young)
n_bootstraps = 200
bootstrap_diffs = []
age_elderly_mask = df_cate['age'] >= 65
age_young_mask = df_cate['age'] < 40
indices = np.arange(len(df_cate))

for i in range(n_bootstraps):
    boot_indices = np.random.choice(indices, size=len(indices), replace=True)
    
    df_boot = df_cate.iloc[boot_indices]
    X_boot = X.iloc[boot_indices]
    Y_boot = Y.iloc[boot_indices]
    T_boot = T.iloc[boot_indices]
    
    X_treat_boot, Y_treat_boot = X_boot[T_boot == 1], Y_boot[T_boot == 1]
    X_control_boot, Y_control_boot = X_boot[T_boot == 0], Y_boot[T_boot == 0]
    
    if len(X_treat_boot) > 5 and len(X_control_boot) > 5:
        m1_boot = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=42+i, n_jobs=-1).fit(X_treat_boot, Y_treat_boot)
        m0_boot = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=42+i, n_jobs=-1).fit(X_control_boot, Y_control_boot)
        
        cate_boot = m1_boot.predict(X_boot) - m0_boot.predict(X_boot)
        
        age_elderly_boot_mask = age_elderly_mask.iloc[boot_indices]
        age_young_boot_mask = age_young_mask.iloc[boot_indices]
        
        if np.sum(age_elderly_boot_mask) > 0 and np.sum(age_young_boot_mask) > 0:
            mean_cate_elderly = np.mean(cate_boot[age_elderly_boot_mask])
            mean_cate_young = np.mean(cate_boot[age_young_boot_mask])
            bootstrap_diffs.append(mean_cate_elderly - mean_cate_young)

bootstrap_diffs = np.array(bootstrap_diffs)
cate_elderly_minus_young_ci = [np.percentile(bootstrap_diffs, 2.5), np.percentile(bootstrap_diffs, 97.5)]


# --- OBJECTIVE 4: HIERARCHICAL MODELING ---
df_mixed = df.copy()
df_mixed['year'] = df_mixed['year'].astype('category') 

# Fit mixed-effects model
formula_mixed = "antibiotic_duration ~ asp_active + age + sex + comorbidity_score + severity_score + C(year)"
mixed_model = smf.mixedlm(formula_mixed, df_mixed, groups=df_mixed["hospital_id"]).fit(reml=False)

# Extract results from mixed model
mixed_model_aic = float(mixed_model.aic)
mixed_asp_coeff = float(mixed_model.params['asp_active'])
random_effect_variance = float(mixed_model.cov_re.iloc[0,0])
residual_variance = float(mixed_model.scale)
icc_value = random_effect_variance / (random_effect_variance + residual_variance)

# Fit naive OLS model for comparison
ols_model = smf.ols(formula_mixed, data=df_mixed).fit()
naive_ols_aic = float(ols_model.aic)