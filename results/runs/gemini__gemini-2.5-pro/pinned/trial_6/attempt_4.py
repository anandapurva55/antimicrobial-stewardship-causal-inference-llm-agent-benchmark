import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from linearmodels.panel import PanelOLS
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import norm

# --- Objective 1: Survival Analysis ---

# Prepare data for lifelines, creating dummy variables for categorical covariates
df_cox = df.copy()
covariates_survival = [
    'asp_active', 'age', 'comorbidity_score', 'severity_score',
    'immune_compromised', 'sex', 'infection_type', 'hospital_size', 'region'
]
df_cox_dummies = pd.get_dummies(
    df_cox[covariates_survival + ['time_to_improvement', 'event_observed', 'hospital_id']],
    columns=['sex', 'infection_type', 'hospital_size', 'region'],
    drop_first=True
)

# Fit Cox Proportional Hazards model with clustered standard errors
cph = CoxPHFitter()
cph.fit(
    df_cox_dummies,
    duration_col='time_to_improvement',
    event_col='event_observed',
    cluster_col='hospital_id',
    formula=" + ".join([col for col in df_cox_dummies.columns if col not in ['time_to_improvement', 'event_observed', 'hospital_id']])
)

cox_hr_asp = cph.hazard_ratios_['asp_active']
cox_se_asp = cph.summary.loc['asp_active', 'se(coef)']

# Test Proportional Hazards assumption using Schoenfeld residuals
# This test doesn't support clustering, so we fit a standard model for the test
cph_test_model = CoxPHFitter()
df_cox_test = df_cox_dummies.drop(columns=['hospital_id'])
cph_test_model.fit(
    df_cox_test,
    duration_col='time_to_improvement',
    event_col='event_observed',
    formula=" + ".join([col for col in df_cox_test.columns if col not in ['time_to_improvement', 'event_observed']])
)
ph_test_results = proportional_hazard_test(cph_test_model, df_cox_test, time_transform='log')
ph_test_pvalues = {
    var: ph_test_results.summary.loc[var, 'p']
    for var in ph_test_results.summary.index
}


# --- Objective 2: Causal Impact (DiD) ---

# Create the required hospital-year panel DataFrame
did_panel = df.groupby(['hospital_id', 'year'])['antibiotic_duration'].mean().reset_index()

# Prepare a panel for analysis with time-invariant hospital characteristics
panel_for_analysis = did_panel.copy()
hospital_chars = df[['hospital_id', 'asp_adoption_year']].drop_duplicates()
panel_for_analysis = pd.merge(panel_for_analysis, hospital_chars, on='hospital_id')
panel_for_analysis = panel_for_analysis.set_index(['hospital_id', 'year'])

# --- Standard TWFE Model ---
panel_for_analysis['treat'] = (panel_for_analysis.index.get_level_values('year') >= panel_for_analysis['asp_adoption_year']).astype(int)
twfe_model = PanelOLS.from_formula('antibiotic_duration ~ treat + EntityEffects + TimeEffects', data=panel_for_analysis)
twfe_results = twfe_model.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = twfe_results.params['treat']

# --- Event Study and Parallel Trends Test ---
panel_for_analysis['event_time'] = panel_for_analysis.index.get_level_values('year') - panel_for_analysis['asp_adoption_year']
panel_for_analysis['event_time'] = panel_for_analysis['event_time'].clip(lower=-3, upper=3)
panel_event_study_subset = panel_for_analysis[panel_for_analysis['asp_adoption_year'] != 9999].copy()

event_dummies = pd.get_dummies(panel_event_study_subset['event_time'], prefix='t', drop_first=False)
if 't_-1.0' in event_dummies.columns:
    event_dummies = event_dummies.drop(columns=['t_-1.0'])

panel_event_study_final = pd.concat([panel_event_study_subset, event_dummies], axis=1)

all_dummy_cols = [col for col in event_dummies.columns if col.startswith('t_')]
event_times_numeric = [float(c.replace('t_', '')) for c in all_dummy_cols]
min_event_time = min(event_times_numeric) if event_times_numeric else np.nan
dummy_to_drop_at_endpoint = f't_{min_event_time}'

dummy_cols_for_formula = [col for col in all_dummy_cols if col != dummy_to_drop_at_endpoint]
sanitized_dummy_cols = [f'`{col}`' for col in dummy_cols_for_formula]

if not sanitized_dummy_cols:
    formula_event = 'antibiotic_duration ~ 1 + EntityEffects + TimeEffects'
else:
    formula_event = f'antibiotic_duration ~ {" + ".join(sanitized_dummy_cols)} + EntityEffects + TimeEffects'

event_model = PanelOLS.from_formula(formula_event, data=panel_event_study_final)
event_results = event_model.fit(cov_type='clustered', cluster_entity=True)

pre_trend_coeff_col = 't_-2.0'
pre_trend_coeff = event_results.params.get(pre_trend_coeff_col, np.nan)
pre_trend_pvalue = event_results.pvalues.get(pre_trend_coeff_col, np.nan)
pre_trend_ci = event_results.conf_int().loc[pre_trend_coeff_col].tolist() if pre_trend_coeff_col in event_results.conf_int().index else [np.nan, np.nan]

event_study_list = []
for col in dummy_cols_for_formula:
    time = float(col.replace('t_', ''))
    coef = event_results.params[col]
    ci = event_results.conf_int().loc[col]
    event_study_list.append({'event_time': time, 'coef': coef, 'ci_low': ci['lower'], 'ci_high': ci['upper']})
event_study_list.append({'event_time': -1.0, 'coef': 0.0, 'ci_low': 0.0, 'ci_high': 0.0})
if dummy_to_drop_at_endpoint in all_dummy_cols:
    event_study_list.append({'event_time': min_event_time, 'coef': 0.0, 'ci_low': 0.0, 'ci_high': 0.0})
event_study = pd.DataFrame(event_study_list).sort_values('event_time').reset_index(drop=True)

# --- Estimator Robust to Staggered Adoption (Callaway-Sant'Anna style) ---
cohort_years = sorted(df[df['asp_adoption_year'] != 9999]['asp_adoption_year'].unique())
cohort_sizes = df[df['asp_adoption_year'].isin(cohort_years)].groupby('asp_adoption_year')['hospital_id'].nunique()
total_treated_hospitals = cohort_sizes.sum()
att_estimates = {}

for cohort in cohort_years:
    cohort_hospitals = cohort_sizes.index[cohort_sizes.index == cohort].tolist()
    never_treated_hospitals = df[df['asp_adoption_year'] == 9999]['hospital_id'].unique().tolist()
    hospitals_in_comparison_ids = cohort_hospitals + never_treated_hospitals
    panel_subset = panel_for_analysis.loc[panel_for_analysis.index.get_level_values('hospital_id').isin(hospitals_in_comparison_ids)].copy()
    
    panel_subset['treat_cohort'] = ((panel_subset.index.get_level_values('year') >= cohort) &
                                     (panel_subset.index.get_level_values('hospital_id').isin(cohort_hospitals))).astype(int)
    
    model_cohort = PanelOLS.from_formula('antibiotic_duration ~ treat_cohort + EntityEffects + TimeEffects', data=panel_subset)
    results_cohort = model_cohort.fit(cov_type='clustered', cluster_entity=True)
    att_estimates[cohort] = results_cohort.params['treat_cohort']

did_att_estimate = sum(att_estimates[c] * (cohort_sizes[c] / total_treated_hospitals) for c in cohort_years if c in att_estimates)

# --- Objective 3: Treatment Heterogeneity (CATE) ---

# Prepare data for T-Learner
df_cate = df.copy()
features = ['age', 'comorbidity_score', 'severity_score', 'immune_compromised',
            'sex', 'infection_type', 'hospital_id', 'year']
X = pd.get_dummies(df_cate[features], columns=['sex', 'infection_type', 'hospital_id'], drop_first=True)
y = df_cate['antibiotic_duration']
T = df_cate['asp_active']

X_treat, y_treat = X[T == 1], y[T == 1]
X_control, y_control = X[T == 0], y[T == 0]

m1 = GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=4)
m0 = GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=4)

m1.fit(X_treat, y_treat)
m0.fit(X_control, y_control)

cate_estimates = m1.predict(X) - m0.predict(X)

# Bootstrap CI for CATE difference (elderly vs. young)
n_bootstraps = 200
bootstrap_diffs = []
df_cate['cate'] = cate_estimates
cate_elderly = df_cate.loc[df_cate['age'] >= 65, 'cate']
cate_young = df_cate.loc[df_cate['age'] < 40, 'cate']

if not cate_elderly.empty and not cate_young.empty:
    for _ in range(n_bootstraps):
        sample_elderly_mean = np.random.choice(cate_elderly, size=len(cate_elderly), replace=True).mean()
        sample_young_mean = np.random.choice(cate_young, size=len(cate_young), replace=True).mean()
        bootstrap_diffs.append(sample_elderly_mean - sample_young_mean)
    cate_elderly_minus_young_ci = np.percentile(bootstrap_diffs, [2.5, 97.5]).tolist()
else:
    cate_elderly_minus_young_ci = [np.nan, np.nan]


# --- Objective 4: Hierarchical Modeling ---

# Fit mixed-effects model with hospital as a random intercept
mix_formula = "antibiotic_duration ~ asp_active + age + comorbidity_score + severity_score + immune_compromised + C(year) + C(infection_type)"
mixed_model = smf.mixedlm(mix_formula, df, groups=df["hospital_id"])
mixed_results = mixed_model.fit(reml=False)

mixed_model_aic = mixed_results.aic
mixed_asp_coeff = mixed_results.params['asp_active']

random_effect_variance = float(mixed_results.cov_re.iloc[0, 0])
residual_variance = mixed_results.scale
icc_value = random_effect_variance / (random_effect_variance + residual_variance)

# Fit naive OLS model with hospital fixed effects for comparison
ols_formula = mix_formula + " + C(hospital_id)"
ols_model = smf.ols(ols_formula, data=df)
ols_results = ols_model.fit()
naive_ols_aic = ols_results.aic