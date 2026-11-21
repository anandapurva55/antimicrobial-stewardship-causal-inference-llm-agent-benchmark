from lifelines.statistics import proportional_hazard_test
from sklearn.ensemble import GradientBoostingRegressor

# =============================================================================
# 1. SURVIVAL ANALYSIS: Kaplan-Meier & Cox PH (clustered by hospital_id)
# =============================================================================
kmf_treated = KaplanMeierFitter()
kmf_control = KaplanMeierFitter()

mask_treated = df['asp_active'] == 1
kmf_treated.fit(
    durations=df.loc[mask_treated, 'time_to_improvement'],
    event_observed=df.loc[mask_treated, 'event_observed'],
    label='ASP Active',
)
kmf_control.fit(
    durations=df.loc[~mask_treated, 'time_to_improvement'],
    event_observed=df.loc[~mask_treated, 'event_observed'],
    label='No ASP',
)

surv_num_cols = [
    'asp_active',
    'age',
    'comorbidity_score',
    'immune_compromised',
    'severity_score',
]
surv_cat_cols = ['sex', 'ward', 'infection_type', 'hospital_size', 'region', 'year']
df_surv_covs = pd.concat(
    [
        df[surv_num_cols].astype(float),
        pd.get_dummies(df[surv_cat_cols].astype(str), drop_first=True, dtype=float),
    ],
    axis=1,
)
non_const_cols = [c for c in df_surv_covs.columns if df_surv_covs[c].std() > 0]
df_surv = pd.concat(
    [
        df[['time_to_improvement', 'event_observed', 'hospital_id']],
        df_surv_covs[non_const_cols],
    ],
    axis=1,
)

cph = CoxPHFitter()
cph.fit(
    df_surv,
    duration_col='time_to_improvement',
    event_col='event_observed',
    cluster_col='hospital_id',
    robust=True,
)

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.standard_errors_['asp_active'])

ph_test = proportional_hazard_test(cph, df_surv, time_transform='rank')
ph_test_pvalues = {str(k): float(v) for k, v in ph_test.summary['p'].items()}

# =============================================================================
# 2. CAUSAL IMPACT (DiD): Hospital-Year Panel, Pre-Trends, TWFE & Staggered DiD
# =============================================================================
did_panel = (
    df.groupby(['hospital_id', 'year'], as_index=False)
    .agg(
        antibiotic_duration=('antibiotic_duration', 'mean'),
        asp_active=('asp_active', 'mean'),
        asp_adoption_year=('asp_adoption_year', 'first'),
    )
    .sort_values(['hospital_id', 'year'])
    .reset_index(drop=True)
)
did_panel['asp_active'] = did_panel['asp_active'].round().astype(int)

# Plain TWFE on the hospital-year panel
panel_indexed = did_panel.set_index(['hospital_id', 'year'])
twfe_mod = PanelOLS.from_formula(
    'antibiotic_duration ~ asp_active + EntityEffects + TimeEffects',
    data=panel_indexed,
)
twfe_res = twfe_mod.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = float(twfe_res.params['asp_active'])

# Staggered adoption event study & ATT (Sun-Abraham IW / Callaway-Sant'Anna with never-treated controls)
treated_cohorts = sorted(
    [int(g) for g in did_panel['asp_adoption_year'].unique() if g < 9000]
)

cohort_event_cols = {}
all_inter_cols = []
for g in treated_cohorts:
    rel_times = sorted(
        (did_panel.loc[did_panel['asp_adoption_year'] == g, 'year'] - g).unique()
    )
    for e in rel_times:
        if e == -1:
            continue
        e_int = int(e)
        col_name = f"cohort_{g}_rel_{'m' + str(abs(e_int)) if e_int < 0 else str(e_int)}"
        did_panel[col_name] = (
            (did_panel['asp_adoption_year'] == g) & ((did_panel['year'] - g) == e_int)
        ).astype(float)
        cohort_event_cols[(g, e_int)] = col_name
        all_inter_cols.append(col_name)

sa_formula = (
    'antibiotic_duration ~ '
    + ' + '.join(all_inter_cols)
    + ' + C(hospital_id) + C(year)'
)
sa_res = smf.ols(sa_formula, data=did_panel).fit(
    cov_type='cluster', cov_kwds={'groups': did_panel['hospital_id']}
)

param_index = {name: idx for idx, name in enumerate(sa_res.params.index)}
cohort_n_hospitals = (
    did_panel[did_panel['asp_adoption_year'].isin(treated_cohorts)]
    .groupby('asp_adoption_year')['hospital_id']
    .nunique()
    .to_dict()
)

unique_event_times = sorted({e for (_, e) in cohort_event_cols.keys()})
event_rows = []
event_pvals = {}

for e in unique_event_times:
    relevant = [(g, col) for (g, rel), col in cohort_event_cols.items() if rel == e]
    total_w = sum(cohort_n_hospitals[g] for g, _ in relevant)
    contrast = np.zeros(len(sa_res.params), dtype=float)
    for g, col in relevant:
        contrast[param_index[col]] = cohort_n_hospitals[g] / total_w
    t_res = sa_res.t_test(contrast)
    coef_val = float(np.asarray(t_res.effect).item())
    ci_vals = np.asarray(t_res.conf_int()).ravel()
    pval_val = float(np.asarray(t_res.pvalue).item())
    event_rows.append(
        {
            'event_time': int(e),
            'coef': coef_val,
            'ci_low': float(ci_vals[0]),
            'ci_high': float(ci_vals[1]),
        }
    )
    event_pvals[int(e)] = pval_val

event_study = pd.DataFrame(event_rows).sort_values('event_time').reset_index(drop=True)

# Pre-trend test on the immediate pre-treatment lead (event_time == -2)
pre_lead_row = event_study.loc[event_study['event_time'] == -2].iloc[0]
pre_trend_coeff = float(pre_lead_row['coef'])
pre_trend_pvalue = float(event_pvals[-2])
pre_trend_ci = [float(pre_lead_row['ci_low']), float(pre_lead_row['ci_high'])]

# Overall staggered ATT across all post-treatment (e >= 0) cells
post_cells = [(g, col) for (g, e), col in cohort_event_cols.items() if e >= 0]
total_post_w = sum(cohort_n_hospitals[g] for g, _ in post_cells)
att_contrast = np.zeros(len(sa_res.params), dtype=float)
for g, col in post_cells:
    att_contrast[param_index[col]] = cohort_n_hospitals[g] / total_post_w
att_res = sa_res.t_test(att_contrast)
did_att_estimate = float(np.asarray(att_res.effect).item())

# Clean extra dummy columns from did_panel so it contains the panel summary columns
did_panel = did_panel[
    ['hospital_id', 'year', 'antibiotic_duration', 'asp_active', 'asp_adoption_year']
].copy()

# =============================================================================
# 3. TREATMENT HETEROGENEITY: T-Learner with Hospital & Year Adjustment
# =============================================================================
fe_formula = (
    'antibiotic_duration ~ asp_active + C(hospital_id) + C(year) + '
    'age + C(sex) + comorbidity_score + immune_compromised + severity_score + '
    'C(ward) + C(infection_type)'
)
fe_res = smf.ols(fe_formula, data=df).fit()
X_fe_design = pd.DataFrame(
    fe_res.model.exog, columns=fe_res.model.exog_names, index=df.index
)
fe_cols = [
    c
    for c in X_fe_design.columns
    if c.startswith('C(hospital_id)') or c.startswith('C(year)')
]
fe_offset = X_fe_design[fe_cols].values @ fe_res.params[fe_cols].values
y_adj = df['antibiotic_duration'].values - fe_offset

cate_num_cols = ['age', 'comorbidity_score', 'immune_compromised', 'severity_score']
cate_cat_cols = ['sex', 'ward', 'infection_type', 'hospital_size', 'region']
X_cate = pd.concat(
    [
        df[cate_num_cols].astype(float),
        pd.get_dummies(df[cate_cat_cols], drop_first=True, dtype=float),
    ],
    axis=1,
).values
W_treat = df['asp_active'].values

m0 = GradientBoostingRegressor(
    n_estimators=60, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=42
)
m1 = GradientBoostingRegressor(
    n_estimators=60, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=42
)
m0.fit(X_cate[W_treat == 0], y_adj[W_treat == 0])
m1.fit(X_cate[W_treat == 1], y_adj[W_treat == 1])
cate_estimates = np.asarray(m1.predict(X_cate) - m0.predict(X_cate), dtype=float)

# Bootstrap 95% CI for mean CATE(age >= 65) - mean CATE(age < 40)
rng = np.random.default_rng(42)
n_obs = len(df)
ages = df['age'].values
n_boot = 100
boot_diffs = np.empty(n_boot, dtype=float)

for b in range(n_boot):
    idx = rng.integers(0, n_obs, size=n_obs)
    X_b = X_cate[idx]
    y_b = y_adj[idx]
    W_b = W_treat[idx]
    age_b = ages[idx]

    m0_b = GradientBoostingRegressor(
        n_estimators=60, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=b
    )
    m1_b = GradientBoostingRegressor(
        n_estimators=60, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=b
    )
    m0_b.fit(X_b[W_b == 0], y_b[W_b == 0])
    m1_b.fit(X_b[W_b == 1], y_b[W_b == 1])
    tau_b = m1_b.predict(X_b) - m0_b.predict(X_b)
    boot_diffs[b] = tau_b[age_b >= 65].mean() - tau_b[age_b < 40].mean()

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5)),
]

# =============================================================================
# 4. HIERARCHICAL MODELING: Mixed-Effects Random Intercept vs Naive OLS
# =============================================================================
mixed_formula = (
    'antibiotic_duration ~ asp_active + C(year) + age + C(sex) + '
    'comorbidity_score + immune_compromised + severity_score + '
    'C(ward) + C(infection_type)'
)
mixed_mod = smf.mixedlm(mixed_formula, data=df, groups=df['hospital_id'])
mixed_res = mixed_mod.fit(reml=False, method=['lbfgs', 'bfgs'])

random_effect_variance = float(mixed_res.cov_re.iloc[0, 0])
residual_variance = float(mixed_res.scale)
icc_value = float(
    random_effect_variance / (random_effect_variance + residual_variance)
)
mixed_asp_coeff = float(mixed_res.fe_params['asp_active'])

mixed_model_aic = float(mixed_res.aic)
if np.isnan(mixed_model_aic):
    mixed_model_aic = float(-2.0 * mixed_res.llf + 2.0 * (len(mixed_res.fe_params) + 2))

ols_res = smf.ols(mixed_formula, data=df).fit()
naive_ols_aic = float(ols_res.aic)