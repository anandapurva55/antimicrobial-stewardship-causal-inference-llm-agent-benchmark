from lifelines.statistics import proportional_hazard_test
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# ==============================================================================
# 1. SURVIVAL ANALYSIS: Kaplan-Meier & Clustered Cox PH + Schoenfeld PH Test
# ==============================================================================
kmf_asp = KaplanMeierFitter()
kmf_no_asp = KaplanMeierFitter()
mask_asp = df['asp_active'] == 1
kmf_asp.fit(
    df.loc[mask_asp, 'time_to_improvement'],
    event_observed=df.loc[mask_asp, 'event_observed'],
    label='ASP Active',
)
kmf_no_asp.fit(
    df.loc[~mask_asp, 'time_to_improvement'],
    event_observed=df.loc[~mask_asp, 'event_observed'],
    label='No ASP',
)

cox_num_cols = [
    'asp_active',
    'age',
    'comorbidity_score',
    'immune_compromised',
    'severity_score',
]
cox_cat_cols = ['sex', 'infection_type', 'ward', 'hospital_size', 'region', 'year']
df_cox_dummies = pd.get_dummies(
    df[cox_cat_cols].astype({c: 'str' for c in cox_cat_cols}),
    drop_first=True,
    dtype=float,
)
df_cox = pd.concat(
    [
        df[['time_to_improvement', 'event_observed', 'hospital_id'] + cox_num_cols].astype(float),
        df_cox_dummies,
    ],
    axis=1,
)

cph = CoxPHFitter()
cph.fit(
    df_cox,
    duration_col='time_to_improvement',
    event_col='event_observed',
    cluster_col='hospital_id',
    robust=True,
)

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.standard_errors_['asp_active'])

ph_test_res = proportional_hazard_test(cph, df_cox, time_transform='rank')
ph_test_pvalues = {str(k): float(v) for k, v in ph_test_res.summary['p'].items()}

# ==============================================================================
# 2. CAUSAL IMPACT (DiD): Hospital-Year Panel, TWFE, Sun-Abraham Event Study
# ==============================================================================
did_panel = (
    df.groupby(['hospital_id', 'year'], as_index=False)
    .agg(
        antibiotic_duration=('antibiotic_duration', 'mean'),
        asp_adoption_year=('asp_adoption_year', 'first'),
        asp_active=('asp_active', 'mean'),
    )
    .sort_values(['hospital_id', 'year'])
    .reset_index(drop=True)
)

panel_indexed = did_panel.set_index(['hospital_id', 'year'])
twfe_mod = PanelOLS.from_formula(
    'antibiotic_duration ~ asp_active + EntityEffects + TimeEffects',
    data=panel_indexed,
)
twfe_res = twfe_mod.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = float(twfe_res.params['asp_active'])

# Sun-Abraham (2021) interaction-weighted estimator using never-treated controls
cohorts = sorted([int(g) for g in did_panel['asp_adoption_year'].unique() if g < 9000])
cohort_sizes = did_panel.groupby('asp_adoption_year')['hospital_id'].nunique().to_dict()

X_es = pd.get_dummies(
    did_panel[['hospital_id', 'year']].astype(str),
    drop_first=True,
    dtype=float,
)
X_es.insert(0, 'const', 1.0)

sa_meta = []
for g in cohorts:
    rel_times = sorted((did_panel.loc[did_panel['asp_adoption_year'] == g, 'year'] - g).unique())
    for e in rel_times:
        if int(e) == -1:
            continue
        col_name = f'sa_{g}_{int(e)}'
        X_es[col_name] = (
            (did_panel['asp_adoption_year'] == g) & ((did_panel['year'] - g) == e)
        ).astype(float)
        sa_meta.append((col_name, g, int(e), int(cohort_sizes[g])))

sa_ols = sm.OLS(did_panel['antibiotic_duration'].astype(float), X_es).fit(
    cov_type='cluster',
    cov_kwds={'groups': did_panel['hospital_id']},
    use_t=True,
)

unique_event_times = sorted(set(item[2] for item in sa_meta))
es_rows = []
es_pvals = {}

for e in unique_event_times:
    items_e = [(col, g, n_g) for (col, g, e_i, n_g) in sa_meta if e_i == e]
    tot_n = sum(n_g for _, _, n_g in items_e)
    r_vec = np.zeros(len(X_es.columns), dtype=float)
    for col, _, n_g in items_e:
        r_vec[X_es.columns.get_loc(col)] = n_g / tot_n
    ttest = sa_ols.t_test(r_vec)
    coef_e = float(np.asarray(ttest.effect).item())
    ci_arr = np.asarray(ttest.conf_int(alpha=0.05)).ravel()
    pval_e = float(np.asarray(ttest.pvalue).item())
    es_rows.append(
        {
            'event_time': int(e),
            'coef': coef_e,
            'ci_low': float(ci_arr[0]),
            'ci_high': float(ci_arr[1]),
        }
    )
    es_pvals[int(e)] = pval_e

event_study = pd.DataFrame(es_rows).sort_values('event_time').reset_index(drop=True)

# Pre-trend test using the immediate pre-treatment lead (event_time = -2)
lead_e = -2 if -2 in es_pvals else max([e for e in unique_event_times if e < -1])
lead_row = event_study.loc[event_study['event_time'] == lead_e].iloc[0]
pre_trend_coeff = float(lead_row['coef'])
pre_trend_pvalue = float(es_pvals[lead_e])
pre_trend_ci = [float(lead_row['ci_low']), float(lead_row['ci_high'])]

# Overall staggered ATT across all post-treatment periods (e >= 0)
post_items = [(col, g, n_g) for (col, g, e_i, n_g) in sa_meta if e_i >= 0]
tot_post_n = sum(n_g for _, _, n_g in post_items)
r_att = np.zeros(len(X_es.columns), dtype=float)
for col, _, n_g in post_items:
    r_att[X_es.columns.get_loc(col)] = n_g / tot_post_n
att_ttest = sa_ols.t_test(r_att)
did_att_estimate = float(np.asarray(att_ttest.effect).item())

# ==============================================================================
# 3. TREATMENT HETEROGENEITY: Cross-Fitted X-Learner + Bootstrap CI
# ==============================================================================
adj_formula = (
    'antibiotic_duration ~ asp_active + C(hospital_id) + C(year) + '
    'age + C(sex) + comorbidity_score + immune_compromised + '
    'C(infection_type) + severity_score + C(ward)'
)
adj_ols = smf.ols(adj_formula, data=df).fit()
hy_idx = [
    i
    for i, name in enumerate(adj_ols.model.exog_names)
    if name.startswith('C(hospital_id)') or name.startswith('C(year)')
]
hy_effect = adj_ols.model.exog[:, hy_idx] @ adj_ols.params.iloc[hy_idx].values
y_adj = df['antibiotic_duration'].to_numpy(dtype=float) - hy_effect

cate_num_cols = ['age', 'comorbidity_score', 'immune_compromised', 'severity_score']
cate_cat_cols = ['sex', 'infection_type', 'ward', 'hospital_size', 'region']
X_cate = pd.concat(
    [
        df[cate_num_cols].astype(float),
        pd.get_dummies(df[cate_cat_cols].astype(str), drop_first=True, dtype=float),
    ],
    axis=1,
).to_numpy(dtype=float)
W_cate = df['asp_active'].to_numpy(dtype=int)

def _gbr(seed=42):
    return HistGradientBoostingRegressor(
        max_iter=80,
        max_depth=4,
        learning_rate=0.08,
        min_samples_leaf=20,
        random_state=seed,
    )

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mu0_oof = np.zeros(len(df), dtype=float)
mu1_oof = np.zeros(len(df), dtype=float)
tau0_oof = np.zeros(len(df), dtype=float)
tau1_oof = np.zeros(len(df), dtype=float)
prop_oof = np.zeros(len(df), dtype=float)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_cate, W_cate)):
    X_tr, W_tr, y_tr = X_cate[tr_idx], W_cate[tr_idx], y_adj[tr_idx]
    X_va = X_cate[va_idx]

    m0 = _gbr(42 + fold).fit(X_tr[W_tr == 0], y_tr[W_tr == 0])
    m1 = _gbr(142 + fold).fit(X_tr[W_tr == 1], y_tr[W_tr == 1])
    mu0_oof[va_idx] = m0.predict(X_va)
    mu1_oof[va_idx] = m1.predict(X_va)

    ps_mod = LogisticRegression(max_iter=1000, random_state=42 + fold)
    ps_mod.fit(X_tr, W_tr)
    prop_oof[va_idx] = np.clip(ps_mod.predict_proba(X_va)[:, 1], 0.05, 0.95)

D1 = y_adj[W_cate == 1] - mu0_oof[W_cate == 1]
D0 = mu1_oof[W_cate == 0] - y_adj[W_cate == 0]

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_cate, W_cate)):
    tr_mask_1 = W_cate[tr_idx] == 1
    tr_mask_0 = W_cate[tr_idx] == 0
    X_tr = X_cate[tr_idx]
    D1_tr = y_adj[tr_idx][tr_mask_1] - mu0_oof[tr_idx][tr_mask_1]
    D0_tr = mu1_oof[tr_idx][tr_mask_0] - y_adj[tr_idx][tr_mask_0]

    t1 = _gbr(242 + fold).fit(X_tr[tr_mask_1], D1_tr)
    t0 = _gbr(342 + fold).fit(X_tr[tr_mask_0], D0_tr)
    tau1_oof[va_idx] = t1.predict(X_cate[va_idx])
    tau0_oof[va_idx] = t0.predict(X_cate[va_idx])

cate_estimates = prop_oof * tau0_oof + (1.0 - prop_oof) * tau1_oof

# Bootstrap 95% CI for mean CATE(age >= 65) - mean CATE(age < 40)
ages = df['age'].to_numpy(dtype=float)
rng = np.random.default_rng(42)
n_obs = len(df)
n_boot = 100
boot_diffs = np.empty(n_boot, dtype=float)

for b in range(n_boot):
    b_idx = rng.integers(0, n_obs, size=n_obs)
    X_b, W_b, y_b, age_b = X_cate[b_idx], W_cate[b_idx], y_adj[b_idx], ages[b_idx]

    m0_b = _gbr(1000 + b).fit(X_b[W_b == 0], y_b[W_b == 0])
    m1_b = _gbr(2000 + b).fit(X_b[W_b == 1], y_b[W_b == 1])

    d1_b = y_b[W_b == 1] - m0_b.predict(X_b[W_b == 1])
    d0_b = m1_b.predict(X_b[W_b == 0]) - y_b[W_b == 0]

    t1_b = _gbr(3000 + b).fit(X_b[W_b == 1], d1_b)
    t0_b = _gbr(4000 + b).fit(X_b[W_b == 0], d0_b)

    ps_b = LogisticRegression(max_iter=1000, random_state=5000 + b).fit(X_b, W_b)
    e_b = np.clip(ps_b.predict_proba(X_b)[:, 1], 0.05, 0.95)
    cate_b = e_b * t0_b.predict(X_b) + (1.0 - e_b) * t1_b.predict(X_b)

    boot_diffs[b] = cate_b[age_b >= 65].mean() - cate_b[age_b < 40].mean()

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5)),
]

# ==============================================================================
# 4. HIERARCHICAL MODELING: Mixed-Effects Random Intercept & AIC Comparison
# ==============================================================================
mixed_formula = (
    'antibiotic_duration ~ asp_active + C(year) + '
    'age + C(sex) + comorbidity_score + immune_compromised + '
    'C(infection_type) + severity_score + C(ward)'
)
mixed_mod = smf.mixedlm(mixed_formula, data=df, groups=df['hospital_id'])
mixed_res = mixed_mod.fit(reml=False, method='lbfgs')
if not mixed_res.converged:
    mixed_res = mixed_mod.fit(reml=False, method='bfgs')

random_effect_variance = float(mixed_res.cov_re.iloc[0, 0])
residual_variance = float(mixed_res.scale)
icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))
mixed_model_aic = float(mixed_res.aic)
mixed_asp_coeff = float(mixed_res.params['asp_active'])

naive_ols_res = smf.ols(mixed_formula, data=df).fit()
naive_ols_aic = float(naive_ols_res.aic)