import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from linearmodels.panel import PanelOLS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.ensemble import HistGradientBoostingRegressor

    def _make_regressor(seed):
        return HistGradientBoostingRegressor(
            max_iter=50, max_depth=4, min_samples_leaf=20, random_state=seed
        )
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor

    def _make_regressor(seed):
        return GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=seed,
        )


# 1. SURVIVAL ANALYSIS
kmf_treated = KaplanMeierFitter().fit(
    df.loc[df["asp_active"] == 1, "time_to_improvement"],
    event_observed=df.loc[df["asp_active"] == 1, "event_observed"],
    label="ASP Active",
)
kmf_control = KaplanMeierFitter().fit(
    df.loc[df["asp_active"] == 0, "time_to_improvement"],
    event_observed=df.loc[df["asp_active"] == 0, "event_observed"],
    label="Control",
)

_surv_num_cols = [
    "time_to_improvement",
    "event_observed",
    "hospital_id",
    "asp_active",
    "age",
    "comorbidity_score",
    "immune_compromised",
    "severity_score",
]
_surv_cat_cols = ["sex", "ward", "infection_type", "hospital_size", "region", "year"]
_surv_dummies = pd.get_dummies(
    df[_surv_cat_cols].astype(str), drop_first=True, dtype=float
)
_surv_df = pd.concat(
    [df[_surv_num_cols].reset_index(drop=True), _surv_dummies.reset_index(drop=True)],
    axis=1,
)

try:
    _cph = CoxPHFitter()
    _cph.fit(
        _surv_df,
        duration_col="time_to_improvement",
        event_col="event_observed",
        cluster_col="hospital_id",
        robust=True,
    )
except Exception:
    _cph = CoxPHFitter(penalizer=0.01)
    _cph.fit(
        _surv_df,
        duration_col="time_to_improvement",
        event_col="event_observed",
        cluster_col="hospital_id",
        robust=True,
    )

cox_hr_asp = float(_cph.hazard_ratios_["asp_active"])
cox_se_asp = float(_cph.standard_errors_["asp_active"])

_ph_res = proportional_hazard_test(_cph, _surv_df, time_transform="rank")
ph_test_pvalues = {str(k): float(v) for k, v in _ph_res.summary["p"].items()}


# 2. CAUSAL IMPACT (DiD)
did_panel = (
    df.groupby(["hospital_id", "year"], as_index=False)
    .agg(
        antibiotic_duration=("antibiotic_duration", "mean"),
        asp_active=("asp_active", "mean"),
        asp_adoption_year=("asp_adoption_year", "first"),
    )
    .sort_values(["hospital_id", "year"])
    .reset_index(drop=True)
)

_panel_idx = did_panel.set_index(["hospital_id", "year"])
_twfe_res = PanelOLS.from_formula(
    "antibiotic_duration ~ asp_active + EntityEffects + TimeEffects", data=_panel_idx
).fit(cov_type="clustered", cluster_entity=True)
twfe_att_estimate = float(_twfe_res.params["asp_active"])

_sa_df = did_panel.copy()
_treated_cohorts = sorted(
    [int(g) for g in _sa_df["asp_adoption_year"].unique() if int(g) != 9999]
)
_cohort_sizes = {
    g: int(_sa_df.loc[_sa_df["asp_adoption_year"] == g, "hospital_id"].nunique())
    for g in _treated_cohorts
}
_sa_df["event_time"] = np.where(
    _sa_df["asp_adoption_year"] != 9999,
    _sa_df["year"] - _sa_df["asp_adoption_year"],
    np.nan,
)

_sa_cols = []
_sa_meta = {}
for _g in _treated_cohorts:
    _e_vals = sorted(
        [
            int(e)
            for e in _sa_df.loc[_sa_df["asp_adoption_year"] == _g, "event_time"].unique()
            if int(e) != -1
        ]
    )
    for _e in _e_vals:
        _tag = f"m{abs(_e)}" if _e < 0 else f"p{_e}"
        _col = f"d_{_g}_{_tag}"
        _sa_df[_col] = (
            (_sa_df["asp_adoption_year"] == _g) & (_sa_df["event_time"] == _e)
        ).astype(float)
        _sa_cols.append(_col)
        _sa_meta[_col] = (_g, _e)

_formula_sa = "antibiotic_duration ~ C(hospital_id) + C(year) + " + " + ".join(_sa_cols)
_sa_mod = smf.ols(_formula_sa, data=_sa_df).fit(
    cov_type="cluster", cov_kwds={"groups": _sa_df["hospital_id"]}
)
_sa_params = _sa_mod.params.values
_sa_cov = _sa_mod.cov_params().values
_param_idx = {name: i for i, name in enumerate(_sa_mod.params.index)}

_n_clusters = int(_sa_df["hospital_id"].nunique())
_df_t = _n_clusters - 1
_t_crit = float(stats.t.ppf(0.975, df=_df_t))

_all_event_times = sorted(
    [int(e) for e in _sa_df["event_time"].dropna().unique()]
)
_es_rows = []
_lead_stats = {}

for _e in _all_event_times:
    if _e == -1:
        _es_rows.append(
            {"event_time": -1, "coef": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        )
        continue
    _matching = [(_col, _g) for _col, (_g, _ev) in _sa_meta.items() if _ev == _e]
    _total_w = sum(_cohort_sizes[_g] for _, _g in _matching)
    _c = np.zeros(len(_sa_params), dtype=float)
    for _col, _g in _matching:
        _c[_param_idx[_col]] = _cohort_sizes[_g] / _total_w
    _coef = float(np.dot(_c, _sa_params))
    _var = float(_c @ _sa_cov @ _c)
    _se = float(np.sqrt(max(_var, 0.0)))
    _ci_lo = float(_coef - _t_crit * _se)
    _ci_hi = float(_coef + _t_crit * _se)
    _pval = (
        float(2.0 * (1.0 - stats.t.cdf(abs(_coef / _se), df=_df_t)))
        if _se > 0
        else 1.0
    )
    _es_rows.append(
        {"event_time": _e, "coef": _coef, "ci_low": _ci_lo, "ci_high": _ci_hi}
    )
    _lead_stats[_e] = (_coef, _pval, [_ci_lo, _ci_hi])

event_study = pd.DataFrame(_es_rows).sort_values("event_time").reset_index(drop=True)

_pre_e = -2 if -2 in _lead_stats else min([e for e in _lead_stats if e < -1])
pre_trend_coeff = float(_lead_stats[_pre_e][0])
pre_trend_pvalue = float(_lead_stats[_pre_e][1])
pre_trend_ci = [float(_lead_stats[_pre_e][2][0]), float(_lead_stats[_pre_e][2][1])]

_post_matching = [(_col, _g) for _col, (_g, _ev) in _sa_meta.items() if _ev >= 0]
_total_post_w = sum(_cohort_sizes[_g] for _, _g in _post_matching)
_c_att = np.zeros(len(_sa_params), dtype=float)
for _col, _g in _post_matching:
    _c_att[_param_idx[_col]] = _cohort_sizes[_g] / _total_post_w
did_att_estimate = float(np.dot(_c_att, _sa_params))


# 3. TREATMENT HETEROGENEITY (X-Learner adjusted for hospital and year)
_clin_num_cols = ["age", "comorbidity_score", "immune_compromised", "severity_score"]
_clin_cat_cols = ["sex", "ward", "infection_type", "hospital_size", "region"]
_all_clin_dummies = pd.get_dummies(
    df[_clin_cat_cols].astype(str), drop_first=True, dtype=float
)
_all_hy_dummies = pd.get_dummies(
    df[["hospital_id", "year"]].astype(str), drop_first=True, dtype=float
)

_X_clin_full = np.hstack(
    [df[_clin_num_cols].to_numpy(dtype=float), _all_clin_dummies.to_numpy(dtype=float)]
)
_X_hy_full = _all_hy_dummies.to_numpy(dtype=float)
_W_full = df["asp_active"].to_numpy(dtype=int)
_Y_full = df["antibiotic_duration"].to_numpy(dtype=float)


def _fit_predict_x_learner(X_clin, X_hy, W, Y, seed=42):
    _M = np.column_stack([np.ones(len(Y)), W.astype(float), X_clin, X_hy])
    _beta = np.linalg.lstsq(_M, Y, rcond=None)[0]
    _hy_start = 2 + X_clin.shape[1]
    _y_adj = Y - X_hy @ _beta[_hy_start:]

    _mask0 = W == 0
    _mask1 = W == 1

    _mu0 = _make_regressor(seed).fit(X_clin[_mask0], _y_adj[_mask0])
    _mu1 = _make_regressor(seed + 1).fit(X_clin[_mask1], _y_adj[_mask1])

    _d1 = _y_adj[_mask1] - _mu0.predict(X_clin[_mask1])
    _d0 = _mu1.predict(X_clin[_mask0]) - _y_adj[_mask0]

    _tau1 = _make_regressor(seed + 2).fit(X_clin[_mask1], _d1)
    _tau0 = _make_regressor(seed + 3).fit(X_clin[_mask0], _d0)

    _X_scaled = StandardScaler().fit_transform(X_clin)
    _prop = LogisticRegression(max_iter=500, random_state=seed).fit(_X_scaled, W)
    _e = np.clip(_prop.predict_proba(_X_scaled)[:, 1], 0.05, 0.95)

    return _e * _tau0.predict(X_clin) + (1.0 - _e) * _tau1.predict(X_clin)


cate_estimates = np.asarray(
    _fit_predict_x_learner(_X_clin_full, _X_hy_full, _W_full, _Y_full, seed=42),
    dtype=float,
)

_rng = np.random.default_rng(42)
_idx_0 = np.where(_W_full == 0)[0]
_idx_1 = np.where(_W_full == 1)[0]
_ages_full = df["age"].to_numpy(dtype=float)
_boot_diffs = []
_n_boot = 100

for _b in range(_n_boot):
    _b0 = _rng.choice(_idx_0, size=len(_idx_0), replace=True)
    _b1 = _rng.choice(_idx_1, size=len(_idx_1), replace=True)
    _b_idx = np.concatenate([_b0, _b1])
    _cate_b = _fit_predict_x_learner(
        _X_clin_full[_b_idx],
        _X_hy_full[_b_idx],
        _W_full[_b_idx],
        _Y_full[_b_idx],
        seed=100 + _b,
    )
    _ages_b = _ages_full[_b_idx]
    _diff_b = float(
        _cate_b[_ages_b >= 65].mean() - _cate_b[_ages_b < 40].mean()
    )
    _boot_diffs.append(_diff_b)

cate_elderly_minus_young_ci = [
    float(np.percentile(_boot_diffs, 2.5)),
    float(np.percentile(_boot_diffs, 97.5)),
]


# 4. HIERARCHICAL MODELING
_mixed_formula = (
    "antibiotic_duration ~ asp_active + age + C(sex) + comorbidity_score + "
    "immune_compromised + severity_score + C(ward) + C(infection_type) + C(year)"
)
_mixed_mod = smf.mixedlm(_mixed_formula, data=df, groups=df["hospital_id"])
_mixed_res = _mixed_mod.fit(reml=False, method="lbfgs")
if not _mixed_res.converged:
    _mixed_res = _mixed_mod.fit(reml=False, method="powell")

_naive_ols_res = smf.ols(_mixed_formula, data=df).fit()

random_effect_variance = float(_mixed_res.cov_re.iloc[0, 0])
_residual_variance = float(_mixed_res.scale)
icc_value = float(
    random_effect_variance / (random_effect_variance + _residual_variance)
)
mixed_model_aic = (
    float(_mixed_res.aic)
    if np.isfinite(_mixed_res.aic)
    else float(-2.0 * _mixed_res.llf + 2.0 * (len(_mixed_res.fe_params) + 2))
)
naive_ols_aic = float(_naive_ols_res.aic)
mixed_asp_coeff = float(_mixed_res.fe_params["asp_active"])