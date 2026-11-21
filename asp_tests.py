"""Behavioural tests for ASP benchmark results.

    python asp_tests.py golden_results.json            # prints per-test PASS/FAIL and k/N
    python asp_tests.py runs/.../results.json --json out.json

Every test runs independently (one failure never hides the others), the score is computed,
and no test trusts a self-declared flag: reference values are recomputed from asp_study_data.csv
or compared with the generator's ground truth (ground_truth.json).
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from asp_common import load_results

OUTCOME_FORMULA = ("antibiotic_duration ~ asp_active + age + severity_score + comorbidity_score "
                   "+ immune_compromised + C(ward) + C(infection_type)")


def _reference(df):
    import statsmodels.formula.api as smf
    from lifelines import CoxPHFitter
    from linearmodels.panel import PanelOLS
    d = df.copy()
    d["sex_binary"] = (d.sex == "M").astype(int)
    for k in ("bloodstream", "respiratory", "urinary"):
        d[f"infection_{k}"] = (d.infection_type == k).astype(int)
    d["ward_icu"] = (d.ward == "ICU").astype(int)
    X = ["asp_active", "age", "severity_score", "comorbidity_score", "immune_compromised", "sex_binary",
         "infection_bloodstream", "infection_respiratory", "infection_urinary", "ward_icu"]
    cph = CoxPHFitter().fit(d[X + ["time_to_improvement", "event_observed", "hospital_id"]],
                            "time_to_improvement", "event_observed", cluster_col="hospital_id")
    naive = CoxPHFitter().fit(d[X + ["time_to_improvement", "event_observed"]], "time_to_improvement", "event_observed")
    ref = dict(cox_hr=float(np.exp(cph.params_["asp_active"])), cox_se_robust=float(cph.standard_errors_["asp_active"]),
               cox_se_naive=float(naive.standard_errors_["asp_active"]))
    panel = d.groupby(["hospital_id", "year"]).agg(m=("antibiotic_duration", "mean"),
                                                    a=("asp_adoption_year", "first")).reset_index()
    ref["panel_means"] = panel.m.values  # sorted by hospital, year
    panel["p"] = (panel.year >= panel.a).astype(int)
    ref["twfe"] = float(PanelOLS.from_formula("m ~ p + EntityEffects + TimeEffects",
                                              data=panel.set_index(["hospital_id", "year"])).fit().params["p"])
    for name, y in (("dur", "antibiotic_duration"), ("los", "length_of_stay")):
        m = smf.mixedlm(OUTCOME_FORMULA.replace("antibiotic_duration", y), d, groups=d.hospital_id).fit(reml=True)
        v = float(m.cov_re.iloc[0, 0])
        ref[f"icc_{name}"], ref[f"re_var_{name}"], ref[f"coef_{name}"] = v / (v + float(m.scale)), v, float(m.params["asp_active"])
    return ref


def _num(x):
    x = float(x)
    assert np.isfinite(x), "not finite"
    return x


def _ci(x, name):
    assert x is not None and len(x) == 2 and np.isfinite(x).all() and x[0] < x[1], f"{name} must be [lo, hi] with lo<hi"
    return [float(x[0]), float(x[1])]


def build_tests(R, df, ref, truth):
    T = []
    add = lambda tid, desc: (lambda f: T.append((tid, desc, f)) or f)

    @add("S1", "Cox HR for ASP matches recomputed HR (+-15%) and is >1")
    def _():
        hr = _num(R["cox_hr_asp"]); assert hr > 1, f"HR={hr:.3f} should be >1"
        assert abs(hr / ref["cox_hr"] - 1) < 0.15, f"HR={hr:.3f} vs reference {ref['cox_hr']:.3f}"

    @add("S2", "Cox SE for ASP is plausible for a hospital-clustered model (0.5x to 2x the reference robust SE)")
    def _():
        # The robust SE depends on the covariate set (year / hospital dummies legitimately change it), so this is a
        # plausibility band, not proof of clustering. Whether cluster_col was used is checked from the code in summarize_runs.py.
        se = _num(R["cox_se_asp"]); r = ref["cox_se_robust"]
        assert 0.5 * r < se < 2.0 * r, f"SE={se:.4f}; robust reference {r:.4f}, naive reference {ref['cox_se_naive']:.4f}"

    @add("S3", "PH test p-values reported for >=5 covariates incl. asp_active")
    def _():
        p = R["ph_test_pvalues"]; assert isinstance(p, dict) and len(p) >= 5 and "asp_active" in p, "need dict incl asp_active"
        assert all(0 <= float(v) <= 1 for v in p.values()), "p-values outside [0,1]"

    @add("S4", "DiD panel: 90 rows, unique (hospital,year), contains true hospital-year mean antibiotic_duration")
    def _():
        p = R["did_panel"]; assert isinstance(p, pd.DataFrame) and len(p) == df.hospital_id.nunique() * df.year.nunique(), "expected one row per hospital-year (90)"
        assert not p.duplicated(["hospital_id", "year"]).any(), "duplicate hospital-year"
        p = p.sort_values(["hospital_id", "year"])
        ok = [c for c in p.select_dtypes("number") if np.allclose(p[c].values, ref["panel_means"], atol=1e-6)]
        assert ok, "no column equals mean antibiotic_duration per hospital-year (wrong outcome?)"

    @add("S5", "Event study has CIs, includes a pre-period lead, CIs bracket estimates")
    def _():
        e = R["event_study"]; assert isinstance(e, pd.DataFrame) and {"event_time", "coef", "ci_low", "ci_high"} <= set(e.columns)
        assert (e.event_time <= -2).any(), "no pre-period lead (event_time <= -2)"
        assert ((e.ci_low <= e.coef) & (e.coef <= e.ci_high)).all(), "CI does not bracket coef"

    @add("S6", "Parallel trends: pre-trend coef/p/CI reported and consistent (p>0.05, CI contains 0)")
    def _():
        c, p, ci = _num(R["pre_trend_coeff"]), _num(R["pre_trend_pvalue"]), _ci(R["pre_trend_ci"], "pre_trend_ci")
        assert ci[0] <= c <= ci[1], "coef outside its CI"
        assert p > 0.05 and ci[0] <= 0 <= ci[1], f"parallel trends rejected (p={p:.3f}); data were generated with parallel trends"

    @add("S7", "TWFE ATT matches recomputed TWFE (+-0.3 days)")
    def _():
        v = _num(R["twfe_att_estimate"]); assert abs(v - ref["twfe"]) < 0.3, f"TWFE={v:.3f}, reference {ref['twfe']:.3f}"

    @add("S8", "Staggered-robust ATT within 0.5 days of ground-truth ATT")
    def _():
        v = _num(R["did_att_estimate"]); t = truth["true_att_treated_days"]
        assert abs(v - t) < 0.5, f"ATT={v:.3f}, truth {t:.3f} (TWFE reference {ref['twfe']:.3f})"

    @add("S9", "CATE: one finite value per patient, real heterogeneity (std>0.3)")
    def _():
        c = np.asarray(R["cate_estimates"], float); assert len(c) == len(df) and np.isfinite(c).all(), "wrong length/NaN"
        assert c.std() > 0.3, f"std={c.std():.3f}"

    @add("S10", "CATE: elderly benefit more than young AND mean treated CATE within 1.5 days of truth")
    def _():
        c = np.asarray(R["cate_estimates"], float); a = df.age.values; t = df.asp_active.values
        assert c[a >= 65].mean() < c[a < 40].mean(), "elderly CATE should be more negative than young"
        assert abs(c[t == 1].mean() - truth["true_att_treated_days"]) < 1.5, f"mean treated CATE {c[t == 1].mean():.3f}"

    @add("S11", "CATE bootstrap CI reported for elderly-minus-young difference")
    def _():
        _ci(R["cate_elderly_minus_young_ci"], "cate_elderly_minus_young_ci")

    @add("S12", "ICC within +-50% of ICC recomputed on antibiotic_duration; flags length_of_stay mix-up")
    def _():
        v = _num(R["icc_value"]); assert 0 <= v <= 1
        assert abs(v / ref["icc_dur"] - 1) < 0.5, (f"ICC={v:.4f}; antibiotic_duration ICC={ref['icc_dur']:.3f}, "
                                                 f"length_of_stay ICC={ref['icc_los']:.3f}"
                                                 + (" -> looks like length_of_stay" if abs(v - ref["icc_los"]) < 0.02 else ""))

    @add("S13", "Mixed-model AIC < naive OLS AIC, with both reported")
    def _():
        assert _num(R["mixed_model_aic"]) < _num(R["naive_ols_aic"]), "mixed AIC not lower"

    @add("S14", "Mixed-model ASP coefficient within 1.0 of recomputed; RE variance within 50%")
    def _():
        c, v = _num(R["mixed_asp_coeff"]), _num(R["random_effect_variance"])
        assert abs(c - ref["coef_dur"]) < 1.0, f"coef={c:.3f}, reference {ref['coef_dur']:.3f}"
        assert abs(v / ref["re_var_dur"] - 1) < 0.5, f"RE var={v:.3f}, reference {ref['re_var_dur']:.3f}"
    return T


def preconditions(df):
    """Dataset sanity checks. Not scored: they say nothing about the agent."""
    out = []
    out.append(("P1", "survival columns valid", {"time_to_improvement", "event_observed"} <= set(df) and
                df.event_observed.isin([0, 1]).all() and (df.time_to_improvement > 0).all()))
    out.append(("P2", "censoring rate 10-60%", 0.10 < 1 - df.event_observed.mean() < 0.60))
    out.append(("P3", "panel is 15 hospitals x 6 years", df.hospital_id.nunique() == 15 and df.year.nunique() == 6))
    return out


def run_tests(results, csv="asp_study_data.csv", truth_path="ground_truth.json"):
    df = pd.read_csv(csv); truth = json.load(open(truth_path)); ref = _reference(df)
    rows = []
    for tid, desc, fn in build_tests(results, df, ref, truth):
        try:
            fn(); rows.append(dict(id=tid, desc=desc, passed=True, msg=""))
        except KeyError as e:
            rows.append(dict(id=tid, desc=desc, passed=False, msg=f"missing output {e}"))
        except Exception as e:
            rows.append(dict(id=tid, desc=desc, passed=False, msg=f"{type(e).__name__}: {e}"))
    return rows


def report(rows):
    for r in rows:
        print(f"{r['id']:>4} {'PASS' if r['passed'] else 'FAIL'}  {r['desc']}" + (f"\n        -> {r['msg']}" if r["msg"] else ""))
    k = sum(r["passed"] for r in rows)
    print(f"\nSCORE: {k}/{len(rows)}")
    return k, len(rows)


if __name__ == "__main__":
    args = sys.argv[1:]
    path = args[0] if args else "golden_results.json"
    rows = run_tests(load_results(path))
    for tid, desc, ok in preconditions(pd.read_csv("asp_study_data.csv")):
        print(f"{tid:>4} {'ok  ' if ok else 'BAD '} (unscored) {desc}")
    k, n = report(rows)
    if "--json" in args:
        json.dump(dict(score=k, total=n, tests=rows), open(args[args.index("--json") + 1], "w"), indent=1)
    sys.exit(0 if k == n else 1)
