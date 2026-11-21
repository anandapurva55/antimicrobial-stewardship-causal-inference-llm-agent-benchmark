"""Generate the synthetic ASP study dataset and its ground truth.

    python -m asp_benchmark.data_generation              # writes data/asp_study_data.csv + data/ground_truth.json (seed 42)
    python -m asp_benchmark.data_generation --seed 7     # a fresh dataset; set ASP_DATA_DIR to keep the committed files

Design (everything the benchmark claims is set here, so it can be checked):

* 15 hospitals, 5 regions, 2017-2022 (the 2020 and 2021 cohorts then have 3 and 4 pre-periods). Hospitals 1-5 adopt ASP in 2020, 6-9 in 2021,
  10-15 never (adoption correlates with size: better-resourced hospitals adopt earlier).
* antibiotic_duration (primary outcome) has a hospital random intercept (SD 1.8 days)
  -> ICC ~ 0.2. length_of_stay has a tiny hospital effect -> ICC ~ 0.04 (the trap).
* The ASP effect is heterogeneous across cohorts AND dynamic in exposure time:
    2020 cohort: -2.0, -3.0, -4.0 days at event time 0, 1, 2
    2021 cohort: -2.5, -2.5 days at event time 0, 1
  and 1.32x larger for patients aged 65+ (so TWFE with staggered adoption is not
  guaranteed to be unbiased, unlike a constant-effect design).
* Survival: lognormal time-to-improvement, ASP speeds recovery. Censoring is NOT
  "study end": it is a 28-day follow-up cap plus independent loss to follow-up
  (transfer/withdrawal), unrelated to calendar year.
"""
import json
import numpy as np
import pandas as pd
from asp_benchmark.paths import CSV, DATA_DIR, TRUTH

SEED = 42
YEARS = [2017, 2018, 2019, 2020, 2021, 2022]
ADOPT = {**{h: 2020 for h in range(1, 6)}, **{h: 2021 for h in range(6, 10)},
         **{h: 9999 for h in range(10, 16)}}
REGIONS = ["North", "South", "East", "West", "Central"]
SIZE = {h: ("large" if h in (1, 2, 6, 10, 11) else "medium" if h in (3, 4, 7, 8, 12, 13) else "small")
        for h in range(1, 16)}
N_PER_HY = {"large": 170, "medium": 130, "small": 90}
HOSP_SD_DUR, HOSP_SD_LOS, RESID_SD_DUR = 1.8, 0.3, 2.9
EFFECT = {2020: {0: -2.0, 1: -3.0, 2: -4.0}, 2021: {0: -2.5, 1: -2.5}}
ELDERLY_MULT = 1.32
FOLLOWUP_CAP = 28.0


def generate(seed=SEED):
    rng = np.random.default_rng(seed)
    hosp_dur = {h: rng.normal(0, HOSP_SD_DUR) for h in ADOPT}
    hosp_los = {h: rng.normal(0, HOSP_SD_LOS) for h in ADOPT}
    rows = []
    for h in ADOPT:
        for y in YEARS:
            n = int(rng.poisson(N_PER_HY[SIZE[h]]))
            for _ in range(n):
                rows.append((h, y))
    df = pd.DataFrame(rows, columns=["hospital_id", "year"])
    n = len(df)
    df["hospital_size"] = df.hospital_id.map(SIZE)
    df["region"] = df.hospital_id.map(lambda h: REGIONS[(h - 1) % 5])
    df["asp_adoption_year"] = df.hospital_id.map(ADOPT)
    df["asp_active"] = (df.year >= df.asp_adoption_year).astype(int)

    df["ward"] = rng.choice(["general", "ICU", "surgical"], n, p=[0.55, 0.2, 0.25])
    df["age"] = np.clip(rng.normal(58, 17, n), 18, 95).round().astype(int)
    df["sex"] = rng.choice(["M", "F"], n)
    df["age_group"] = pd.cut(df.age, [0, 39, 64, 200], labels=["<40", "40-65", "65+"]).astype(str)
    df["comorbidity_score"] = np.clip(rng.poisson(1.8 + (df.age - 58) / 40, n), 0, 8)
    df["immune_compromised"] = (rng.random(n) < 0.12).astype(int)
    df["infection_type"] = rng.choice(["respiratory", "urinary", "bloodstream", "wound"], n,
                                      p=[0.35, 0.3, 0.15, 0.2])
    sev = 40 + 0.25 * (df.age - 58) + 4 * df.comorbidity_score + 10 * (df.ward == "ICU") + rng.normal(0, 12, n)
    df["severity_score"] = np.clip(sev, 10, 100).round().astype(int)

    # ---- ground-truth treatment effect on antibiotic duration (days) ----
    event_time = df.year - df.asp_adoption_year
    tau = np.zeros(n)
    for g, d in EFFECT.items():
        for k, v in d.items():
            tau[(df.asp_adoption_year == g) & (event_time == k)] = v
    tau = tau * np.where(df.age >= 65, ELDERLY_MULT, 1.0)
    df["_tau"] = tau

    inf_eff = df.infection_type.map({"respiratory": 0.0, "urinary": -0.5, "bloodstream": 2.0, "wound": 0.8})
    base = (7.5 + 0.03 * df.severity_score + 0.25 * df.comorbidity_score + 1.0 * (df.ward == "ICU")
            + 0.5 * df.immune_compromised + inf_eff - 0.2 * (df.year - 2017)
            + df.hospital_id.map(hosp_dur))
    dur = base + df._tau + rng.normal(0, RESID_SD_DUR, n)
    df["antibiotic_duration"] = np.clip(dur, 1, None).round(1)

    # ---- survival: lognormal time to improvement, ASP speeds recovery ----
    lin = (1.85 + 0.012 * df.severity_score + 0.10 * df.comorbidity_score + 0.25 * (df.ward == "ICU")
           + 0.002 * (df.age - 58) - 0.35 * df.asp_active)
    t_event = np.exp(lin + rng.normal(0, 0.55, n))
    t_loss = rng.exponential(60.0, n)             # independent loss to follow-up
    t_cens = np.minimum(t_loss, FOLLOWUP_CAP)
    df["event_observed"] = (t_event <= t_cens).astype(int)
    df["time_to_improvement"] = np.maximum(np.minimum(t_event, t_cens), 0.1).round(1)

    # ---- secondary outcomes ----
    df["length_of_stay"] = np.clip(np.round(
        2 + 0.06 * df.severity_score + 2.0 * (df.ward == "ICU") + 0.2 * df.antibiotic_duration
        + df.hospital_id.map(hosp_los) + rng.normal(0, 3.5, n)), 1, None).astype(int)
    p = lambda z: 1 / (1 + np.exp(-z))
    df["c_diff_infection"] = (rng.random(n) < p(-4.2 + 0.12 * df.antibiotic_duration)).astype(int)
    df["mortality"] = (rng.random(n) < p(-5.2 + 0.035 * df.severity_score + 0.02 * (df.age - 58))).astype(int)
    df["readmission_30d"] = (rng.random(n) < p(-3.0 + 0.2 * df.comorbidity_score)).astype(int)

    df.insert(0, "patient_id", np.arange(1, n + 1))
    truth = {
        "seed": seed,
        "n_rows": int(n),
        "true_att_treated_days": float(df.loc[df.asp_active == 1, "_tau"].mean()),
        "true_att_by_event_time": {f"{g}:{k}": v for g, d in EFFECT.items() for k, v in d.items()},
        "true_att_cohort_agg_unweighted_young": float(
            df.loc[(df.asp_active == 1) & (df.age < 65), "_tau"].mean()),
        "true_att_elderly": float(df.loc[(df.asp_active == 1) & (df.age >= 65), "_tau"].mean()),
        "elderly_effect_multiplier": ELDERLY_MULT,
        "hospital_sd_duration": HOSP_SD_DUR,
        "residual_sd_duration": RESID_SD_DUR,
        "hospital_sd_los": HOSP_SD_LOS,
        "cox_log_hr_asp_generating": 0.35,
        "followup_cap_days": FOLLOWUP_CAP,
        "censoring_note": "28-day cap + independent exponential loss to follow-up (mean 60d); not calendar-driven",
    }
    out = df.drop(columns=["_tau"])
    cols = ["patient_id", "hospital_id", "hospital_size", "region", "ward", "year", "age", "sex", "age_group",
            "comorbidity_score", "immune_compromised", "infection_type", "severity_score",
            "asp_adoption_year", "asp_active", "time_to_improvement", "event_observed",
            "antibiotic_duration", "length_of_stay", "readmission_30d", "c_diff_infection", "mortality"]
    return out[cols], truth


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    data, truth = generate(ap.parse_args().seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(CSV, index=False)
    with open(TRUTH, "w") as f:
        json.dump(truth, f, indent=2)
    print(f"wrote {CSV} {data.shape}; true ATT on treated = {truth['true_att_treated_days']:.3f}")
    print("censoring rate:", round(1 - data.event_observed.mean(), 3))
    print(data.groupby("year").event_observed.mean().round(3).to_dict())
