"""Aggregate stored trials into results/summary.json and results/comparison.md.

    python evaluation/summarize_runs.py

* "Valid results" = the generated code ran and defined at least one of the required outputs.
* "Cox clustered (code)" is a static check of the final generated script for `cluster_col`; it is not scored.
* Every model has 5 trials per prompt.
* Intervals are 95% percentile bootstrap intervals over trials (10,000 resamples, fixed seed). With 5 to 10 trials per
  model they are wide, and they are reported so that the size of the uncertainty is visible.
"""
import collections as C
import glob
import json
import os

import numpy as np

from asp_benchmark.paths import RESULTS_DIR, RUNS_DIR

HEADER = ["| Model | Prompt | Trials | Valid results | Mean score (all) | 95% CI of mean | Mean score (valid) | Best | ICC test (S12) | Mean attempts | Cox clustered (code) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]


def bootstrap_ci(values, resamples=10_000, seed=0):
    """95% percentile bootstrap interval of the mean."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    means = rng.choice(v, (resamples, len(v))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_diff_ci(a, b, resamples=10_000, seed=0):
    """95% percentile bootstrap interval of mean(b) minus mean(a)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = rng.choice(b, (resamples, len(b))).mean(axis=1) - rng.choice(a, (resamples, len(a))).mean(axis=1)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def load():
    rows = C.defaultdict(list)
    for sp in sorted(glob.glob(str(RUNS_DIR / "*/*/trial_*/score.json"))):
        d = os.path.dirname(sp)
        parts = d.split(os.sep)[-3:]  # model directory, prompt condition, trial
        meta = json.load(open(os.path.join(d, "meta.json")))
        final = os.path.join(d, f"attempt_{meta['attempts']}.py")
        rp = os.path.join(d, "results.json")
        meta["valid"] = bool(meta["executed_ok"] and os.path.exists(rp) and json.load(open(rp)))
        meta["clustered"] = os.path.exists(final) and "cluster_col" in open(final).read()
        rows[(parts[0].replace("gemini__", ""), parts[1])].append((json.load(open(sp)), meta))
    return rows


def stats(v):
    valid = [(s, m) for s, m in v if m["valid"]]
    all_sc, ok_sc = [s["score"] for s, _ in v], [s["score"] for s, _ in valid]
    fails = C.Counter(t["id"] for s, _ in valid for t in s["tests"] if not t["passed"])
    return dict(
        trials=len(v), valid=len(valid), scores=all_sc,
        mean_all=sum(all_sc) / len(v), ci_all=bootstrap_ci(all_sc), mean_valid=(sum(ok_sc) / len(ok_sc)) if ok_sc else None, best=max(all_sc),
        s12=sum(next(t["passed"] for t in s["tests"] if t["id"] == "S12") for s, _ in valid),
        attempts=sum(m["attempts"] for _, m in v) / len(v), clustered=sum(m["clustered"] for _, m in valid),
        failures=dict(sorted(fails.items(), key=lambda kv: int(kv[0][1:]))))


def line(model, cond, st):
    mv = "n/a" if st["mean_valid"] is None else f"{st['mean_valid']:.1f}"
    ci = f"{st['ci_all'][0]:.1f} to {st['ci_all'][1]:.1f}"
    return (f"| {model} | {cond} | {st['trials']} | {st['valid']}/{st['trials']} | {st['mean_all']:.1f} | {ci} | {mv} | {st['best']} | "
            f"{st['s12']}/{st['valid']} | {st['attempts']:.1f} | {st['clustered']}/{st['valid']} |")


rows = load()
summary = {f"{m}/{c}": stats(v) for (m, c), v in rows.items()}
if "gemini-2.5-pro/original" in summary and "gemini-2.5-pro/pinned" in summary:
    lo, hi = bootstrap_diff_ci(summary["gemini-2.5-pro/original"]["scores"], summary["gemini-2.5-pro/pinned"]["scores"])
    summary["gemini-2.5-pro/pinned_minus_original"] = dict(mean_difference=summary["gemini-2.5-pro/pinned"]["mean_all"] - summary["gemini-2.5-pro/original"]["mean_all"], ci95=[lo, hi])
json.dump(summary, open(RESULTS_DIR / "summary.json", "w"), indent=1)

ranked = sorted(summary.items(), key=lambda kv: (kv[0].split("/")[1] != "original", -kv[1]["mean_all"], kv[1]["attempts"])
                if "pinned_minus_original" not in kv[0] else (9, 0, 0))
ranked = [(k, v) for k, v in ranked if "pinned_minus_original" not in k]
out = HEADER + [line(k.split("/")[0], k.split("/")[1], st) for k, st in ranked]
open(RESULTS_DIR / "comparison.md", "w").write("\n".join(out) + "\n")

print("\n".join(out))
for k, st in ranked:
    print(k, "failures among valid runs:", st["failures"])
if "gemini-2.5-pro/pinned_minus_original" in summary:
    d = summary["gemini-2.5-pro/pinned_minus_original"]
    print(f"gemini-2.5-pro pinned minus original: {d['mean_difference']:+.1f} (95% CI {d['ci95'][0]:.1f} to {d['ci95'][1]:.1f})")
