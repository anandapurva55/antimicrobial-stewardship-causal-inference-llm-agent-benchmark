"""Aggregate stored trials into runs/summary.json and runs/comparison.md.

    python summarize_runs.py

* "Valid results" = the generated code ran and defined at least one of the required outputs.
* "Cox clustered (code)" is a static check of the final generated script for `cluster_col`; it is not scored.
* The model comparison uses trials 0 to COMPARE_N-1 of the original prompt for every model, so n is equal.
"""
import collections as C
import glob
import json
import os

COMPARE_N = 5
HEADER = ["| Model | Prompt | Trials | Valid results | Mean score (all) | Mean score (valid) | Best | ICC test (S12) | Mean attempts | Cox clustered (code) |",
          "|---|---|---|---|---|---|---|---|---|---|"]


def load():
    rows = C.defaultdict(list)
    for sp in sorted(glob.glob("runs/*/*/trial_*/score.json")):
        d = os.path.dirname(sp)
        parts = d.split(os.sep)
        meta = json.load(open(os.path.join(d, "meta.json")))
        final = os.path.join(d, f"attempt_{meta['attempts']}.py")
        rp = os.path.join(d, "results.json")
        meta["valid"] = bool(meta["executed_ok"] and os.path.exists(rp) and json.load(open(rp)))
        meta["clustered"] = os.path.exists(final) and "cluster_col" in open(final).read()
        rows[(parts[1].replace("gemini__", ""), parts[2])].append((json.load(open(sp)), meta))
    return rows


def stats(v):
    valid = [(s, m) for s, m in v if m["valid"]]
    all_sc, ok_sc = [s["score"] for s, _ in v], [s["score"] for s, _ in valid]
    fails = C.Counter(t["id"] for s, _ in valid for t in s["tests"] if not t["passed"])
    return dict(
        trials=len(v), valid=len(valid), scores=all_sc,
        mean_all=sum(all_sc) / len(v), mean_valid=(sum(ok_sc) / len(ok_sc)) if ok_sc else None, best=max(all_sc),
        s12=sum(next(t["passed"] for t in s["tests"] if t["id"] == "S12") for s, _ in valid),
        attempts=sum(m["attempts"] for _, m in v) / len(v), clustered=sum(m["clustered"] for _, m in valid),
        failures=dict(sorted(fails.items(), key=lambda kv: int(kv[0][1:]))))


def line(model, cond, st):
    mv = "n/a" if st["mean_valid"] is None else f"{st['mean_valid']:.1f}"
    return (f"| {model} | {cond} | {st['trials']} | {st['valid']}/{st['trials']} | {st['mean_all']:.1f} | {mv} | {st['best']} | "
            f"{st['s12']}/{st['valid']} | {st['attempts']:.1f} | {st['clustered']}/{st['valid']} |")


rows = load()
summary = {f"{m}/{c}": stats(v) for (m, c), v in rows.items()}
json.dump(summary, open("runs/summary.json", "w"), indent=1)

compare = {m: stats([(s, x) for s, x in v if x["trial"] < COMPARE_N]) for (m, c), v in rows.items() if c == "original"}
ranked = sorted(compare.items(), key=lambda kv: (-kv[1]["mean_all"], kv[1]["attempts"]))
out = HEADER + [line(m, "original", st) for m, st in ranked]
open("runs/comparison.md", "w").write("\n".join(out) + "\n")

print(f"Equal-n comparison (trials 0-{COMPARE_N - 1}, original prompt)\n" + "\n".join(out))
print("\nAll completed trials\n" + "\n".join(HEADER + [line(m, c, st) for (m, c), st in
      sorted(((k, stats(v)) for k, v in rows.items()), key=lambda kv: (kv[0][1], -kv[1]["mean_all"]))]))
for m, st in ranked:
    print(m, "failures among valid runs:", st["failures"])
