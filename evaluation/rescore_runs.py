"""Re-score every stored trial from its results.json (offline, no API calls). Use after changing the scoring tests.

    python evaluation/rescore_runs.py
"""
import glob, json, os
from asp_benchmark.common import load_results
from asp_benchmark.paths import RUNS_DIR
from asp_benchmark.scoring import run_tests

for sp in sorted(glob.glob(str(RUNS_DIR / "*/*/trial_*/score.json"))):
    d = os.path.dirname(sp); rp = os.path.join(d, "results.json")
    if not os.path.exists(rp):
        continue
    rows = run_tests(load_results(rp))
    json.dump(dict(score=sum(r["passed"] for r in rows), total=len(rows), tests=rows), open(sp, "w"), indent=1)
    print(d, sum(r["passed"] for r in rows), "/", len(rows))
