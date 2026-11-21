"""Re-score every stored trial from its results.json (offline, no API calls). Use after changing asp_tests.py."""
import glob, json, os
from asp_common import load_results
from asp_tests import run_tests

for sp in sorted(glob.glob("runs/*/*/trial_*/score.json")):
    d = os.path.dirname(sp); rp = os.path.join(d, "results.json")
    if not os.path.exists(rp):
        continue
    rows = run_tests(load_results(rp))
    json.dump(dict(score=sum(r["passed"] for r in rows), total=len(rows), tests=rows), open(sp, "w"), indent=1)
    print(d, sum(r["passed"] for r in rows), "/", len(rows))
