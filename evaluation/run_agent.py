"""Run Gemini agent trials and score them. Finished trials are skipped, so the command can be repeated.

    python evaluation/run_agent.py --model gemini-4-argon --condition original --trials 5
    python evaluation/run_agent.py --model gemini-2.5-pro --condition pinned --trials 10 --start 0
"""
import argparse

from asp_benchmark.agent import run_trial, trial_dir

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["gemini"], default="gemini")
    ap.add_argument("--model", default="gemini-2.5-pro", help="any Gemini model name available to your API key")
    ap.add_argument("--condition", choices=["original", "pinned"], default="original")
    ap.add_argument("--trials", type=int, default=1, help="number of trials to run")
    ap.add_argument("--start", type=int, default=0, help="first trial number (also used as the random seed)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--exec-timeout", type=int, default=900, help="seconds allowed for each generated script")
    a = ap.parse_args()
    for k in range(a.start, a.start + a.trials):
        if (trial_dir(a.backend, a.model, a.condition, k) / "meta.json").exists():
            print(f"skip trial_{k} (already complete)")
            continue
        run_trial(a.backend, a.model, a.condition, k, a.temperature, a.max_attempts, a.exec_timeout)
