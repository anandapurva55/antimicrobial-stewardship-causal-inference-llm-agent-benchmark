"""Gemini agent: prompts a model, runs its code in a subprocess, feeds back errors, scores the result.

Command line: `python evaluation/run_agent.py --model gemini-2.5-pro --condition original --trials 5`.

Backend: Gemini (needs GEMINI_API_KEY in env or .env).
There is NO model fallback: if the requested model is unavailable the run fails loudly.
Generated code runs in a separate subprocess (timeout) in the trial directory, never via exec().

Layout: results/runs/<backend>__<model>/<condition>/trial_<k>/
  prompt.txt, attempt_<n>.py, attempt_<n>_stderr.txt, meta.json, results.json, score.json
"""
import json, os, re, subprocess, sys, time
import pandas as pd
from asp_benchmark.common import build_prompt, load_results
from asp_benchmark.paths import CSV, RUNS_DIR, SRC_DIR
from asp_benchmark.scoring import run_tests

PRELUDE = f"""import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, {str(SRC_DIR)!r})
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf, scipy.stats as stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from linearmodels.panel import PanelOLS
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
df = pd.read_csv({str(CSV)!r})
"""
EPILOGUE = "\nfrom asp_benchmark.common import save_results\nsave_results('results.json', globals())\n"
FEEDBACK = """Your code raised this error when executed:

{err}

AVAILABLE PACKAGES (do not use anything else): lifelines, linearmodels, statsmodels, sklearn, pandas, numpy, scipy.
Do NOT use !pip install. Provide the complete corrected Python code only. No explanations."""


def extract_code(text):
    m = re.search(r"```(?:python)?\n?(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


class Gemini:
    def __init__(self, model, temperature, seed):
        from dotenv import load_dotenv
        load_dotenv()
        from google import genai
        from google.genai import types
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.exit("GEMINI_API_KEY not set (put it in .env)")
        self.model = model
        self.client = genai.Client(api_key=key)  # keep a reference or the HTTP client is closed
        limit = self.client.models.get(model=model).output_token_limit or 65536  # fails loudly if the model is unavailable
        self.chat = self.client.chats.create(
            model=model, config=types.GenerateContentConfig(temperature=temperature, seed=seed,
                                                            max_output_tokens=min(65536, limit)))
        self.version = None

    def send(self, msg):
        # Transient network/server errors (e.g. 'Server disconnected') are retried with backoff.
        # These retries re-send the same message; they are not extra agent attempts.
        for i in range(6):
            try:
                r = self.chat.send_message(msg)
                break
            except Exception as e:
                if i == 5:
                    raise
                wait = 10 * 2 ** i
                print(f"  API error ({type(e).__name__}); retry {i + 1}/5 in {wait}s", flush=True)
                time.sleep(wait)
        self.version = getattr(r, "model_version", None)
        return r.text or ""


def run_generated_code(code, tdir, timeout):
    """Run `code` (with the prelude and epilogue) in a separate Python process inside `tdir`.
    Returns the error text, or an empty string when the code ran without error."""
    script = os.path.join(tdir, "_run.py")
    open(script, "w").write(PRELUDE + code + EPILOGUE)
    try:
        p = subprocess.run([sys.executable, "-I", "_run.py"], cwd=tdir, capture_output=True, text=True, timeout=timeout)
        err = p.stderr[-3000:] if p.returncode else ""
    except subprocess.TimeoutExpired:
        err = f"TimeoutError: code exceeded {timeout}s"
    finally:
        os.remove(script)
    return err


def trial_dir(backend, model, condition, k):
    return RUNS_DIR / f"{backend}__{model.replace(':', '-')}" / condition / f"trial_{k}"


def run_trial(backend, model, condition, k, temperature, max_attempts, exec_timeout):
    tdir = str(trial_dir(backend, model, condition, k))
    os.makedirs(tdir, exist_ok=True)
    df = pd.read_csv(CSV)
    prompt = build_prompt(df, condition)
    open(os.path.join(tdir, "prompt.txt"), "w").write(prompt)
    agent = Gemini(model, temperature, k)
    meta = dict(backend=backend, model_requested=model, condition=condition, trial=k, seed=k, temperature=temperature,
                started=time.strftime("%Y-%m-%dT%H:%M:%S"), attempts=0, executed_ok=False, errors=[])
    msg = prompt
    for n in range(1, max_attempts + 1):
        meta["attempts"] = n
        code = extract_code(agent.send(msg))
        meta["model_reported"] = agent.version
        open(os.path.join(tdir, f"attempt_{n}.py"), "w").write(code)
        err = run_generated_code(code, tdir, exec_timeout)
        open(os.path.join(tdir, f"attempt_{n}_stderr.txt"), "w").write(err)
        if not err:
            meta["executed_ok"] = True
            break
        meta["errors"].append(err.strip().splitlines()[-1][:300])
        msg = FEEDBACK.format(err=err[-1500:])
    results_path = os.path.join(tdir, "results.json")
    if meta["executed_ok"] and os.path.exists(results_path):
        rows = run_tests(load_results(results_path))
    else:
        rows = [dict(id=f"S{i}", desc="", passed=False, msg="no results") for i in range(1, 15)]
    score = dict(score=sum(r["passed"] for r in rows), total=len(rows), tests=rows)
    json.dump(score, open(os.path.join(tdir, "score.json"), "w"), indent=1)
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(meta, open(os.path.join(tdir, "meta.json"), "w"), indent=1)
    print(f"[{backend}/{model}/{condition}/trial_{k}] attempts={meta['attempts']} executed={meta['executed_ok']} "
          f"score={score['score']}/{score['total']}", flush=True)
    return score
