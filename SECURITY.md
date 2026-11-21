# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub. Open the **Security** tab of this repository and choose **Report a vulnerability**, or use [the report form](https://github.com/anandapurva55/antimicrobial-stewardship-causal-inference-llm-agent-benchmark/security/advisories/new).

Include what you found, the steps to reproduce it and what an attacker could gain. You can expect a reply within seven days, and a confirmed problem is fixed before it is discussed in public.

## Scope

The parts of this project that handle sensitive input are the Gemini API key in `.env`, which is excluded from version control, and the code that `evaluation/run_agent.py` receives from a model and runs in a separate process with a time limit and without the API key in its environment. That process is not a sandbox, and it can still read files such as `.env`. Reports about these two areas are the most useful. The dataset is synthetic and contains no personal or clinical data.
