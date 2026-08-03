# Project knowledge

This file gives Freebuff context about your project: goals, commands, conventions, and gotchas.

## What this is
- **coin-gecko-api** — a Python project that fetches Bitcoin market data from the CoinGecko API (`GET /coins/bitcoin/market_chart`, `days=90`) and feeds it into a PyTorch LSTM price-prediction pipeline (pandas preprocessing, matplotlib loss plots, `torchmetrics` available for evaluation).
- Early-stage repo: there is **no `main.py`** — the entire pipeline lives in `main-notebook.ipynb` (12 cells: fetch → merge → z-score normalize → window → train → evaluate; the checkpoint-save cell is commented out).

## How to work with the user
- **The user is here to learn.** Treat their questions as learning opportunities, not just tasks. Follow the tutoring approach in `.agents/skills/learn/learn-skill.md`: diagnose before teaching, then take one step forward per turn (one focused question + one small scaffold — a hint, a worked parallel example, an inline visual). Prefer guided discovery and worked examples over dumping answers; if they're impatient, narrow the question and hold the line; if they're genuinely stuck, give them a concrete foothold. They're a self-taught Python/ML learner building this crypto pipeline — no professor or grade, so the real obligation is just that they actually learn.
- **Use context7 for up-to-date docs.** If I'm unsure about a library's API, a framework pattern, a version-specific behavior, or anything where docs may have changed (pandas, torch, torchmetrics, the CoinGecko API, etc.), look it up with the context7 skill (`.agents/skills/context-7/SKILL.md`) instead of answering from memory. Workflow: `.agents/skills/context-7/scripts/context7.sh search "<library>"` to find the library ID (prefer the official source, highest score), then `.agents/skills/context-7/scripts/context7.sh docs "<library-id>" "<topic>" "<mode>"` — `code` mode for API references/code examples, `info` mode for conceptual guides. Always extract a specific topic from the user's question (e.g. `"dataloader"` rather than just `"torch"`). Note: the script needs `curl`/`fetch` and `jq` on PATH.
- **Ask open-ended questions, never multiple-choice.** When asking the user anything (spec interviews, design decisions, check-ins), do NOT offer premade answer choices — ask open questions as plain text in the chat (not via the multiple-choice widget, which forces options) and let the user type their own answer. The user explicitly requested this — premade choices rob them of the thinking they want to do themselves. If the user asks for an explanation before deciding, explain first, then ask.
- **Guide, never hand over answers.** The user writes all their own code. NEVER write code for them or edit files they intend to write themselves — unless they explicitly ask (e.g. they did ask me to update `knowledge.md` and to write the spec file). They want to think like a professional engineer with my guidance: hints and questions that push them in the right direction, never finished solutions. Explaining concepts (like how the CoinGecko API works) is fine and welcome. Do not volunteer model architecture or code design unless the user explicitly asks for it.

## Key files
- `main-notebook.ipynb` — the working pipeline (see AGENTS.md for the step-by-step): fetches BTC price/market_cap/volume, merges on timestamp (~99 **daily** rows for `days=90` on the free tier), z-score normalizes (fit on train split only), builds 30-step sliding windows, trains a 1-layer LSTM (input 3 → hidden 5 → Linear(5,1)) for 12 epochs with Adam, evaluates RMSE/MAE in USD against a "copy previous close" baseline. The `torch.save` cell is currently commented out (kernelspec: `coin-gecko-api`, Python 3.13.5).
- `pyproject.toml` — project metadata + dependencies, uv-managed.
- `README.md` — empty; not yet written.
- `bitcoin_lstm_model.pt` / `bitcoin_lstm_model_v1.pt` — saved checkpoints from earlier runs (dict: state_dict, optimizer state, normalization params, metrics).
- `.agents/skills/learn/learn-skill.md` — tutoring skill: how to teach the user when they ask learning questions.
- `.agents/skills/context-7/SKILL.md` — docs-lookup skill: fetches current library documentation via the Context7 REST API.
- `.agents/skills/coingecko/SKILL.md` — CoinGecko/GeckoTerminal API skill; `references/` has endpoint docs.
- `.crush/` — local tool cache (crush.db, logs/); not project code.

## Commands (uv toolchain)
- Setup: `uv sync` (installs deps from `uv.lock` into `.venv`)
- Add a dependency: `uv add <package>`
- Run code: there is **no `main.py`** — run cells in `main-notebook.ipynb` with the `coin-gecko-api` kernel (Python 3.13); use `uv run python <script>.py` for standalone scripts
- Test / lint: none written yet — `pytest` is declared in `pyproject.toml` but no tests exist; no linter config

## Conventions
- Python `>=3.13` required; `.python-version` pins 3.13 and the venv is named `coin-gecko-api`.
- Declared deps: `dotenv`, `matplotlib`, `pandas>=3.0.5`, `pycoingecko`, `pytest`, `torch>=2.13.0`, `torchmetrics>=1.9.0`.
- The notebook fetches with raw `requests` (key from `COIN_API` via `dotenv`), not the `pycoingecko` wrapper.
- `.venv` and `.env` are gitignored.

## Gotchas
- No tests written, no linter/formatter config, and no CI are set up yet — there's nothing automated to run on changes.
- CoinGecko's free tier is rate-limited (~10–30 requests/min), so cache fetched market data rather than hammering the API on every run.
- Granularity is automatic: the `days=90` call returned **daily** points (~99 rows), not hourly. Pass `interval=hourly` explicitly if finer granularity is needed.
- The `torch.save(...)` cell (11) is commented out — the on-disk checkpoints won't refresh until it's re-enabled.
