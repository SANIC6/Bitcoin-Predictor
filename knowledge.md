# Project knowledge

This file gives Freebuff context about your project: goals, commands, conventions, and gotchas.

## What this is
- **coin-gecko-api** — a Python project that fetches Bitcoin market data from the CoinGecko API (`GET /coins/bitcoin/market_chart`, `days=90`) and explores **whether BTC's future movement is predictable**. The user has iterated from raw-price prediction → log-return prediction → binary **direction** classification, and used ARIMA as a statistical check. So far the honest answer the data keeps giving: **log returns show no exploitable linear autocorrelation** (ARIMA's RMSE was identical to the mean baseline), and the classification LSTM's loss sits near the ~0.69 BCE chance floor.
- There is **no `main.py`** — everything lives in notebooks. The **active** pipeline is `model 2.ipynb` (direction classification + ARIMA analysis). `main-notebook.ipynb` is the older price-prediction pipeline (kept for reference). `main-notebook copy.ipynb` no longer exists (it was the log-return iteration, later replaced by `model 2.ipynb`).

## How to work with the user
- **The user is here to learn.** Treat their questions as learning opportunities, not just tasks. Follow the tutoring approach in `.agents/skills/learn/learn-skill.md`: diagnose before teaching, then take one step forward per turn (one focused question + one small scaffold — a hint, a worked parallel example, an inline visual). Prefer guided discovery and worked examples over dumping answers; if they're impatient, narrow the question and hold the line; if they're genuinely stuck, give them a concrete foothold. They're a self-taught Python/ML learner building this crypto pipeline — no professor or grade, so the real obligation is just that they actually learn.
- **Use context7 for up-to-date docs.** If I'm unsure about a library's API, a framework pattern, a version-specific behavior, or anything where docs may have changed (pandas, torch, torchmetrics, pmdarima/statsmodels, the CoinGecko API, etc.), look it up with the context7 skill (`.agents/skills/context-7/SKILL.md`) instead of answering from memory. Workflow: `.agents/skills/context-7/scripts/context7.sh search "<library>"` to find the library ID (prefer the official source, highest score), then `.agents/skills/context-7/scripts/context7.sh docs "<library-id>" "<topic>" "<mode>"` — `code` mode for API references/code examples, `info` mode for conceptual guides. Always extract a specific topic from the user's question (e.g. `"dataloader"` rather than just `"torch"`). Note: the script needs `curl`/`fetch` and `jq` on PATH.
- **Ask open-ended questions, never multiple-choice.** When asking the user anything (spec interviews, design decisions, check-ins), do NOT offer premade answer choices — ask open questions as plain text in the chat (not via the multiple-choice widget, which forces options) and let the user type their own answer. The user explicitly requested this — premade choices rob them of the thinking they want to do themselves. If the user asks for an explanation before deciding, explain first, then ask.
- **Guide, never hand over answers.** The user writes all their own code. NEVER write code for them or edit files they intend to write themselves — unless they explicitly ask (e.g. they did ask me to update `knowledge.md` and to write the spec file). They want to think like a professional engineer with my guidance: hints and questions that push them in the right direction, never finished solutions. Explaining concepts (like how the CoinGecko API works, stationarity, baselines, ARIMA) is fine and welcome. Do not volunteer model architecture or code design unless the user explicitly asks for it.

## Key files
- `model 2.ipynb` — **the active pipeline** (13 cells):
  1. Fetch price + volume from CoinGecko, merge on timestamp
  2. Feature engineering: `log_return`, `log_return_lag1`, `log_return_lag2`, `volatility_6`, `volume_change`, and binary `direction` (1=up, 0=down)
  3. **ARIMA check** (`pmdarima.auto_arima` on `log_return`, `d=0`, no seasonality) — picked `(2,0,1)`+intercept by AIC
  4. ARIMA multi-step forecast vs **mean baseline** — **RMSE identical (0.003121)** → no usable linear signal
  5. Correlation heatmap (seaborn)
  6. Feature matrix (8 cols: price, volume, log_return, lag2, lag1, volatility_6, volume_change, **direction**): **features z-scored (fit on train only), target `direction` left raw 0/1**; 30-step windows predicting `data[i+30, 7]`; custom `TimeSeriesDataset` + DataLoaders
  7. Classification LSTM: LSTM(hidden=5, layers=2) → Linear(1) → **Sigmoid** (outputs probabilities)
  8. Train loop (per-epoch train/val loss)
  9. 100 epochs, Adam lr=0.01, **BCELoss**, loss plot (⚠️ ylabel still says "Huber" — leftover)
  10. Test loss
  11. **STALE regression-eval cell (Huber/RMSE/z-score-undo)** — prints meaningless numbers for a classifier; flagged to be replaced with accuracy + majority-class baseline
  12. `torch.save(...)` — commented out
- `main-notebook.ipynb` — older price-prediction pipeline (raw price target, RMSE/MAE in USD vs "copy previous close" baseline). Not the current focus, but its eval cells show the correct reverse-normalization pattern.
- `pyproject.toml` — project metadata + dependencies, uv-managed (now includes `pmdarima`, `seaborn`).
- `README.md` — empty; not yet written.
- `bitcoin_lstm_model.pt` / `bitcoin_lstm_model_v1.pt` — saved checkpoints from earlier runs (dict: state_dict, optimizer state, normalization params, metrics).
- `.agents/skills/learn/learn-skill.md` — tutoring skill: how to teach the user when they ask learning questions.
- `.agents/skills/context-7/SKILL.md` — docs-lookup skill: fetches current library documentation via the Context7 REST API.
- `.agents/skills/coingecko/SKILL.md` — CoinGecko/GeckoTerminal API skill; `references/` has endpoint docs.
- `.crush/` — local tool cache (crush.db, logs/); not project code.

## Commands (uv toolchain)
- Setup: `uv sync` (installs deps from `uv.lock` into `.venv`)
- Add a dependency: `uv add <package>`
- Run code: there is **no `main.py`** — run cells in the notebooks with the `coin-gecko-api` kernel (Python 3.13); use `uv run python <script>.py` for standalone scripts
- Test / lint: none written yet — `pytest` is declared in `pyproject.toml` but no tests exist; no linter config

## Conventions
- Python `>=3.13` required; `.python-version` pins 3.13 and the venv is named `coin-gecko-api`.
- Declared deps: `dotenv`, `matplotlib`, `pandas>=3.0.5`, `pmdarima>=2.1.1`, `pycoingecko`, `pytest`, `seaborn>=0.13.2`, `torch>=2.13.0`, `torchmetrics>=1.9.0`.
- The notebooks fetch with raw `requests` (key from `COIN_API` via `dotenv`), not the `pycoingecko` wrapper.
- `.venv` and `.env` are gitignored.

## Gotchas
- No tests written, no linter/formatter config, and no CI are set up yet — there's nothing automated to run on changes.
- CoinGecko's free tier is rate-limited (~10–30 requests/min), so cache fetched market data rather than hammering the API on every run.
- Granularity is automatic: recent `days=90` calls have returned **hourly** points (~2160+ rows). Don't assume a fixed cadence.
- **Column-index bugs are the recurring trap.** The user has hit wrong-target-index bugs twice (predicting a lagged feature instead of the true next value). When reviewing `windows()` / feature matrices, always verify the target column index matches the intended column.
- **Never z-score the classification target.** In `model 2.ipynb`, features are normalized but `direction` (col 7) must stay raw 0/1 or `BCELoss` trains on garbage. Verified fixed in the current cell 6.
- Cell 11 in `model 2.ipynb` is a stale regression-evaluation cell — it computes Huber/RMSE and reverse-normalizes sigmoid outputs as if they were log returns. It will print meaningless numbers until replaced with accuracy vs majority-class baseline.
- The `torch.save(...)` cells are commented out — the on-disk checkpoints won't refresh until re-enabled.
- **Key result so far (do not lose this):** ARIMA (2,0,1) and the mean baseline produced identical RMSE (0.003121) on log returns — the baseline is the reference point, and the classification model needs to beat ~0.69 BCE loss / ~50% accuracy to show any signal.
