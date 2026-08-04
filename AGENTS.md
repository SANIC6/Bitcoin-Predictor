# AGENTS.md — Coin-Gecko-API

## What this is

A Python project that fetches Bitcoin market data from the CoinGecko API and explores **whether BTC's next move is predictable** — via a PyTorch LSTM (binary direction classification) plus ARIMA as a statistical sanity check. The working code lives in notebooks; there is no `main.py`. The **active pipeline is `model 2.ipynb`** (direction classification + ARIMA comparison). `main-notebook.ipynb` is the older price-prediction pipeline, kept for reference.

**Current state / key finding:** the ARIMA model (`auto_arima`, picked `(2,0,1)`+intercept by AIC) produced an RMSE **identical to the mean baseline** (0.003121) on log returns — i.e. no exploitable linear autocorrelation. The classification LSTM (BCE loss) is expected to sit near the ~0.69 chance floor. The mean/majority-class baseline is the reference point for all evaluation.

## Essential commands

| Command | Purpose |
|---|---|
| `uv sync` | Install dependencies from `uv.lock` into `.venv` |
| `uv add <package>` | Add a new dependency |
| `uv run python <script>.py` | Run a standalone script (no `main.py` exists yet — notebooks are the source of truth) |

No tests, linter, formatter, or CI are configured yet — nothing automated runs on changes. (`pytest` is declared in `pyproject.toml` but no test files exist.)

## Project structure

```
.
├── model 2.ipynb                # ACTIVE pipeline: direction-classification LSTM + ARIMA analysis
├── main-notebook.ipynb          # Older price-prediction pipeline (reference only)
├── pyproject.toml               # uv-managed project config, Python >=3.13
├── .python-version              # Pins Python 3.13, venv name: coin-gecko-api
├── uv.lock                      # Lockfile
├── knowledge.md                 # Agent knowledge / instructions for this repo
├── bitcoin_lstm_model.pt        # Saved model checkpoint
├── bitcoin_lstm_model_v1.pt     # Saved model checkpoint (v1)
├── .gitignore                   # Ignores __pycache__, .venv, .env
├── .crush/                      # Local tool cache (crush.db, logs/) — not project code; self-ignored
├── .agents/
│   ├── skills/
│   │   ├── coingecko/           # CoinGecko/GeckoTerminal API skill — references/ dir has endpoint docs
│   │   ├── learn/               # Tutoring skill — how to teach the user
│   │   └── context-7/           # Docs-lookup skill — fetches live library docs via Context7 API
│   └── types/                   # TypeScript type definitions for agent configuration
└── AGENTS.md                    # ← This file
```

## Architecture & data flow (from `model 2.ipynb`)

1. **Auth** — API key loaded from env var `COIN_API` via `dotenv`, sent as `x-cg-demo-api-key` header
2. **Fetch** — `GET /api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90` returns prices and total_volumes as parallel arrays of `[timestamp, value]`. Recent runs return **hourly** points (~2160+ rows). Market caps are commented out (not used)
3. **Merge** — Arrays converted to DataFrames (`pd.to_datetime(timestamp, unit='ms')`), merged on `timestamp`
4. **Feature engineering** — `log_return`, `log_return_lag1`, `log_return_lag2`, `volatility_6` (rolling std), `volume_change`, and binary `direction` = `(log_return > 0).astype(int)`; `dropna()`
5. **ARIMA sanity check** — `pmdarima.auto_arima` on raw `log_return` (d=0, non-seasonal, stepwise, AIC). Picked `(2,0,1)`+intercept; multi-step forecast on the test window matched the mean-baseline RMSE exactly → **no usable linear signal in log returns**
6. **Feature matrix** — 8 columns: `[price, volume, log_return, log_return_lag2, log_return_lag1, volatility_6, volume_change, direction]`. **Features z-scored (fit on train split only); the `direction` target (col 7) is left raw 0/1** — z-scoring the classification target would silently break `BCELoss`
7. **Windowing** — 30-step sliding windows; target `y = data[i+30, 7]` (next direction). Custom `TimeSeriesDataset` + DataLoader (batch_size=32, shuffle=True for training)
8. **Model** — LSTM (input 8 → hidden 5, 2 layers, batch_first) → Linear(1) → **Sigmoid** (binary classifier). h0/c0 zeroed each forward pass
9. **Training** — 100 epochs, Adam (lr=0.01), **BCELoss**, train/val/test split 70/15/15; loss curves plotted. Note: the plot's ylabel still says "Huber" (leftover from the regression iteration)
10. **Evaluation** — Test loss cell, plus a **stale regression-eval cell** (Huber/RMSE/z-score undo) that is WRONG for a classifier and should be replaced with accuracy vs the majority-class baseline
11. **Save** — `torch.save({...})` commented out

## Key conventions & patterns

- **Notebook-first development** — The real implementation lives in `model 2.ipynb`; there is no `main.py`. Script-ify from notebook when ready.
- **CoinGecko API key** — Required in `COIN_API` env var (`.env` file). The demo tier is rate-limited to ~10-30 req/min. Cache fetched data rather than hammering the API.
- **Z-score normalization** — Fit on training split only (`mean = train.mean(axis=0)`, `std = train.std(axis=0)`, with `std[std==0] = 1.0` guard). **Never scale the classification target** — features only.
- **Baseline-first evaluation** — The mean baseline (regression) / majority-class baseline (classification) is the reference point. A model that doesn't beat it has learned nothing. The ARIMA result already matched the baseline exactly.
- **Binary classification setup** — Sigmoid output + `BCELoss` (consistent pairing). Model outputs probabilities; threshold at 0.5 for predicted class.
- **Model checkpoint format** — `torch.save({...})` dict containing: `model_state_dict`, `optimizer_state_dict`, `learning_rate`, `num_epochs`, `mean`, `std`, `price_mean`, `price_std`, `training_loss`, `testing_loss`, `rmse_usd`, `mae_usd` (mirrors the older regression pipeline).
- **Hidden state management** — `h0`/`c0` are zeroed on every forward pass (not stateful across batches), which is correct for this use case.
- **Dependencies** — `dotenv`, `matplotlib`, `pandas`, `pmdarima`, `pycoingecko`, `pytest`, `seaborn`, `torch`, `torchmetrics`. The notebooks use raw `requests` + `dotenv`; `pycoingecko` is declared but unused so far.

## Important gotchas

- **There is no `main.py`** — the notebooks are the source of truth. Don't try to `uv run python main.py` expecting real output.
- **No git history** — the repo has zero commits. `git log` will fail.
- **No tests, no linter, no CI** — nothing to verify changes against. Test manually via the notebook.
- **CoinGecko free tier limits** — historical data beyond 365 days requires a paid plan; granularity is automatic (recent `days=90` runs returned hourly points). Don't assume a fixed cadence.
- **Column-index bugs are the recurring trap** — the user has twice predicted the wrong target column (a lagged feature instead of the true next value). Always verify `windows()` target indices against the feature matrix's column order.
- **Cell 11 in `model 2.ipynb` is stale** — it's the old regression evaluation (Huber/RMSE/reverse-normalize) and will print meaningless numbers for the classifier. Replace with accuracy + majority-class baseline.
- **Save cells are commented out** — re-enable to regenerate `.pt` checkpoints; on-disk checkpoints reflect earlier runs.
- **The `pycoingecko` library** is a declared dependency but the notebooks use raw `requests` calls instead. Both approaches work.
- **`.venv` and `.env` are gitignored** — anyone cloning needs to `uv sync` and create their own `.env` with `COIN_API`.
- **`.crush/` is a local tool cache** — not project code; it self-ignores via its own `.gitignore`, so it stays out of git.

## Agent skills in this repo

| Skill file | When it triggers |
|---|---|
| `.agents/skills/coingecko/SKILL.md` | Any crypto market data, token prices, CoinGecko/GeckoTerminal API calls |
| `.agents/skills/learn/learn-skill.md` | User asks learning questions — tutors via guided discovery, never hands over answers |
| `.agents/skills/context-7/SKILL.md` | User asks about library APIs, framework patterns, version-specific behavior — fetches current docs |

The user writes all their own code. Do not write code for them or edit files they intend to write themselves unless explicitly asked. Explain concepts freely, but guide rather than hand over solutions.

## Related files

- `knowledge.md` — Agent knowledge file with detailed instructions, conventions, and gotchas (keep in sync with this file)
- `.agents/types/` — TypeScript type definitions for the agent configuration system (not project code)
