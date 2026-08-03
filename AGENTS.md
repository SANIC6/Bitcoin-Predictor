# AGENTS.md — Coin-Gecko-API

## What this is

A Python project that fetches Bitcoin market data from the CoinGecko API and feeds it into a PyTorch LSTM model for price prediction. The real working code lives in `main-notebook.ipynb` — there is no `main.py` script yet (the project is early-stage).

## Essential commands

| Command | Purpose |
|---|---|
| `uv sync` | Install dependencies from `uv.lock` into `.venv` |
| `uv add <package>` | Add a new dependency |
| `uv run python <script>.py` | Run a standalone script (no `main.py` exists yet — the notebook is the source of truth) |

No tests, linter, formatter, or CI are configured yet — nothing automated runs on changes. (`pytest` is declared in `pyproject.toml` but no test files exist.)

## Project structure

```
.
├── main-notebook.ipynb          # Core implementation: data fetch, preprocessing, model, training, eval
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

## Architecture & data flow (from notebook)

1. **Auth** — API key loaded from env var `COIN_API` via `dotenv`, sent as `x-cg-demo-api-key` header
2. **Fetch** — `GET /api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90` returns prices, market_caps, total_volumes as parallel arrays of `[timestamp, value]`. Granularity is automatic — on the free tier this call came back as **daily** points (~99 rows covering ~99 days), not hourly. Pass `interval=hourly` explicitly if finer data is needed
3. **Merge** — Each array is converted to a DataFrame (`pd.to_datetime(timestamp, unit='ms')`), then merged on `timestamp` into one combined DataFrame (columns: price, market_cap, volume)
4. **Normalize** — Z-score standardization (mean/std computed on training split only to avoid data leakage)
5. **Windowing** — Sliding windows of 30 timesteps, each predicting the next period's price (feature `y = data[i+30, 0]` — the price column)
6. **Dataset** — Custom `TimeSeriesDataset` class wrapping numpy arrays, fed through `DataLoader` (batch_size=32, shuffle=True for training)
7. **Model** — Single-layer LSTM (`input_size=3 → hidden_size=5 → Linear(5,1)`), batch_first=True, hidden/cell state zeroed each forward pass
8. **Training** — 12 epochs, Adam optimizer (lr=0.01, weight_decay=1e-4), MSE loss, train/val/test split 70/15/15; loss curves plotted with matplotlib
9. **Evaluation** — Two eval cells exist. Cell 9 reverse-normalizes with whole-DataFrame price stats (biased — see the comment in cell 10). Cell 10 is the correct version: un-scales with the train-split `mean[0]`/`std[0]` (price column) and compares the model against a "copy previous close" baseline (baseline RMSE/MAE + % of test windows beaten)
10. **Save** — Model checkpoint (`torch.save({...})`) is currently **commented out** in cell 11; the `.pt` files on disk are from earlier runs

## Key conventions & patterns

- **Notebook-first development** — The real implementation lives in the notebook; there is no `main.py`. Script-ify from notebook when ready.
- **CoinGecko API key** — Required in `COIN_API` env var (`.env` file). The demo tier is rate-limited to ~10-30 req/min. Cache fetched data rather than hammering the API.
- **Z-score normalization** — Fit on training split only (`mean = train.mean(axis=0)`, `std = train.std(axis=0)`, with `std[std==0] = 1.0` guard). Reverse-normalize for USD metrics using the **train-split** `mean[0]`/`std[0]` (price column), not whole-DataFrame stats.
- **Model checkpoint format** — `torch.save({...})` dict containing: `model_state_dict`, `optimizer_state_dict`, `learning_rate`, `num_epochs`, `mean`, `std`, `price_mean`, `price_std`, `training_loss`, `testing_loss`, `rmse_usd`, `mae_usd`.
- **Hidden state management** — `h0`/`c0` are zeroed on every forward pass (not stateful across batches), which is correct for this use case.
- **Metrics** — MSE in normalized space, RMSE/MAE in USD after reverse normalization, plus a "copy previous close" baseline comparison.
- **Dependencies** — `dotenv`, `matplotlib`, `pandas`, `pycoingecko`, `pytest`, `torch`, `torchmetrics`. The notebook uses raw `requests` + `dotenv`; `pycoingecko` is declared but unused so far.

## Important gotchas

- **There is no `main.py`** — the notebook is the source of truth. Don't try to `uv run python main.py` expecting real output.
- **No git history** — the repo has zero commits. `git log` will fail.
- **No tests, no linter, no CI** — nothing to verify changes against. Test manually via the notebook (pytest is declared but unused).
- **CoinGecko free tier limits** — historical data beyond 365 days requires a paid plan, and granularity is automatic: `days=90` returned daily data in the current run. Don't assume hourly rows.
- **Save cell is commented out** — re-enable cell 11 to regenerate `bitcoin_lstm_model.pt`; the on-disk checkpoints reflect earlier runs.
- **The `pycoingecko` library** is a declared dependency but the notebook uses raw `requests` calls instead. Both approaches work.
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
