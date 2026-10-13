"""Streamlit showcase — LSTM trained from scratch on fresh BTC data.

The model: an LSTM that looks back 30 hours of BTC (price, volume, market
cap) and calls the next hour's price. This app trains it live on *fresh*
data and puts it in action:

    - fetches the last 90 days of hourly BTC data from CoinGecko
    - trains the LSTM from scratch on that data (no checkpoint file)
    - rolls the 30-hour window forward hour by hour, comparing each call
      to what actually happened
    - lets you scrub through time and watch the model work

Run it from the project root:

    uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch import optim
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------
# Constants — training defaults mirror main-notebook.ipynb
# --------------------------------------------------------------------------
SEQ_LEN = 30          # hours of lookback
DEFAULT_HIDDEN_SIZE = 5   # LSTM hidden units
DEFAULT_NUM_LAYERS = 1    # LSTM layers
DEFAULT_EPOCHS = 100
DEFAULT_LR = 0.01
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 42
FEATURES = ["price", "volume", "market_cap"]   # column order matters!
PRICE_COL = 0         # index of the price feature inside FEATURES
PLAY_STEP_SLEEP = 0.12   # seconds between frames while ▶ Play is running

# Dark dashboard palette
BG = "#0B1020"
CARD = "#141A2E"
BORDER = "#232C4D"
ORANGE = "#F7931A"
VIOLET = "#8B5CF6"
GREEN = "#34D399"
TEXT = "#E6EAF2"
MUTED = "#8A94B0"


# --------------------------------------------------------------------------
# Model — same architecture as main-notebook.ipynb
# --------------------------------------------------------------------------
class PriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=DEFAULT_HIDDEN_SIZE,
                 num_stacked_layers=DEFAULT_NUM_LAYERS, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(self.input_size, hidden_size,
                            self.num_stacked_layers, batch_first=batch_first)
        self.final = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.final(out[:, -1, :])
        return out


class TimeSeriesDataset(Dataset):
    """Sliding-window dataset — mirrors main-notebook.ipynb cell 4."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]


def _make_windows(data: np.ndarray, seq_len: int = SEQ_LEN):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len, PRICE_COL])  # next-hour price
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


@st.cache_resource(show_spinner="Training LSTM from scratch on fresh data…")
def train_fresh_model(df: pd.DataFrame, num_epochs: int = DEFAULT_EPOCHS,
                      learning_rate: float = DEFAULT_LR,
                      hidden_size: int = DEFAULT_HIDDEN_SIZE,
                      num_layers: int = DEFAULT_NUM_LAYERS,
                      batch_size: int = DEFAULT_BATCH_SIZE,
                      seed: int = DEFAULT_SEED) -> tuple[PriceLSTM, dict]:
    """Train the price LSTM from scratch on the freshly fetched frame.

    Mirrors main-notebook.ipynb cells 4/5/7/10: 70/15/15 split, z-score
    scaler fit on train only, MSELoss + Adam(weight_decay=1e-4), and USD
    metrics un-scaled with the *training* mean/std (index 0 = price).

    Cached on (data + hyperparams) so slider/playback reruns don't retrain;
    any new fetch or hyperparameter change triggers a real retrain.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    values = df[FEATURES].to_numpy(dtype=np.float32)
    n = len(values)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    mean = values[:train_end].mean(axis=0)
    std = values[:train_end].std(axis=0)
    std[std == 0] = 1.0

    train = (values[:train_end] - mean) / std
    val = (values[train_end:val_end] - mean) / std
    test = (values[val_end:] - mean) / std

    X_train, y_train = _make_windows(train)
    X_val, y_val = _make_windows(val)
    X_test, y_test = _make_windows(test)

    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test),
                             batch_size=batch_size, shuffle=False)

    model = PriceLSTM(input_size=len(FEATURES), hidden_size=hidden_size,
                      num_stacked_layers=num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=1e-4)

    training_loss: list[float] = []
    testing_loss: list[float] = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for feature, label in train_loader:
            optimizer.zero_grad()
            pred = model(feature).squeeze(-1)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        training_loss.append(train_loss / max(len(train_loader), 1))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feature, label in val_loader:
                pred = model(feature).squeeze(-1)
                val_loss += criterion(pred, label).item()
        testing_loss.append(val_loss / max(len(val_loader), 1))

    # Test metrics in real USD (notebook cell 10 fix: use training mean/std).
    model.eval()
    price_mean, price_std = float(mean[PRICE_COL]), float(std[PRICE_COL])
    all_preds, all_true = [], []
    with torch.no_grad():
        for feature, label in test_loader:
            pred = model(feature).squeeze(-1)
            all_preds.append(pred.cpu().numpy() * price_std + price_mean)
            all_true.append(label.cpu().numpy() * price_std + price_mean)
    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_true = np.concatenate(all_true) if all_true else np.array([])
    rmse_usd = float(np.sqrt(np.mean((all_preds - all_true) ** 2))) if len(all_true) else 0.0
    mae_usd = float(np.mean(np.abs(all_preds - all_true))) if len(all_true) else 0.0

    stats = {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "training_loss": training_loss,
        "testing_loss": testing_loss,
        "rmse_usd": rmse_usd,
        "mae_usd": mae_usd,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "batch_size": batch_size,
        "seed": seed,
    }
    model.eval()
    return model, stats


# --------------------------------------------------------------------------
# Data — fresh hourly BTC from CoinGecko (cached to respect rate limits)
# --------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner="Pulling fresh BTC data from CoinGecko…")
def fetch_market_data(days: int = 90) -> pd.DataFrame:
    load_dotenv()
    headers = {"accept": "application/json"}
    api_key = os.getenv("COIN_API")
    if api_key:
        headers["x-cg-demo-api-key"] = api_key
    resp = requests.get(
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
        params={"vs_currency": "usd", "days": days},
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()

    prices = pd.DataFrame(payload["prices"], columns=["timestamp", "price"])
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
    caps = pd.DataFrame(payload["market_caps"], columns=["timestamp", "market_cap"])
    caps["timestamp"] = pd.to_datetime(caps["timestamp"], unit="ms")
    vols = pd.DataFrame(payload["total_volumes"], columns=["timestamp", "volume"])
    vols["timestamp"] = pd.to_datetime(vols["timestamp"], unit="ms")

    df = prices.merge(caps, on="timestamp").merge(vols, on="timestamp")
    return df.sort_values("timestamp").reset_index(drop=True)


# --------------------------------------------------------------------------
# Prediction helpers — identical normalization to training
# --------------------------------------------------------------------------
def _normalize(values: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    std[std == 0] = 1.0
    return (values - mean) / std


def rolling_backtest(df: pd.DataFrame, model: PriceLSTM,
                     stats: dict) -> pd.DataFrame:
    """For every hour, the model sees only the preceding 30 and calls the next.

    Returns a frame aligned on the *predicted* hour: actual vs prediction vs
    the last price the model actually saw.
    """
    values = df[FEATURES].to_numpy(dtype=np.float32)
    norm = _normalize(values, stats)

    X = np.stack([norm[i:i + SEQ_LEN] for i in range(len(norm) - SEQ_LEN)])
    with torch.no_grad():
        preds_norm = model(torch.from_numpy(X)).squeeze(-1).numpy()

    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    preds_usd = preds_norm * std[PRICE_COL] + mean[PRICE_COL]      # un-scale price
    actual_usd = values[SEQ_LEN:, PRICE_COL]                       # the true next price

    # For window i, the last price the model actually saw is the 30th
    # element of that window: values[i + SEQ_LEN - 1].
    last_seen_usd = values[SEQ_LEN - 1:-1, PRICE_COL]
    assert last_seen_usd.shape == preds_usd.shape

    return pd.DataFrame({
        "timestamp": df["timestamp"].iloc[SEQ_LEN:].values,
        "actual": actual_usd,
        "prediction": preds_usd,
        "last_seen": last_seen_usd,
    })


def next_hour_call(df: pd.DataFrame, model: PriceLSTM,
                   stats: dict) -> float:
    """The model's call for the hour after the most recent data point."""
    values = df[FEATURES].to_numpy(dtype=np.float32)
    norm = _normalize(values, stats)
    window = torch.from_numpy(norm[-SEQ_LEN:]).unsqueeze(0)
    with torch.no_grad():
        pred = model(window).item()
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return float(pred * std[PRICE_COL] + mean[PRICE_COL])


# --------------------------------------------------------------------------
# Plotly styling helpers
# --------------------------------------------------------------------------
def _base_layout(height: int, show_legend: bool = True) -> dict:
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=8, r=8, t=40, b=8),
        font=dict(family="Inter, system-ui, sans-serif", color=TEXT, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        showlegend=show_legend,
        hovermode="x unified",
        hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER, font_color=TEXT),
        xaxis=dict(gridcolor="#1C2542", zeroline=False),
        yaxis=dict(gridcolor="#1C2542", zeroline=False,
                   tickprefix="$", tickformat=",.0f"),
    )


def backtest_chart(bt: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bt["timestamp"], y=bt["actual"], name="Actual price",
        line=dict(color=ORANGE, width=2.4),
        hovertemplate="%{x|%b %d, %H:%M}<br>Actual: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=bt["timestamp"], y=bt["prediction"], name="Model's call",
        line=dict(color=VIOLET, width=2.4, dash="dot"),
        hovertemplate="%{x|%b %d, %H:%M}<br>Predicted: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(430))
    fig.update_xaxes(title_text=title, rangeslider_visible=True)
    return fig


def watch_chart(df: pd.DataFrame, bt: pd.DataFrame, i: int) -> go.Figure:
    """The 30-hour window the model saw, plus its call vs reality."""
    window = df.iloc[i:i + SEQ_LEN]
    row = bt.iloc[i]
    target_time = df["timestamp"].iloc[i + SEQ_LEN]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=window["timestamp"], y=window["price"], name="Window the model saw",
        line=dict(color=ORANGE, width=2.6),
        hovertemplate="%{x|%b %d, %H:%M}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[target_time], y=[row["prediction"]], name="Model's call",
        mode="markers+text", text=["call"], textposition="top center",
        textfont=dict(color=VIOLET, size=11),
        marker=dict(color=VIOLET, size=14, symbol="star",
                    line=dict(color="#0B1020", width=1.5)),
        hovertemplate="%{x|%b %d, %H:%M}<br>Called: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[target_time], y=[row["actual"]], name="Reality",
        mode="markers+text", text=["actual"], textposition="bottom center",
        textfont=dict(color=GREEN, size=11),
        marker=dict(color=GREEN, size=13, symbol="diamond",
                    line=dict(color="#0B1020", width=1.5)),
        hovertemplate="%{x|%b %d, %H:%M}<br>Actual: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(360))
    fig.update_yaxes(title_text="BTC price")
    fig.update_xaxes(title_text="30-hour lookback → next hour")
    return fig


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(f"""
    <style>
      .stApp {{ background: {BG}; }}
      .hero {{
        padding: 1.4rem 0 0.2rem;
        border-bottom: 1px solid {BORDER};
        margin-bottom: 1.1rem;
      }}
      .hero .live {{
        display: inline-block; font-size: 0.72rem; letter-spacing: 0.18em;
        color: {GREEN}; border: 1px solid {GREEN}; border-radius: 999px;
        padding: 0.15rem 0.7rem; margin-bottom: 0.6rem;
        text-transform: uppercase; font-weight: 600;
      }}
      .hero h1 {{
        font-size: 2.15rem; font-weight: 800; color: {TEXT}; margin: 0 0 0.35rem;
      }}
      .hero h1 span {{ color: {ORANGE}; }}
      .hero p {{ color: {MUTED}; font-size: 1.02rem; margin: 0; max-width: 720px; }}
      .metric-card {{
        background: linear-gradient(160deg, {CARD}, {BG});
        border: 1px solid {BORDER}; border-radius: 14px;
        padding: 0.9rem 1.1rem; height: 100%;
      }}
      .metric-card .label {{
        font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
        color: {MUTED}; margin-bottom: 0.35rem;
      }}
      .metric-card .value {{ font-size: 1.45rem; font-weight: 700; color: {TEXT}; }}
      .metric-card .sub {{ font-size: 0.8rem; color: {MUTED}; margin-top: 0.25rem; }}
      .section-title {{ font-size: 1.05rem; font-weight: 700; color: {TEXT}; margin: 1.4rem 0 0.3rem; }}
      .section-sub {{ font-size: 0.86rem; color: {MUTED}; margin-bottom: 0.6rem; }}
      .footnote {{ color: {MUTED}; font-size: 0.78rem; margin-top: 1.6rem; }}
      div[data-testid="stSidebar"] {{ background: {BG}; }}
    </style>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, sub: str, accent: str = TEXT) -> str:
    return f"""
    <div class="metric-card">
      <div class="label">{label}</div>
      <div class="value" style="color:{accent}">{value}</div>
      <div class="sub">{sub}</div>
    </div>
    """


def fmt_usd(x: float) -> str:
    return f"${x:,.0f}"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def render_training_controls() -> dict:
    """Training hyperparameters — lives in the sidebar, read before training.

    Changing any widget changes the cache key of `train_fresh_model`, so a
    real retrain happens automatically. Playback/slider reruns with unchanged
    values hit the cache and do NOT retrain.
    """
    with st.sidebar:
        st.markdown("### 🏋️ Training (from scratch)")
        st.caption("No checkpoint file — the LSTM trains live on the data above.")
        num_epochs = st.slider("Epochs", 10, 200, DEFAULT_EPOCHS, step=10)
        learning_rate = st.selectbox(
            "Learning rate",
            [0.001, 0.005, 0.01, 0.02],
            index=2,
        )
        c1, c2 = st.columns(2)
        with c1:
            hidden_size = st.number_input("Hidden", 2, 32, DEFAULT_HIDDEN_SIZE, step=1)
        with c2:
            num_layers = st.number_input("Layers", 1, 3, DEFAULT_NUM_LAYERS, step=1)
        seed = st.number_input("Seed", 0, 9999, DEFAULT_SEED, step=1)
        if st.button("🏋️ Retrain from scratch", use_container_width=True):
            train_fresh_model.clear()
            st.rerun()
    return {
        "num_epochs": int(num_epochs),
        "learning_rate": float(learning_rate),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "seed": int(seed),
    }


def render_sidebar(stats: dict, fetched_at: datetime) -> None:
    with st.sidebar:
        st.markdown("### 🧠 Model card")
        # NOTE: built with a plain f-string below to avoid nested-quote issues.
        st.markdown(
            f"<div class='metric-card' style='margin-bottom:0.6rem;'>"
            f"<div class='label'>Architecture</div>"
            f"<div class='value' style='font-size:1.02rem;'>LSTM → Linear</div>"
            f"<div class='sub'>hidden {stats.get('hidden_size', '?')} · "
            f"{stats.get('num_layers', '?')} layer(s) · 30h lookback · "
            f"fresh-trained, no .pt file</div></div>",
            unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Test RMSE", f"${stats.get('rmse_usd', 0):,.0f}")
        with c2:
            st.metric("Test MAE", f"${stats.get('mae_usd', 0):,.0f}")

        st.markdown(
            f"<div class='sub' style='font-size:0.8rem;color:{MUTED};'>"
            f"trained {stats.get('num_epochs', '?')} epochs · "
            f"lr {stats.get('learning_rate', '?')} · "
            f"seed {stats.get('seed', '?')} · "
            f"features: price, volume, market cap</div>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("### 🔎 How it works")
        st.markdown(
            f"<div style='font-size:0.85rem;color:{MUTED};'>"
            "1. Fresh hourly BTC data is pulled from CoinGecko.<br>"
            "2. The LSTM trains <b>from scratch</b> on that data (70/15/15 "
            "split, z-scored on train only).<br>"
            "3. Each hour, the model reads only the <b>previous 30 hours</b> "
            "of price / volume / market cap.<br>"
            "4. It calls the next hour's price — then we compare with reality.</div>",
            unsafe_allow_html=True,
        )

        st.divider()
        if st.button("↻ Refresh data", use_container_width=True):
            fetch_market_data.clear()
            train_fresh_model.clear()
            st.rerun()
        st.caption(f"Fetched {fetched_at:%b %d, %H:%M} UTC · CoinGecko hourly")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="BTC LSTM · Live Predictor",
                       page_icon="₿", layout="wide")
    inject_css()

    try:
        df = fetch_market_data()
    except requests.RequestException as exc:
        st.error("Couldn't reach CoinGecko. Check your `COIN_API` key in `.env` "
                 "and your connection.")
        st.exception(exc)
        return

    if len(df) < SEQ_LEN + 10:
        st.error("Not enough data fetched to run the model.")
        return

    # Hyperparams must be read BEFORE training so widget changes retrain.
    train_params = render_training_controls()

    with st.spinner("Training LSTM from scratch on fresh data…"):
        model, stats = train_fresh_model(df, **train_params)

    bt = rolling_backtest(df, model, stats)
    fetched_at = datetime.now(timezone.utc)

    # --- Hero -------------------------------------------------------------
    st.markdown(
        f"""
        <div class="hero">
          <span class="live">● Live · trained from scratch on this data</span>
          <h1>₿ Watch an LSTM <span>call BTC</span>, hour by hour</h1>
          <p>The model trains from scratch on fresh price, volume and market cap —
          and never sees the future. Every point on the violet line was made
          from only the <b>30 hours before it</b>.</p>
        </div>
        """, unsafe_allow_html=True)

    # --- KPI row ----------------------------------------------------------
    last_price = float(df["price"].iloc[-1])
    prev_price = float(df["price"].iloc[-25])          # ~24h earlier
    next_call = next_hour_call(df, model, stats)
    bt_rmse = float(np.sqrt(np.mean((bt["actual"] - bt["prediction"]) ** 2)))
    bt_mae = float(np.mean(np.abs(bt["actual"] - bt["prediction"])))

    cols = st.columns(4)
    cols[0].markdown(
        metric_card("BTC price · now", fmt_usd(last_price),
                    f"{(last_price / prev_price - 1) * 100:+.2f}% · 24h",
                    ORANGE), unsafe_allow_html=True)
    cols[1].markdown(
        metric_card("Next-hour call", fmt_usd(next_call),
                    f"{(next_call / last_price - 1) * 100:+.2f}% vs latest price",
                    VIOLET), unsafe_allow_html=True)
    cols[2].markdown(
        metric_card("Rolling RMSE", fmt_usd(bt_rmse),
                    "mean |call − reality|² over the backtest",
                    GREEN), unsafe_allow_html=True)
    cols[3].markdown(
        metric_card("Rolling MAE", fmt_usd(bt_mae),
                    "typical miss vs the actual hour",
                    GREEN), unsafe_allow_html=True)

    # --- Training curves (this run, from scratch) -------------------------
    st.markdown('<div class="section-title">Training — from scratch, just now</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-sub">MSE loss per epoch on the fresh fetch. '
                'Change epochs / lr / hidden size in the sidebar to retrain.</div>',
                unsafe_allow_html=True)
    st.line_chart(
        {"train": stats["training_loss"], "val": stats["testing_loss"]},
    )

    # --- Rolling backtest chart ------------------------------------------
    st.markdown('<div class="section-title">Rolling backtest — every call vs reality</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-sub">No peeking: each violet call was made '
                'with only the 30 hours of data that came before it.</div>',
                unsafe_allow_html=True)

    days = st.radio("Window", ["7 days", "30 days", "All 90 days"],
                    horizontal=True, index=1)
    hours = {"7 days": 7 * 24, "30 days": 30 * 24, "All 90 days": len(bt)}[days]
    view = bt.iloc[-hours:].reset_index(drop=True)

    st.plotly_chart(backtest_chart(view, f"last {days.lower()} · hourly"),
                    use_container_width=True,
                    config={"displayModeBar": False})

    # --- Watch it work ----------------------------------------------------
    st.markdown('<div class="section-title">Watch it work, hour by hour</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Scrub through time (or hit play). The '
                'orange trace is exactly what the model saw; the star is its call '
                'for the next hour; the diamond is what actually happened.</div>',
                unsafe_allow_html=True)

    scrub = bt.tail(7 * 24).reset_index(drop=True)       # last 7 days
    n_scrub = len(scrub)
    offset = len(bt) - n_scrub                            # index inside bt

    # Playback state lives in session_state. IMPORTANT: `scrub` is a widget
    # key (the slider owns it), so it may only be WRITTEN before the slider
    # is instantiated in a given run — never after. That's why the advance
    # block below sits *above* the st.slider call, and why the buttons use
    # on_click callbacks (which run before the script body re-executes).
    st.session_state.setdefault("playing", False)
    st.session_state.setdefault("scrub", n_scrub - 1)

    def _play() -> None:
        st.session_state.playing = True
        if st.session_state.scrub >= n_scrub - 1:
            st.session_state.scrub = -1   # restart just before the first frame

    def _pause() -> None:
        st.session_state.playing = False

    c1, c2 = st.columns([1, 1])
    with c1:
        st.button("▶ Play", use_container_width=True, type="primary",
                  on_click=_play)
    with c2:
        st.button("⏹ Pause", use_container_width=True, on_click=_pause)

    # Advance the frame one step (before the slider exists → legal write).
    # When the frame reaches the end we stop and render the final state.
    if st.session_state.get("playing", False):
        nxt = st.session_state.scrub + 1
        if nxt >= n_scrub:
            st.session_state.playing = False
        else:
            st.session_state.scrub = nxt

    idx = st.slider("Step through time", 0, n_scrub - 1, key="scrub")

    row = scrub.iloc[idx]
    err = row["actual"] - row["prediction"]
    st.plotly_chart(watch_chart(df, bt, offset + idx),
                    use_container_width=True,
                    config={"displayModeBar": False})
    st.markdown(
        f"<div style='font-size:0.92rem;color:{TEXT};'>"
        f"At <b>{row['timestamp']:%b %d, %H:%M} UTC</b> the model had seen "
        f"{SEQ_LEN} hours of BTC ending at <b>{fmt_usd(row['last_seen'])}</b> "
        f"and called <b style='color:{VIOLET}'>{fmt_usd(row['prediction'])}</b>. "
        f"Reality: <b style='color:{GREEN}'>{fmt_usd(row['actual'])}</b> · "
        f"{'under' if err < 0 else 'over'}called by "
        f"<b>{fmt_usd(abs(err))}</b>.</div>",
        unsafe_allow_html=True)

    # Keep the animation going: the chart above is already drawn this run, so
    # the browser shows each frame; then we sleep a beat and re-run for the
    # next frame. Pause stops this by flipping `playing` to False.
    if st.session_state.get("playing", False):
        time.sleep(PLAY_STEP_SLEEP)
        st.rerun()

    st.markdown(
        f"<div class='footnote'>Fresh-trained {SEQ_LEN}-hour price LSTM "
        f"(no checkpoint file · {stats.get('num_epochs')} epochs, seed "
        f"{stats.get('seed')}). Metrics: {fmt_usd(bt_rmse)} "
        f"RMSE / {fmt_usd(bt_mae)} MAE across the full {len(bt):,}-hour backtest.</div>",
        unsafe_allow_html=True)

    render_sidebar(stats, fetched_at)


if __name__ == "__main__":
    main()
