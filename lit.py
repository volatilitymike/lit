# app.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from zoneinfo import ZoneInfo


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="VolMike Minimal — yfinance (Mike + RVOL + JSON)",
    page_icon="📦",
    layout="centered",
)

NY = ZoneInfo("America/New_York")


# -----------------------------
# DATA MODEL
# -----------------------------
@dataclass(frozen=True)
class OutputPayload:
    ticker: str
    interval: str
    period: str
    time: str
    price: float
    prev_close: float
    mike: float
    rvol: Optional[float]
    rvol_window: int
    computed_at_utc: str


# -----------------------------
# HELPERS
# -----------------------------
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_float(x) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def compute_mike(price: float, prev_close: float) -> float:
    if prev_close <= 0:
        raise ValueError("prev_close must be > 0")
    return (price / prev_close) * 10_000.0


def compute_rvol_series(volume: pd.Series, window: int) -> pd.Series:
    w = int(max(1, window))
    v = pd.to_numeric(volume, errors="coerce").astype(float)
    base = v.rolling(window=w, min_periods=max(1, w // 2)).mean()
    rvol = v / base
    return rvol.replace([np.inf, -np.inf], np.nan)


def _pick_prev_close_from_daily(daily: pd.DataFrame) -> float:
    """
    Pick *previous trading day* close.
    If daily includes today's partial bar, use the prior row.
    """
    if daily is None or daily.empty or "Close" not in daily.columns:
        raise ValueError("Daily data missing or invalid.")

    d = daily.copy()
    d = d.dropna(subset=["Close"])
    if d.empty:
        raise ValueError("Daily Close data is empty after cleaning.")

    # daily index is typically date-like; normalize to NY date
    idx = pd.to_datetime(d.index, errors="coerce")
    if idx.isna().all():
        # fallback: just use last available close
        return float(d["Close"].iloc[-1])

    # Compare last row date to "today" NY
    today_ny = datetime.now(NY).date()
    last_date = pd.Timestamp(idx[-1]).date()

    if last_date == today_ny and len(d) >= 2:
        return float(d["Close"].iloc[-2])

    return float(d["Close"].iloc[-1])


@st.cache_data(ttl=30, show_spinner=False)
def fetch_intraday(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        prepost=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    # yfinance uses either "Datetime" or "Date"
    time_col = "Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else None)
    if time_col is None:
        return pd.DataFrame()

    df = df.rename(columns={time_col: "time"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df.dropna(subset=["time", "Close"]).sort_values("time").reset_index(drop=True)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_daily(ticker: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        period="15d",
        interval="1d",
        auto_adjust=False,
        prepost=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    return df


def compute_payload(
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
    ticker: str,
    period: str,
    interval: str,
    rvol_window: int,
) -> OutputPayload:
    if intraday is None or intraday.empty:
        raise ValueError("No intraday bars returned (ticker/period/interval may be invalid).")

    if "Close" not in intraday.columns or "Volume" not in intraday.columns:
        raise ValueError("Intraday data missing Close/Volume columns.")

    last = intraday.iloc[-1]
    price = float(last["Close"])
    t_iso = pd.Timestamp(last["time"]).to_pydatetime().isoformat()

    prev_close = _pick_prev_close_from_daily(daily)
    mike = compute_mike(price=price, prev_close=prev_close)

    rvol_s = compute_rvol_series(intraday["Volume"], rvol_window)
    rvol = to_float(rvol_s.iloc[-1])

    return OutputPayload(
        ticker=ticker.upper().strip(),
        interval=interval,
        period=period,
        time=t_iso,
        price=price,
        prev_close=float(prev_close),
        mike=float(mike),
        rvol=rvol,
        rvol_window=int(rvol_window),
        computed_at_utc=now_utc_iso(),
    )


def enforce_yf_limits(period: str, interval: str) -> Tuple[str, Optional[str]]:
    """
    yfinance intraday constraints (common):
      - 1m/2m data only works for short periods (often <= 7d)
    We won't overcomplicate; we gently coerce period when needed.
    """
    if interval in {"1m", "2m"} and period not in {"1d", "5d", "7d"}:
        return "5d", "Coerced period to 5d because 1m/2m intraday usually won't load for longer periods."
    return period, None


# -----------------------------
# UI
# -----------------------------
st.title("📦 VolMike Minimal (yfinance)")
st.caption("Only: Mike (F), RVOL, and JSON download — fetched live from yfinance.")

with st.sidebar:
    st.header("Inputs")

    ticker = st.text_input("Ticker", value="SPY").upper().strip()
    interval = st.selectbox("Interval", ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d"], index=2)
    period = st.selectbox("Period", ["1d", "5d", "7d", "1mo", "3mo", "6mo"], index=1)

    rvol_window = st.number_input("RVOL rolling window (bars)", min_value=1, max_value=500, value=20, step=1)

    st.divider()
    st.caption("Note: yfinance data can be delayed and sometimes returns empty during outages/market closures.")


# Guardrails
if not ticker or any(ch.isspace() for ch in ticker):
    st.error("Enter a valid ticker (no spaces).")
    st.stop()

period_fixed, warn = enforce_yf_limits(period, interval)
if warn:
    st.warning(warn)
period = period_fixed

# Fetch
with st.spinner("Fetching data…"):
    intraday = fetch_intraday(ticker=ticker, period=period, interval=interval)
    daily = fetch_daily(ticker=ticker)

if intraday.empty:
    st.error("No intraday data returned. Try a different period/interval, or verify the ticker.")
    st.stop()

if daily.empty:
    st.error("No daily data returned (needed for prev close). Try again or verify the ticker.")
    st.stop()

# Compute
try:
    payload = compute_payload(
        intraday=intraday,
        daily=daily,
        ticker=ticker,
        period=period,
        interval=interval,
        rvol_window=int(rvol_window),
    )
except Exception as e:
    st.error(f"Computation error: {e}")
    st.stop()

# Display minimal metrics
c1, c2, c3 = st.columns(3)
c1.metric("Price", f"{payload.price:.4f}")
c2.metric("Mike (F)", f"{payload.mike:.2f}")
c3.metric("RVOL", "NA" if payload.rvol is None else f"{payload.rvol:.2f}")

# JSON (only what you care about, plus tiny metadata)
core_json = {
    "ticker": payload.ticker,
    "time": payload.time,
    "price": payload.price,
    "mike": payload.mike,
    "rvol": payload.rvol,
    # (optional but useful)
    "interval": payload.interval,
    "period": payload.period,
    "computed_at_utc": payload.computed_at_utc,
}
json_str = json.dumps(core_json, indent=2)

st.subheader("JSON")
st.code(json_str, language="json")

st.download_button(
    label="⬇️ Download JSON",
    data=json_str.encode("utf-8"),
    file_name=f"{payload.ticker}_volmike_minimal.json",
    mime="application/json",
)

with st.expander("Data preview (last 60 bars)"):
    st.dataframe(intraday.tail(60), use_container_width=True)
