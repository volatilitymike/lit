# app.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="VolMike Minimal — yfinance",
    page_icon="📦",
    layout="centered",
)

INTERVAL_MAP = {
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "60m": "60m",
}


# -----------------------------
# MODELS
# -----------------------------
@dataclass(frozen=True)
class CoreResult:
    ticker: str
    time_iso: str
    price: float
    mike: float
    rvol: Optional[float]


# -----------------------------
# HELPERS
# -----------------------------
def _is_finite(x: object) -> bool:
    try:
        v = float(x)
        return bool(np.isfinite(v))
    except Exception:
        return False


def _to_float(x: object) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_mike(price: float, prev_close: float) -> float:
    if prev_close <= 0:
        raise ValueError("prev_close must be > 0")
    return (price / prev_close) * 10_000.0


def compute_rvol_series(volume: pd.Series, window: int) -> pd.Series:
    w = int(max(1, window))
    v = pd.to_numeric(volume, errors="coerce").astype(float)
    base = v.rolling(window=w, min_periods=max(1, w // 2)).mean()
    r = v / base
    return r.replace([np.inf, -np.inf], np.nan)


def normalize_yf_intraday(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance may return either:
      - MultiIndex index with DatetimeIndex
      - Or columns with "Datetime"/"Date" after reset_index
    Normalize to columns: time, Close, Volume
    """
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    # If yf returns MultiIndex columns (rare in single ticker), flatten best-effort
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]

    d = d.reset_index()

    time_col = None
    for c in ("Datetime", "Date", "index"):
        if c in d.columns:
            time_col = c
            break
    if time_col is None:
        return pd.DataFrame()

    d = d.rename(columns={time_col: "time"})
    d["time"] = pd.to_datetime(d["time"], errors="coerce", utc=True)
    d["Close"] = pd.to_numeric(d.get("Close"), errors="coerce")
    d["Volume"] = pd.to_numeric(d.get("Volume"), errors="coerce")

    d = d.dropna(subset=["time", "Close"]).sort_values("time").reset_index(drop=True)
    return d


def fetch_intraday(ticker: str, start_d: date, end_d: date, interval: str) -> pd.DataFrame:
    """
    Fetch intraday bars from start_d through end_d (inclusive).
    yfinance 'end' is typically exclusive -> we add +1 day.
    """
    start = datetime.combine(start_d, datetime.min.time())
    end_exclusive = datetime.combine(end_d + timedelta(days=1), datetime.min.time())

    raw = yf.download(
        tickers=ticker,
        start=start,
        end=end_exclusive,
        interval=interval,
        auto_adjust=False,
        prepost=False,
        progress=False,
        threads=False,
    )
    return normalize_yf_intraday(raw)


def fetch_daily(ticker: str, start_d: date, end_d: date) -> pd.DataFrame:
    """
    Fetch daily bars with enough backfill to find a prior close for the last intraday bar.
    """
    backfill_start = start_d - timedelta(days=30)
    start = datetime.combine(backfill_start, datetime.min.time())
    end_exclusive = datetime.combine(end_d + timedelta(days=1), datetime.min.time())

    raw = yf.download(
        tickers=ticker,
        start=start,
        end=end_exclusive,
        interval="1d",
        auto_adjust=False,
        prepost=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    d = raw.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
    d = d.dropna(subset=["Close"]).copy()
    return d


def pick_prev_close_for_time(daily: pd.DataFrame, ts_utc: pd.Timestamp) -> float:
    """
    Choose previous trading day close relative to the intraday timestamp's calendar date (UTC).
    Works even if the last daily row is the same day (partial) or data is delayed.
    """
    if daily is None or daily.empty or "Close" not in daily.columns:
        raise ValueError("Daily data missing/invalid for prev_close.")

    idx = pd.to_datetime(daily.index, errors="coerce")
    if idx.isna().all():
        raise ValueError("Daily index is not parseable for prev_close.")

    last_day = ts_utc.date()
    day_dates = pd.Series([pd.Timestamp(x).date() for x in idx], index=daily.index)

    # All daily rows strictly before the intraday bar date
    mask = day_dates < last_day
    prior = daily.loc[mask]
    if not prior.empty:
        return float(prior["Close"].iloc[-1])

    # Fallback: if no prior day exists in range, use last available close
    return float(daily["Close"].iloc[-1])


def compute_core_result(
    ticker: str,
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
    rvol_window: int = 20,
) -> CoreResult:
    if intraday is None or intraday.empty:
        raise ValueError("No intraday bars returned. Check ticker/date range/interval.")

    if "Close" not in intraday.columns or "Volume" not in intraday.columns:
        raise ValueError("Intraday data missing Close/Volume columns.")

    last = intraday.iloc[-1]
    px = float(last["Close"])
    ts = pd.Timestamp(last["time"])
    prev_close = pick_prev_close_for_time(daily, ts)

    mike = compute_mike(px, prev_close)

    rvol_s = compute_rvol_series(intraday["Volume"], rvol_window)
    rvol = _to_float(rvol_s.iloc[-1])

    return CoreResult(
        ticker=ticker.upper().strip(),
        time_iso=ts.to_pydatetime().isoformat(),
        price=px,
        mike=float(mike),
        rvol=rvol,
    )


# -----------------------------
# UI
# -----------------------------
st.title("📦 VolMike Minimal (yfinance)")
st.caption("Inputs only: ticker, start date, end date, interval, run. Output: price, Mike, RVOL + JSON download.")

ticker = st.text_input("ticker", value="SPY").upper().strip()

col1, col2 = st.columns(2)
with col1:
    start_d = st.date_input("date start", value=date.today() - timedelta(days=5))
with col2:
    end_d = st.date_input("date end", value=date.today())

interval_choice = st.selectbox("minutes", ["2m", "5m", "15m", "30m", "60m"], index=1)

run = st.button("run", type="primary")

if run:
    if not ticker or any(ch.isspace() for ch in ticker):
        st.error("Invalid ticker.")
        st.stop()

    if start_d > end_d:
        st.error("Start date must be <= end date.")
        st.stop()

    interval = INTERVAL_MAP[interval_choice]

    with st.spinner("Fetching from yfinance…"):
        intraday = fetch_intraday(ticker, start_d, end_d, interval)
        daily = fetch_daily(ticker, start_d, end_d)

    if intraday.empty:
        st.error("No intraday data returned. Try a wider date range or a different ticker/interval.")
        st.stop()

    if daily.empty:
        st.error("No daily data returned (needed for prev close). Try again.")
        st.stop()

    try:
        result = compute_core_result(ticker, intraday, daily, rvol_window=20)
    except Exception as e:
        st.error(f"Compute error: {e}")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("price", f"{result.price:.4f}")
    c2.metric("mike", f"{result.mike:.2f}")
    c3.metric("rvol", "NA" if result.rvol is None else f"{result.rvol:.2f}")

    payload = {
        "ticker": result.ticker,
        "time": result.time_iso,
        "price": result.price,
        "mike": result.mike,
        "rvol": result.rvol,
        "computed_at_utc": _now_utc_iso(),
    }
    json_str = json.dumps(payload, indent=2)

    st.code(json_str, language="json")
    st.download_button(
        "download json",
        data=json_str.encode("utf-8"),
        file_name=f"{result.ticker}_mike_rvol.json",
        mime="application/json",
    )
