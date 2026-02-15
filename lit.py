# app.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="VolMike Minimal — yfinance (Table)",
    page_icon="📦",
    layout="centered",
)

NY_TZ = "America/New_York"

INTERVAL_MAP = {
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "60m": "60m",
}

RTH_START = time(9, 30)
RTH_END = time(16, 0)  # treated as exclusive end


# -----------------------------
# MODELS
# -----------------------------
@dataclass(frozen=True)
class RowOut:
    time_ny: str
    price: float
    mike: Optional[float]
    rvol: Optional[float]


# -----------------------------
# HELPERS
# -----------------------------
_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,12}$")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(x: object) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _validate_ticker(t: str) -> str:
    t2 = (t or "").upper().strip()
    if not t2 or not _TICKER_RE.match(t2):
        raise ValueError("Invalid ticker. Use letters/numbers plus . or - (no spaces).")
    return t2


def compute_mike(price: float, prev_close: float) -> float:
    if prev_close <= 0 or not np.isfinite(prev_close):
        raise ValueError("prev_close must be a finite number > 0")
    return (price / prev_close) * 10_000.0


def compute_rvol_series(volume: pd.Series, window: int = 20) -> pd.Series:
    w = int(max(1, window))
    v = pd.to_numeric(volume, errors="coerce").astype(float)
    base = v.rolling(window=w, min_periods=max(1, w // 2)).mean()
    r = v / base
    return r.replace([np.inf, -np.inf], np.nan)


def _ensure_utc(ts: pd.Series) -> pd.Series:
    s = pd.to_datetime(ts, errors="coerce")
    # If tz-naive, treat as UTC. If tz-aware, convert to UTC.
    if getattr(s.dt, "tz", None) is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")


def normalize_yf_intraday(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance intraday output into columns:
      time (UTC tz-aware), Close, Volume
    """
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    # Flatten MultiIndex columns if present
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
    d["time"] = _ensure_utc(d["time"])
    d["Close"] = pd.to_numeric(d.get("Close"), errors="coerce")
    d["Volume"] = pd.to_numeric(d.get("Volume"), errors="coerce")

    d = d.dropna(subset=["time", "Close"]).sort_values("time").reset_index(drop=True)
    return d


def fetch_intraday(ticker: str, start_d: date, end_d: date, interval: str) -> pd.DataFrame:
    """
    Fetch intraday bars from start_d through end_d (inclusive).
    yfinance end is typically exclusive -> we add +1 day.
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


def fetch_daily_with_backfill(ticker: str, start_d: date, end_d: date) -> pd.DataFrame:
    """
    Fetch daily bars with backfill so we can always find previous trading-day close.
    """
    backfill_start = start_d - timedelta(days=45)
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


def build_prev_close_lookup(daily: pd.DataFrame) -> tuple[list[date], list[float]]:
    """
    Build sorted trading dates and their closes.
    We treat daily index labels as the trading date.
    """
    if daily is None or daily.empty or "Close" not in daily.columns:
        raise ValueError("Daily data missing/invalid (needed for prev close).")

    idx = pd.to_datetime(daily.index, errors="coerce")
    if idx.isna().all():
        raise ValueError("Daily index is not parseable.")

    d = pd.DataFrame(
        {
            "d": idx.date,
            "close": pd.to_numeric(daily["Close"], errors="coerce").astype(float),
        }
    ).dropna(subset=["d", "close"])

    d = d.drop_duplicates(subset=["d"]).sort_values("d")
    dates = d["d"].tolist()
    closes = d["close"].tolist()
    if len(dates) < 2:
        raise ValueError("Not enough daily history to compute prev close.")
    return dates, closes


def prev_close_for_day(trading_dates: list[date], closes: list[float], day: date) -> Optional[float]:
    """
    For intraday trading date = day, return previous trading day's close.
    If day isn't in daily list (e.g., yfinance didn't include it yet), still return last close before day.
    """
    # Find insertion point for 'day' in trading_dates (sorted)
    pos = int(np.searchsorted(np.array(trading_dates, dtype="O"), day, side="left"))

    # If day is exactly a trading date in the list, prev is pos-1
    if pos < len(trading_dates) and trading_dates[pos] == day:
        if pos - 1 >= 0:
            return float(closes[pos - 1])
        return None

    # Otherwise, day isn't present; prev is the last date strictly before day -> pos-1
    if pos - 1 >= 0:
        return float(closes[pos - 1])
    return None


def filter_rth_ny(intraday: pd.DataFrame) -> pd.DataFrame:
    """
    Convert intraday UTC times to NY times and filter to RTH 09:30 <= t < 16:00 (NY).
    """
    if intraday is None or intraday.empty:
        return pd.DataFrame()

    d = intraday.copy()
    d["time_ny"] = d["time"].dt.tz_convert(NY_TZ)
    t = d["time_ny"].dt.time
    mask = (t >= RTH_START) & (t < RTH_END)
    d = d.loc[mask].sort_values("time_ny").reset_index(drop=True)
    return d


def compute_table(
    ticker: str,
    intraday_rth: pd.DataFrame,
    daily: pd.DataFrame,
    rvol_window: int = 20,
) -> pd.DataFrame:
    if intraday_rth is None or intraday_rth.empty:
        raise ValueError("No RTH intraday bars returned. Try a wider date range or different interval.")

    trading_dates, closes = build_prev_close_lookup(daily)

    d = intraday_rth.copy()
    d["day_ny"] = d["time_ny"].dt.date

    # prev close per NY trading day
    d["prev_close"] = d["day_ny"].map(lambda day: prev_close_for_day(trading_dates, closes, day))

    # mike
    def _mike_row(row: pd.Series) -> Optional[float]:
        px = _to_float(row.get("Close"))
        pc = _to_float(row.get("prev_close"))
        if px is None or pc is None or pc <= 0:
            return None
        return float((px / pc) * 10_000.0)

    d["mike"] = d.apply(_mike_row, axis=1)

    # rvol
    rvol_s = compute_rvol_series(d["Volume"], window=rvol_window)
    d["rvol"] = pd.to_numeric(rvol_s, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)

    out = pd.DataFrame(
        {
            "time": d["time_ny"].dt.strftime("%Y-%m-%d %H:%M"),
            "price": pd.to_numeric(d["Close"], errors="coerce").astype(float),
            "mike": pd.to_numeric(d["mike"], errors="coerce").astype(float),
            "rvol": pd.to_numeric(d["rvol"], errors="coerce").astype(float),
        }
    )

    # Clean
    out = out.dropna(subset=["time", "price"]).reset_index(drop=True)
    return out


# -----------------------------
# UI
# -----------------------------
st.title("📦 VolMike Minimal (yfinance) — RTH Table")
st.caption("RTH table (New York time): time, price, Mike (vs prev close), RVOL (20-bar). JSON download only.")

ticker_in = st.text_input("ticker", value="SPY")

col1, col2 = st.columns(2)
with col1:
    start_d = st.date_input("date start", value=date.today() - timedelta(days=5))
with col2:
    end_d = st.date_input("date end", value=date.today())

interval_choice = st.selectbox("minutes", ["2m", "5m", "15m", "30m", "60m"], index=1)

run = st.button("run", type="primary")

if run:
    try:
        ticker = _validate_ticker(ticker_in)
        if start_d > end_d:
            raise ValueError("Start date must be <= end date.")
        interval = INTERVAL_MAP[interval_choice]
    except Exception as e:
        st.error(str(e))
        st.stop()

    with st.spinner("Fetching from yfinance…"):
        intraday = fetch_intraday(ticker, start_d, end_d, interval)
        daily = fetch_daily_with_backfill(ticker, start_d, end_d)

    if intraday.empty:
        st.error("No intraday data returned. Try a wider date range or a different ticker/interval.")
        st.stop()

    if daily.empty:
        st.error("No daily data returned (needed for prev close). Try again.")
        st.stop()

    intraday_rth = filter_rth_ny(intraday)

    try:
        table = compute_table(ticker, intraday_rth, daily, rvol_window=20)
    except Exception as e:
        st.error(f"Compute error: {e}")
        st.stop()

    if table.empty:
        st.error("No RTH rows found for that range. Try different dates/interval.")
        st.stop()

    st.subheader("RTH table")
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )

    # JSON download (NOT visible)
    payload = {
        "ticker": ticker,
        "start_date": str(start_d),
        "end_date": str(end_d),
        "interval": interval_choice,
        "timezone": NY_TZ,
        "rvol_window": 20,
        "computed_at_utc": _now_utc_iso(),
        "rows": table.to_dict(orient="records"),
    }
    json_bytes = json.dumps(payload, indent=2).encode("utf-8")

    st.download_button(
        "download json",
        data=json_bytes,
        file_name=f"{ticker}_mike_rvol_table_{start_d}_{end_d}_{interval_choice}.json",
        mime="application/json",
    )
