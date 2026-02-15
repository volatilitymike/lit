"""
VolMike - Financial Market Microstructure Analysis Tool

A production-ready Streamlit application for analyzing intraday movements
using Mike (basis points from previous close, displacement-style) and RVOL.

Mike (bps displacement) = ((Close - PrevClose) / PrevClose) * 10,000
- Prev close baseline is 0 bps
- +0.10% move => +10 bps
- -0.10% move => -10 bps
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="VolMike - Market Microstructure Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("volmike")

NY_TZ = "America/New_York"
RTH_START = time(9, 30)
RTH_END = time(16, 0)

INTERVAL_MAP = {
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "60m": "60m",
}

TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-]{1,12}$")

MIKE_LINE_COLOR = "dodgerblue"
MIKE_LINE_WIDTH = 2


# -----------------------------------------------------------------------------
# DATA MODELS
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class AnalysisParams:
    ticker: str
    start_date: date
    end_date: date
    interval: str
    rvol_window: int = 20


class ValidationError(Exception):
    pass


# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------

def get_utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None

def validate_ticker(ticker: str) -> str:
    t = (ticker or "").upper().strip()
    if not t:
        raise ValidationError("Ticker cannot be empty.")
    if not TICKER_PATTERN.match(t):
        raise ValidationError("Invalid ticker. Use only letters/numbers/dot/hyphen (no spaces).")
    return t

def validate_date_range(start: date, end: date) -> None:
    if start > end:
        raise ValidationError(f"Start date ({start}) cannot be after end date ({end}).")
    if end > date.today():
        raise ValidationError(f"End date ({end}) cannot be in the future.")


def ensure_utc_datetime(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    if dt.dt.tz is None:
        return dt.dt.tz_localize("UTC")
    return dt.dt.tz_convert("UTC")


def normalize_intraday_yf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return columns: time(UTC tz-aware), Close(float), Volume(float)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]

    out = out.reset_index()

    time_col = None
    for cand in ("Datetime", "Date", "index"):
        if cand in out.columns:
            time_col = cand
            break
    if time_col is None:
        # yfinance usually puts time in the index; after reset_index it's the first col
        time_col = out.columns[0]

    out = out.rename(columns={time_col: "time"})
    out["time"] = ensure_utc_datetime(out["time"])
    out["Close"] = pd.to_numeric(out.get("Close"), errors="coerce")
    out["Volume"] = pd.to_numeric(out.get("Volume"), errors="coerce")

    out = out.dropna(subset=["time", "Close"]).sort_values("time").reset_index(drop=True)
    return out[["time", "Close", "Volume"]]


def normalize_daily_yf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a clean daily lookup with columns: date (python date), close(float)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]

    idx = pd.to_datetime(out.index, errors="coerce")
    out = out.copy()
    out["date"] = idx.date
    out["close"] = pd.to_numeric(out.get("Close"), errors="coerce").astype(float)
    out = out.dropna(subset=["date", "close"]).drop_duplicates(subset=["date"]).sort_values("date")
    return out[["date", "close"]].reset_index(drop=True)


def filter_rth(intraday: pd.DataFrame) -> pd.DataFrame:
    """
    Adds time_ny (tz-aware), then filters to 09:30 <= time < 16:00 ET.
    """
    if intraday is None or intraday.empty:
        return pd.DataFrame()

    d = intraday.copy()
    d["time_ny"] = d["time"].dt.tz_convert(NY_TZ)
    tonly = d["time_ny"].dt.time
    mask = (tonly >= RTH_START) & (tonly < RTH_END)
    d = d.loc[mask].sort_values("time_ny").reset_index(drop=True)
    return d


def calc_mike_bps(close: pd.Series, prev_close: pd.Series) -> pd.Series:
    """
    Displacement-style Mike in basis points (0 at prev close).
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    prev_close = pd.to_numeric(prev_close, errors="coerce").astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mike = ((close - prev_close) / prev_close) * 10_000.0
    mike = pd.Series(mike).replace([np.inf, -np.inf], np.nan)
    return mike


def calc_rvol(volume: pd.Series, window: int) -> pd.Series:
    window = max(1, int(window))
    minp = max(1, window // 2)
    v = pd.to_numeric(volume, errors="coerce").astype(float)
    ma = v.rolling(window=window, min_periods=minp).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = v / ma
    r = pd.Series(r).replace([np.inf, -np.inf], np.nan)
    return r


# -----------------------------------------------------------------------------
# FETCHING
# -----------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def fetch_intraday(ticker: str, start_d: date, end_d: date, interval: str) -> pd.DataFrame:
    start_dt = datetime.combine(start_d, datetime.min.time())
    end_dt = datetime.combine(end_d + timedelta(days=1), datetime.min.time())

    raw = yf.download(
        tickers=ticker,
        start=start_dt,
        end=end_dt,
        interval=interval,
        auto_adjust=False,
        prepost=False,
        progress=False,
        threads=False,
    )
    return normalize_intraday_yf(raw)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_daily_backfill(ticker: str, start_d: date, end_d: date, backfill_days: int = 60) -> pd.DataFrame:
    backfill_start = start_d - timedelta(days=backfill_days)
    start_dt = datetime.combine(backfill_start, datetime.min.time())
    end_dt = datetime.combine(end_d + timedelta(days=1), datetime.min.time())

    raw = yf.download(
        tickers=ticker,
        start=start_dt,
        end=end_dt,
        interval="1d",
        auto_adjust=False,
        prepost=False,
        progress=False,
        threads=False,
    )
    return normalize_daily_yf(raw)


# -----------------------------------------------------------------------------
# PROCESSING
# -----------------------------------------------------------------------------

def build_prev_close_map(daily_lookup: pd.DataFrame) -> dict[date, float]:
    """
    daily_lookup columns: date, close (sorted)
    returns map: date -> prev_close
    """
    if daily_lookup is None or daily_lookup.empty:
        raise ValueError("No daily data for previous close calculation.")

    dl = daily_lookup.sort_values("date").reset_index(drop=True)
    dl["prev_close"] = dl["close"].shift(1)

    # Keep only rows where we actually have a previous close
    dl = dl.dropna(subset=["prev_close"])
    return dict(zip(dl["date"].tolist(), dl["prev_close"].astype(float).tolist()))


def compute_table(params: AnalysisParams, intraday_rth: pd.DataFrame, daily_lookup: pd.DataFrame) -> pd.DataFrame:
    if intraday_rth is None or intraday_rth.empty:
        raise ValueError("No RTH intraday data. Try different dates/interval.")

    prev_map = build_prev_close_map(daily_lookup)

    d = intraday_rth.copy()
    d["trading_date"] = d["time_ny"].dt.date
    d["prev_close"] = d["trading_date"].map(prev_map)

    # Mike (bps displacement)
    d["mike"] = calc_mike_bps(d["Close"], d["prev_close"])

    # Old compatibility columns
    d["F_numeric"] = d["mike"]
    d["F%"] = np.where(d["F_numeric"].notna(), d["F_numeric"].round(0).astype("Int64").astype(str) + "%", "N/A")

    # RVOL per day
    d["rvol"] = d.groupby("trading_date")["Volume"].transform(lambda s: calc_rvol(s, params.rvol_window))

    # Output (keep time as datetime for proper Plotly time axis)
    out = pd.DataFrame(
        {
            "time": d["time_ny"],  # tz-aware datetime (best for chart)
            "price": pd.to_numeric(d["Close"], errors="coerce").astype(float),
            "prev_close": pd.to_numeric(d["prev_close"], errors="coerce").astype(float),
            "mike": pd.to_numeric(d["mike"], errors="coerce").astype(float),
            "F_numeric": pd.to_numeric(d["F_numeric"], errors="coerce").astype(float),
            "F%": d["F%"].astype(str),
            "rvol": pd.to_numeric(d["rvol"], errors="coerce").astype(float),
        }
    )

    out = out.dropna(subset=["time", "price"]).reset_index(drop=True)
    return out


def calc_stats(df: pd.DataFrame) -> dict[str, Any]:
    def stat_block(s: pd.Series) -> dict[str, Any]:
        if s is None or s.isna().all():
            return {"min": None, "max": None, "mean": None, "std": None}
        return {
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "std": float(s.std()),
        }

    return {
        "total_rows": int(len(df)),
        "missing_prev_close_rows": int(df["prev_close"].isna().sum()) if "prev_close" in df else None,
        "mike": stat_block(df.get("mike")),
        "rvol": stat_block(df.get("rvol")),
    }


# -----------------------------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------------------------

def create_mike_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["mike"],
            mode="lines",
            name="Mike (bps)",
            line=dict(color=MIKE_LINE_COLOR, width=MIKE_LINE_WIDTH),
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Mike: %{y:.0f} bps<extra></extra>",
        )
    )

    # Baseline at 0 bps (prev close)
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        annotation_text="Prev Close (0 bps)",
        annotation_position="right",
    )

    fig.update_layout(
        title=dict(
            text=f"{ticker} — Mike (bps from prev close)",
            font=dict(size=20),
        ),
        xaxis=dict(
            title="Time (New York)",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.35)",
        ),
        yaxis=dict(
            title="Mike (bps)",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.35)",
        ),
        hovermode="x unified",
        # transparent background so it blends with Streamlit theme
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=520,
        margin=dict(l=60, r=30, t=80, b=60),
    )

    return fig


# -----------------------------------------------------------------------------
# APP
# -----------------------------------------------------------------------------

def main() -> None:
    st.title("📈 VolMike — Market Microstructure Analysis")

    st.markdown(
        """
        **Mike** = basis points from the previous close (displacement-style).  
        **RVOL** = current bar volume / rolling mean volume (per day).
        """
    )

    with st.sidebar:
        st.header("Parameters")

        ticker_input = st.text_input("Ticker", value="SPY")
        c1, c2 = st.columns(2)
        with c1:
            start_d = st.date_input("Start", value=date.today() - timedelta(days=5))
        with c2:
            end_d = st.date_input("End", value=date.today())

        interval_choice = st.selectbox("Interval", list(INTERVAL_MAP.keys()), index=1)
        rvol_window = st.number_input("RVOL Window (bars)", min_value=5, max_value=200, value=20, step=1)

        run = st.button("🚀 Run", type="primary", use_container_width=True)

    # Nice default: run on first load as well (optional)
    if "auto_ran" not in st.session_state:
        st.session_state.auto_ran = True
        run = True

    if not run:
        st.info("Adjust parameters in the sidebar, then press **Run**.")
        return

    try:
        ticker = validate_ticker(ticker_input)
        validate_date_range(start_d, end_d)
        interval = INTERVAL_MAP[interval_choice]
        params = AnalysisParams(ticker=ticker, start_date=start_d, end_date=end_d, interval=interval, rvol_window=int(rvol_window))
    except ValidationError as e:
        st.error(f"❌ {e}")
        return
    except Exception as e:
        st.error(f"❌ Unexpected validation error: {e}")
        logger.exception("Validation failed")
        return

    # yfinance intraday constraints hint
    if interval in ("2m", "5m") and (params.end_date - params.start_date).days > 59:
        st.warning("yfinance often limits 2m/5m history. If data is missing, try a smaller date range.")

    with st.spinner("📡 Fetching data..."):
        try:
            intraday = fetch_intraday(params.ticker, params.start_date, params.end_date, params.interval)
            daily_lookup = fetch_daily_backfill(params.ticker, params.start_date, params.end_date, backfill_days=90)
        except Exception as e:
            st.error(f"❌ Data fetch failed: {e}")
            logger.exception("Fetch failed")
            return

    if intraday.empty:
        st.error("❌ No intraday bars returned. Try a wider range, different interval, or different ticker.")
        return

    if daily_lookup.empty or len(daily_lookup) < 2:
        st.error("❌ Not enough daily history to compute previous close.")
        return

    with st.spinner("⚙️ Processing..."):
        try:
            intraday_rth = filter_rth(intraday)
            if intraday_rth.empty:
                st.error("❌ No RTH bars found (09:30–16:00 ET).")
                return

            table = compute_table(params, intraday_rth, daily_lookup)
            stats = calc_stats(table)
        except Exception as e:
            st.error(f"❌ Processing failed: {e}")
            logger.exception("Processing failed")
            return

    st.success(f"✅ Done — {len(table)} RTH bars")

    # Chart
    st.subheader("📊 Mike Chart (Interactive)")
    fig = create_mike_chart(table, params.ticker)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "responsive": True})

    # Stats
    st.subheader("📈 Summary")
    a, b, c = st.columns(3)
    a.metric("Bars", stats["total_rows"])
    b.metric("Mike (mean)", f"{stats['mike']['mean']:.1f}" if stats["mike"]["mean"] is not None else "N/A")
    c.metric("RVOL (mean)", f"{stats['rvol']['mean']:.2f}" if stats["rvol"]["mean"] is not None else "N/A")

    if stats.get("missing_prev_close_rows"):
        st.warning(f"Some rows are missing prev close mapping: {stats['missing_prev_close_rows']}")

    # Table
    st.subheader("📋 Data Table")
    show = table.copy()
    show["time"] = show["time"].dt.strftime("%Y-%m-%d %H:%M")
    show["price"] = show["price"].map(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    show["prev_close"] = show["prev_close"].map(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    show["mike"] = show["mike"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
    show["rvol"] = show["rvol"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

    st.dataframe(
        show[["time", "price", "prev_close", "mike", "F%", "rvol"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "time": st.column_config.TextColumn("Time (NY)"),
            "price": st.column_config.TextColumn("Close"),
            "prev_close": st.column_config.TextColumn("Prev Close"),
            "mike": st.column_config.TextColumn("Mike (bps)"),
            "F%": st.column_config.TextColumn("F%"),
            "rvol": st.column_config.TextColumn("RVOL"),
        },
    )

    # Export
    st.subheader("💾 Export")
    export_payload = {
        "metadata": {
            "ticker": params.ticker,
            "start_date": str(params.start_date),
            "end_date": str(params.end_date),
            "interval": interval_choice,
            "timezone": NY_TZ,
            "rvol_window": params.rvol_window,
            "computed_at_utc": get_utc_timestamp(),
            "mike_definition": "((Close - PrevClose)/PrevClose) * 10000  (bps displacement)",
        },
        "statistics": stats,
        "data": table.assign(time=table["time"].dt.strftime("%Y-%m-%d %H:%M")).to_dict(orient="records"),
    }

    st.download_button(
        "📥 Download JSON",
        data=json.dumps(export_payload, indent=2).encode("utf-8"),
        file_name=f"{params.ticker}_mike_rvol_{params.start_date}_{params.end_date}_{interval_choice}.json",
        mime="application/json",
        use_container_width=True,
    )

    st.download_button(
        "📥 Download CSV",
        data=table.assign(time=table["time"].dt.strftime("%Y-%m-%d %H:%M")).to_csv(index=False).encode("utf-8"),
        file_name=f"{params.ticker}_mike_rvol_{params.start_date}_{params.end_date}_{interval_choice}.csv",
        mime="text/csv",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
