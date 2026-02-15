"""
VolMike - Market Microstructure Analysis (Streamlit)

Production-ready Streamlit app for intraday analysis using:
- Mike (level): (price / previous_close) * 10,000  -> baseline 10,000
- F_bps (move): Mike - 10,000                      -> basis-point move from prev close
- RVOL: intraday relative volume vs rolling mean (per day)

Run:
  pip install -r requirements.txt
  streamlit run volmike_app.py
"""

from __future__ import annotations

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
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

APP_TITLE = "📈 VolMike - Market Microstructure Analysis"
NY_TZ = "America/New_York"
RTH_START = time(9, 30)
RTH_END = time(16, 0)  # exclusive

TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,12}$")

INTERVALS = ["2m", "5m", "15m", "30m", "60m"]

DEFAULT_TICKERS = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AVGO", "META", "AMZN",
    "COIN", "PLTR", "MU", "QCOM", "MRVL", "HOOD", "VIXY",
]

# Chart styling
MIKE_COLOR = "dodgerblue"
F_COLOR = "#57c7ff"
TENKAN_COLOR = "#E63946"
KIJUN_COLOR = "#2ECC71"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("volmike")


# -----------------------------------------------------------------------------
# MODELS
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Params:
    tickers: list[str]
    start: date
    end: date
    interval: str
    rvol_window: int
    show_ichimoku: bool
    show_bbands: bool


# -----------------------------------------------------------------------------
# VALIDATION
# -----------------------------------------------------------------------------

class ValidationError(Exception):
    pass


def validate_ticker(raw: str) -> str:
    t = (raw or "").upper().strip()
    if not t:
        raise ValidationError("Ticker cannot be empty.")
    if not TICKER_RE.match(t):
        raise ValidationError(
            f"Invalid ticker: '{raw}'. Use letters/numbers/dot/hyphen only."
        )
    return t


def validate_dates(start: date, end: date) -> None:
    if start > end:
        raise ValidationError("Start date cannot be after end date.")
    # Use UTC today to avoid server timezone weirdness; allow end == today in NY.
    if end > datetime.now(timezone.utc).date():
        raise ValidationError("End date cannot be in the future.")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# -----------------------------------------------------------------------------
# YFINANCE NORMALIZATION
# -----------------------------------------------------------------------------

def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    return out


def _normalize_intraday(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Returns columns: time_utc (tz-aware), Open, High, Low, Close, Volume
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = _flatten_cols(raw).copy()
    df = df.reset_index()

    time_col = None
    for c in ("Datetime", "Date", "index"):
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        return pd.DataFrame()

    df = df.rename(columns={time_col: "time"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # Make tz-aware UTC
    if getattr(df["time"].dt, "tz", None) is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    else:
        df["time"] = df["time"].dt.tz_convert("UTC")

    keep = ["time", "Open", "High", "Low", "Close", "Volume"]
    for col in keep[1:]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    df = df.dropna(subset=["time", "Close"]).sort_values("time").reset_index(drop=True)
    return df[keep]


def _normalize_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Returns index as date (python date), columns include Close/High/Low.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = _flatten_cols(raw).copy()
    df = df.dropna(subset=["Close"]).copy()

    idx = pd.to_datetime(df.index, errors="coerce")
    if idx.isna().all():
        return pd.DataFrame()

    df = df.copy()
    df.index = idx.date
    # Dedup dates
    df = df[~pd.Index(df.index).duplicated(keep="last")]
    return df.sort_index()


# -----------------------------------------------------------------------------
# FETCHING (CACHED)
# -----------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def fetch_intraday(ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
    """
    Fetch intraday bars from yfinance (pre/post excluded).
    end is inclusive in UI; yfinance end is exclusive -> add 1 day.
    """
    start_dt = datetime.combine(start, time.min)
    end_dt = datetime.combine(end + timedelta(days=1), time.min)

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
    return _normalize_intraday(raw)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_daily_backfill(
    ticker: str, start: date, end: date, backfill_days: int = 45
) -> pd.DataFrame:
    """
    Daily bars including backfill so we can find previous close for start day.
    """
    bf_start = start - timedelta(days=backfill_days)
    start_dt = datetime.combine(bf_start, time.min)
    end_dt = datetime.combine(end + timedelta(days=1), time.min)

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
    return _normalize_daily(raw)


# -----------------------------------------------------------------------------
# PROCESSING
# -----------------------------------------------------------------------------

def to_rth(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["time_ny"] = out["time"].dt.tz_convert(NY_TZ)
    t = out["time_ny"].dt.time
    mask = (t >= RTH_START) & (t < RTH_END)
    out = out.loc[mask].sort_values("time_ny").reset_index(drop=True)
    out["trading_date"] = out["time_ny"].dt.date
    return out


def build_prev_close_lookup(daily: pd.DataFrame) -> tuple[list[date], list[float]]:
    if daily.empty or "Close" not in daily.columns:
        raise ValidationError("Daily data missing (need Close).")

    dates = list(daily.index)
    closes = pd.to_numeric(daily["Close"], errors="coerce").astype(float).tolist()

    good = [(d, c) for d, c in zip(dates, closes) if d is not None and np.isfinite(c)]
    if len(good) < 2:
        raise ValidationError("Not enough daily history to compute previous close.")

    good.sort(key=lambda x: x[0])
    td, cp = zip(*good)
    return list(td), list(cp)


def prev_close_for_date(
    trading_dates: list[date], close_prices: list[float], d: date
) -> Optional[float]:
    # insertion point
    arr = np.array(trading_dates, dtype="O")
    i = int(np.searchsorted(arr, d, side="left"))

    if i < len(trading_dates) and trading_dates[i] == d:
        return float(close_prices[i - 1]) if i > 0 else None
    return float(close_prices[i - 1]) if i > 0 else None


def rvol_per_day(volume: pd.Series, window: int) -> pd.Series:
    w = max(1, int(window))
    mp = max(1, w // 2)
    v = pd.to_numeric(volume, errors="coerce").astype(float)
    avg = v.rolling(window=w, min_periods=mp).mean()
    out = v / avg
    return out.replace([np.inf, -np.inf], np.nan)


def add_ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26) -> pd.DataFrame:
    out = df.copy()
    hi = pd.to_numeric(out["High"], errors="coerce").astype(float)
    lo = pd.to_numeric(out["Low"], errors="coerce").astype(float)

    tenkan_hi = hi.rolling(window=tenkan, min_periods=1).max()
    tenkan_lo = lo.rolling(window=tenkan, min_periods=1).min()
    kijun_hi = hi.rolling(window=kijun, min_periods=1).max()
    kijun_lo = lo.rolling(window=kijun, min_periods=1).min()

    out["tenkan_price"] = (tenkan_hi + tenkan_lo) / 2.0
    out["kijun_price"] = (kijun_hi + kijun_lo) / 2.0
    return out


def add_bbands(series: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    ma = x.rolling(window=window, min_periods=1).mean()
    sd = x.rolling(window=window, min_periods=1).std()
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    return pd.DataFrame({"bb_ma": ma, "bb_upper": upper, "bb_lower": lower})


def compute_table(
    ticker: str, intraday: pd.DataFrame, daily: pd.DataFrame, rvol_window: int, show_ich: bool, show_bb: bool
) -> pd.DataFrame:
    if intraday.empty:
        raise ValidationError("No intraday data returned.")
    if daily.empty:
        raise ValidationError("No daily data returned.")

    rth = to_rth(intraday)
    if rth.empty:
        raise ValidationError("No RTH bars found for selected date range.")

    trading_dates, close_prices = build_prev_close_lookup(daily)
    rth["prev_close"] = rth["trading_date"].map(lambda d: prev_close_for_date(trading_dates, close_prices, d))

    cl = pd.to_numeric(rth["Close"], errors="coerce").astype(float)
    pc = pd.to_numeric(rth["prev_close"], errors="coerce").astype(float)

    # Mike level (baseline 10,000)
    rth["mike"] = np.where(pc > 0, (cl / pc) * 10_000.0, np.nan)
    # F_bps move (baseline 0)
    rth["f_bps"] = rth["mike"] - 10_000.0

    # RVOL per trading day (no overnight bleed)
    rth["rvol"] = rth.groupby("trading_date")["Volume"].transform(lambda s: rvol_per_day(s, rvol_window))

    if show_ich:
        rth = add_ichimoku(rth)
        # Convert ichimoku prices into Mike/F space using prev_close
        tp = pd.to_numeric(rth["tenkan_price"], errors="coerce").astype(float)
        kp = pd.to_numeric(rth["kijun_price"], errors="coerce").astype(float)
        rth["tenkan_mike"] = np.where(pc > 0, (tp / pc) * 10_000.0, np.nan)
        rth["kijun_mike"] = np.where(pc > 0, (kp / pc) * 10_000.0, np.nan)

    if show_bb:
        bb = add_bbands(rth["f_bps"], window=20, n_std=2.0)
        rth = pd.concat([rth, bb], axis=1)

    out = pd.DataFrame(
        {
            "time_ny": rth["time_ny"],
            "time": rth["time_ny"].dt.strftime("%Y-%m-%d %H:%M"),
            "price": cl,
            "prev_close": pc,
            "mike": pd.to_numeric(rth["mike"], errors="coerce"),
            "f_bps": pd.to_numeric(rth["f_bps"], errors="coerce"),
            "rvol": pd.to_numeric(rth["rvol"], errors="coerce"),
            "volume": pd.to_numeric(rth["Volume"], errors="coerce"),
        }
    )

    if show_ich:
        out["tenkan_mike"] = pd.to_numeric(rth.get("tenkan_mike"), errors="coerce")
        out["kijun_mike"] = pd.to_numeric(rth.get("kijun_mike"), errors="coerce")

    if show_bb:
        out["bb_ma"] = pd.to_numeric(rth.get("bb_ma"), errors="coerce")
        out["bb_upper"] = pd.to_numeric(rth.get("bb_upper"), errors="coerce")
        out["bb_lower"] = pd.to_numeric(rth.get("bb_lower"), errors="coerce")

    return out.dropna(subset=["time", "price"]).reset_index(drop=True)


def stats_block(df: pd.DataFrame) -> dict[str, Any]:
    def _s(col: str) -> dict[str, Optional[float]]:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.isna().all():
            return {"min": None, "max": None, "mean": None, "std": None}
        return {
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "std": float(s.std()),
        }

    return {
        "rows": int(len(df)),
        "mike": _s("mike"),
        "f_bps": _s("f_bps"),
        "rvol": _s("rvol"),
        "computed_at_utc": utc_now_iso(),
    }


# -----------------------------------------------------------------------------
# CHARTS
# -----------------------------------------------------------------------------

def make_chart(df: pd.DataFrame, ticker: str, show_ich: bool, show_bb: bool) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
    )

    # Main: F_bps (move) line
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["f_bps"],
            mode="lines",
            name="F_bps (move)",
            line=dict(color=F_COLOR, width=2),
            hovertemplate="<b>%{x}</b><br>F_bps: %{y:.1f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Zero line (prev close)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

    if show_bb and all(c in df.columns for c in ("bb_upper", "bb_lower", "bb_ma")):
        fig.add_trace(go.Scatter(x=df["time"], y=df["bb_upper"], mode="lines",
                                 name="BB Upper", line=dict(width=1, color="#d0d0d0")),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["bb_lower"], mode="lines",
                                 name="BB Lower", line=dict(width=1, color="#d0d0d0")),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=df["bb_ma"], mode="lines",
                                 name="BB Mid", line=dict(width=2, dash="dash", color="#c0c0c0")),
                      row=1, col=1)

    if show_ich:
        if "kijun_mike" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=(df["kijun_mike"] - 10_000.0),
                    mode="lines",
                    name="Kijun (bps)",
                    line=dict(color=KIJUN_COLOR, width=1.6),
                ),
                row=1, col=1,
            )
        if "tenkan_mike" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=(df["tenkan_mike"] - 10_000.0),
                    mode="lines",
                    name="Tenkan (bps)",
                    line=dict(color=TENKAN_COLOR, width=1.0),
                ),
                row=1, col=1,
            )

    # RVOL
    fig.add_trace(
        go.Bar(
            x=df["time"],
            y=df["rvol"],
            name="RVOL",
            hovertemplate="<b>%{x}</b><br>RVOL: %{y:.2f}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.4, row=2, col=1)

    fig.update_layout(
        title=f"{ticker} — F_bps + RVOL (RTH only)",
        height=780,
        margin=dict(l=40, r=30, t=60, b=40),
        hovermode="x unified",
        plot_bgcolor="white",
    )
    fig.update_yaxes(title_text="F_bps", row=1, col=1, showgrid=True, gridcolor="#e9e9e9")
    fig.update_yaxes(title_text="RVOL", row=2, col=1, showgrid=True, gridcolor="#f0f0f0")
    fig.update_xaxes(showgrid=False)
    return fig


# -----------------------------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="VolMike",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    st.title(APP_TITLE)
    st.caption("Mike baseline is 10,000. F_bps is Mike - 10,000. RTH only (09:30–16:00 ET).")

    with st.sidebar:
        st.header("Inputs")

        tickers = st.multiselect("Tickers", options=DEFAULT_TICKERS, default=["SPY"])
        raw_extra = st.text_input("Add ticker (optional)", value="")
        if raw_extra.strip():
            try:
                extra = validate_ticker(raw_extra)
                if extra not in tickers:
                    tickers = tickers + [extra]
            except ValidationError as e:
                st.warning(str(e))

        c1, c2 = st.columns(2)
        with c1:
            start = st.date_input("Start", value=date.today() - timedelta(days=5))
        with c2:
            end = st.date_input("End", value=date.today())

        interval = st.selectbox("Interval", options=INTERVALS, index=1)
        rvol_window = st.number_input("RVOL window (bars)", min_value=3, max_value=200, value=20, step=1)

        show_ich = st.toggle("Show Ichimoku (Tenkan/Kijun)", value=True)
        show_bb = st.toggle("Show Bollinger (on F_bps)", value=True)

        run = st.button("🚀 Run", type="primary", use_container_width=True)

    if not run:
        st.info("Set your inputs and press **Run**.")
        return

    try:
        validate_dates(start, end)
        if not tickers:
            raise ValidationError("Select at least one ticker.")
        tickers = [validate_ticker(t) for t in tickers]

        params = Params(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval,
            rvol_window=int(rvol_window),
            show_ichimoku=bool(show_ich),
            show_bbands=bool(show_bb),
        )
    except ValidationError as e:
        st.error(f"❌ {e}")
        return

    tabs = st.tabs([f"📊 {t}" for t in params.tickers])

    for i, ticker in enumerate(params.tickers):
        with tabs[i]:
            st.subheader(ticker)

            with st.spinner("Fetching data..."):
                intraday = fetch_intraday(ticker, params.start, params.end, params.interval)
                daily = fetch_daily_backfill(ticker, params.start, params.end)

            if intraday.empty:
                st.error("No intraday data returned. Try another ticker or interval.")
                continue
            if daily.empty:
                st.error("No daily data returned (needed for previous close).")
                continue

            try:
                table = compute_table(
                    ticker=ticker,
                    intraday=intraday,
                    daily=daily,
                    rvol_window=params.rvol_window,
                    show_ich=params.show_ichimoku,
                    show_bb=params.show_bbands,
                )
            except ValidationError as e:
                st.error(f"❌ {e}")
                continue

            s = stats_block(table)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Bars (RTH)", s["rows"])
            if s["f_bps"]["mean"] is not None:
                m2.metric("F_bps mean", f"{s['f_bps']['mean']:.1f}")
            if s["f_bps"]["min"] is not None and s["f_bps"]["max"] is not None:
                m3.metric("F_bps range", f"{s['f_bps']['min']:.1f} → {s['f_bps']['max']:.1f}")
            if s["rvol"]["max"] is not None:
                m4.metric("RVOL max", f"{s['rvol']['max']:.2f}")

            fig = make_chart(table, ticker, params.show_ichimoku, params.show_bbands)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Data table"):
                show_cols = ["time", "price", "prev_close", "mike", "f_bps", "rvol", "volume"]
                if params.show_ichimoku:
                    for c in ("tenkan_mike", "kijun_mike"):
                        if c in table.columns:
                            show_cols.append(c)
                if params.show_bbands:
                    for c in ("bb_upper", "bb_lower", "bb_ma"):
                        if c in table.columns:
                            show_cols.append(c)

                view = table[show_cols].copy()
                view["price"] = view["price"].map(lambda x: f"${x:.2f}" if np.isfinite(x) else "")
                view["prev_close"] = view["prev_close"].map(lambda x: f"${x:.2f}" if np.isfinite(x) else "")
                for c in ("mike", "f_bps"):
                    if c in view.columns:
                        view[c] = pd.to_numeric(view[c], errors="coerce").map(
                            lambda x: f"{x:.1f}" if np.isfinite(x) else ""
                        )
                if "rvol" in view.columns:
                    view["rvol"] = pd.to_numeric(view["rvol"], errors="coerce").map(
                        lambda x: f"{x:.2f}" if np.isfinite(x) else ""
                    )

                st.dataframe(view, use_container_width=True, hide_index=True)

            # Downloads
            st.divider()
            export = {
                "metadata": {
                    "ticker": ticker,
                    "start": str(params.start),
                    "end": str(params.end),
                    "interval": params.interval,
                    "timezone": NY_TZ,
                    "rvol_window": params.rvol_window,
                    "computed_at_utc": s["computed_at_utc"],
                },
                "stats": s,
                "data": table.drop(columns=["time_ny"], errors="ignore").to_dict(orient="records"),
            }

            st.download_button(
                "📥 Download JSON",
                data=pd.Series(export).to_json(),
                file_name=f"{ticker}_volmike_{params.start}_{params.end}_{params.interval}.json",
                mime="application/json",
                use_container_width=True,
            )

            st.download_button(
                "📥 Download CSV",
                data=table.drop(columns=["time_ny"], errors="ignore").to_csv(index=False).encode("utf-8"),
                file_name=f"{ticker}_volmike_{params.start}_{params.end}_{params.interval}.csv",
                mime="text/csv",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
