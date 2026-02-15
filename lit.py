"""
VolMike - Financial Market Microstructure Analysis Tool

A production-ready Streamlit application for analyzing intraday price movements
using Mike (basis points from previous close) and relative volume metrics.

Author: Trading Analytics Team
Version: 2.0.0
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
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

# Configure Streamlit page
st.set_page_config(
    page_title="VolMike - Market Microstructure Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
NY_TZ = "America/New_York"
RTH_START = time(9, 30)  # Regular Trading Hours start
RTH_END = time(16, 0)  # Regular Trading Hours end (exclusive)

# Available intervals for intraday data
INTERVAL_MAP = {
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "60m": "60m",
}

# Validation patterns
TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-]{1,12}$")

# Styling
MIKE_LINE_COLOR = "dodgerblue"
MIKE_LINE_WIDTH = 2


# -----------------------------------------------------------------------------
# DATA MODELS
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketDataRow:
    """Represents a single row of processed market data."""

    timestamp_ny: str
    price: float
    mike: Optional[float]
    rvol: Optional[float]


@dataclass(frozen=True)
class AnalysisParams:
    """Parameters for market analysis."""

    ticker: str
    start_date: date
    end_date: date
    interval: str
    rvol_window: int = 20


@dataclass(frozen=True)
class AnalysisResult:
    """Complete analysis result with metadata."""

    params: AnalysisParams
    data: pd.DataFrame
    computed_at_utc: str
    stats: dict[str, Any]


# -----------------------------------------------------------------------------
# VALIDATION & UTILITIES
# -----------------------------------------------------------------------------


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass


def validate_ticker(ticker: str) -> str:
    """
    Validate and normalize ticker symbol.

    Args:
        ticker: Raw ticker input from user

    Returns:
        Normalized ticker symbol (uppercase, stripped)

    Raises:
        ValidationError: If ticker format is invalid
    """
    normalized = (ticker or "").upper().strip()

    if not normalized:
        raise ValidationError("Ticker cannot be empty.")

    if not TICKER_PATTERN.match(normalized):
        raise ValidationError(
            f"Invalid ticker format: '{ticker}'. "
            "Use only letters, numbers, dots, and hyphens (no spaces)."
        )

    return normalized


def validate_date_range(start: date, end: date) -> None:
    """
    Validate date range is logical.

    Args:
        start: Start date
        end: End date

    Raises:
        ValidationError: If date range is invalid
    """
    if start > end:
        raise ValidationError(f"Start date ({start}) cannot be after end date ({end}).")

    if end > date.today():
        raise ValidationError(f"End date ({end}) cannot be in the future.")

    # Warn if range is very large
    delta = (end - start).days
    if delta > 365:
        logger.warning(f"Large date range requested: {delta} days")


def safe_float_convert(value: Any) -> Optional[float]:
    """
    Safely convert value to float, handling NaN/inf.

    Args:
        value: Value to convert

    Returns:
        Float value if valid, None otherwise
    """
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def get_utc_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


# -----------------------------------------------------------------------------
# DATA FETCHING
# -----------------------------------------------------------------------------


def ensure_utc_timezone(series: pd.Series) -> pd.Series:
    """
    Ensure datetime series is UTC timezone-aware.

    Args:
        series: Pandas datetime series

    Returns:
        UTC timezone-aware series
    """
    dt_series = pd.to_datetime(series, errors="coerce")

    if dt_series.dt.tz is None:
        # Naive datetime - localize to UTC
        return dt_series.dt.tz_localize("UTC")
    else:
        # Already timezone-aware - convert to UTC
        return dt_series.dt.tz_convert("UTC")


def normalize_yfinance_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance output to consistent format.

    Args:
        df: Raw yfinance DataFrame

    Returns:
        Normalized DataFrame with columns: time (UTC), Close, Volume
    """
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    data = data.reset_index()

    # Find time column
    time_col = None
    for candidate in ("Datetime", "Date", "index"):
        if candidate in data.columns:
            time_col = candidate
            break

    if time_col is None:
        logger.error("No time column found in yfinance data")
        return pd.DataFrame()

    # Rename and process
    data = data.rename(columns={time_col: "time"})
    data["time"] = ensure_utc_timezone(data["time"])
    data["Close"] = pd.to_numeric(data.get("Close"), errors="coerce")
    data["Volume"] = pd.to_numeric(data.get("Volume"), errors="coerce")

    # Clean and sort
    data = data.dropna(subset=["time", "Close"]).sort_values("time").reset_index(drop=True)

    return data[["time", "Close", "Volume"]]

@st.cache_data(ttl=300, show_spinner=False)

def fetch_intraday_data(
    ticker: str,
    start_date: date,
    end_date: date,
    interval: str,
) -> pd.DataFrame:
    """
    Fetch intraday bars from yfinance.

    Args:
        ticker: Stock symbol
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        interval: Time interval (e.g., '5m')

    Returns:
        DataFrame with normalized intraday data

    Raises:
        Exception: If fetching fails
    """
    logger.info(f"Fetching intraday data: {ticker} from {start_date} to {end_date}, interval={interval}")

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

    try:
        raw_data = yf.download(
            tickers=ticker,
            start=start_dt,
            end=end_dt,
            interval=interval,
            auto_adjust=False,
            prepost=False,
            progress=False,
            threads=False,
        )

        normalized = normalize_yfinance_data(raw_data)
        logger.info(f"Fetched {len(normalized)} intraday bars")

        return normalized

    except Exception as e:
        logger.error(f"Error fetching intraday data: {e}")
        raise


def fetch_daily_data_with_backfill(
    ticker: str,
    start_date: date,
    end_date: date,
    backfill_days: int = 45,
) -> pd.DataFrame:
    """
    Fetch daily bars with backfill for previous close calculation.

    Args:
        ticker: Stock symbol
        start_date: Analysis start date
        end_date: Analysis end date
        backfill_days: Days to fetch before start_date

    Returns:
        DataFrame with daily data

    Raises:
        Exception: If fetching fails
    """
    logger.info(f"Fetching daily data with {backfill_days}-day backfill")

    backfill_start = start_date - timedelta(days=backfill_days)
    start_dt = datetime.combine(backfill_start, datetime.min.time())
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

    try:
        raw_data = yf.download(
            tickers=ticker,
            start=start_dt,
            end=end_dt,
            interval="1d",
            auto_adjust=False,
            prepost=False,
            progress=False,
            threads=False,
        )

        if raw_data is None or raw_data.empty:
            raise ValueError("No daily data returned from yfinance")

        data = raw_data.copy()

        # Flatten MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

        # Clean data
        data = data.dropna(subset=["Close"]).copy()

        logger.info(f"Fetched {len(data)} daily bars")
        return data

    except Exception as e:
        logger.error(f"Error fetching daily data: {e}")
        raise


# -----------------------------------------------------------------------------
# DATA PROCESSING
# -----------------------------------------------------------------------------


def build_previous_close_lookup(daily_data: pd.DataFrame) -> tuple[list[date], list[float]]:
    """
    Build lookup tables for previous trading day closes.

    Args:
        daily_data: Daily OHLCV data

    Returns:
        Tuple of (trading_dates, close_prices) sorted by date

    Raises:
        ValueError: If daily data is invalid or insufficient
    """
    if daily_data is None or daily_data.empty or "Close" not in daily_data.columns:
        raise ValueError("Daily data is missing or invalid (required for previous close calculation)")

    # Parse index as dates
    index_dates = pd.to_datetime(daily_data.index, errors="coerce")

    if index_dates.isna().all():
        raise ValueError("Daily data index is not parseable as dates")

    # Create clean lookup DataFrame
    lookup_df = pd.DataFrame(
        {
            "date": index_dates.date,
            "close": pd.to_numeric(daily_data["Close"], errors="coerce").astype(float),
        }
    ).dropna(subset=["date", "close"])

    # Remove duplicates and sort
    lookup_df = lookup_df.drop_duplicates(subset=["date"]).sort_values("date")

    trading_dates = lookup_df["date"].tolist()
    close_prices = lookup_df["close"].tolist()

    if len(trading_dates) < 2:
        raise ValueError(
            "Insufficient daily history for previous close calculation "
            f"(got {len(trading_dates)} days, need at least 2)"
        )

    logger.info(f"Built previous close lookup with {len(trading_dates)} trading days")
    return trading_dates, close_prices


def get_previous_close(
    trading_dates: list[date],
    close_prices: list[float],
    current_date: date,
) -> Optional[float]:
    """
    Get the previous trading day's close for a given date.

    Args:
        trading_dates: Sorted list of trading dates
        close_prices: Corresponding close prices
        current_date: Date to find previous close for

    Returns:
        Previous trading day's close, or None if not available
    """
    # Binary search for insertion point
    insertion_idx = int(
        np.searchsorted(
            np.array(trading_dates, dtype="O"),
            current_date,
            side="left",
        )
    )

    # If current_date is exactly a trading date, get previous
    if insertion_idx < len(trading_dates) and trading_dates[insertion_idx] == current_date:
        if insertion_idx > 0:
            return float(close_prices[insertion_idx - 1])
        return None

    # Otherwise, get the last close before current_date
    if insertion_idx > 0:
        return float(close_prices[insertion_idx - 1])

    return None


def filter_regular_trading_hours(intraday_data: pd.DataFrame) -> pd.DataFrame:
    """
    Filter intraday data to Regular Trading Hours (RTH) in New York time.

    RTH is defined as 09:30 <= time < 16:00 ET.

    Args:
        intraday_data: Intraday data with UTC 'time' column

    Returns:
        Filtered DataFrame with only RTH data, with 'time_ny' column added
    """
    if intraday_data is None or intraday_data.empty:
        return pd.DataFrame()

    data = intraday_data.copy()

    # Convert to NY timezone
    data["time_ny"] = data["time"].dt.tz_convert(NY_TZ)

    # Filter to RTH
    time_only = data["time_ny"].dt.time
    rth_mask = (time_only >= RTH_START) & (time_only < RTH_END)

    filtered = data.loc[rth_mask].sort_values("time_ny").reset_index(drop=True)

    logger.info(
        f"Filtered to RTH: {len(filtered)} bars "
        f"({len(filtered)/len(data)*100:.1f}% of total)"
    )

    return filtered


def calculate_mike(price: float, previous_close: float) -> Optional[float]:
    """
    Calculate Mike: price movement in basis points from previous close.

    Mike = (price / previous_close) * 10,000

    This represents the price as basis points (hundredths of a percent)
    relative to the previous close.

    Args:
        price: Current price
        previous_close: Previous trading day's close

    Returns:
        Mike value, or None if calculation not possible
    """
    if previous_close is None or previous_close <= 0:
        return None

    if price is None or not np.isfinite(price):
        return None

    return float((price / previous_close) * 10_000.0)


def calculate_relative_volume(volume_series: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling relative volume (RVOL).

    RVOL = current_volume / rolling_average_volume

    Args:
        volume_series: Series of volume values
        window: Rolling window size

    Returns:
        Series of RVOL values
    """
    window = max(1, int(window))
    min_periods = max(1, window // 2)

    # Convert to numeric and compute rolling average
    vol_numeric = pd.to_numeric(volume_series, errors="coerce").astype(float)
    rolling_avg = vol_numeric.rolling(window=window, min_periods=min_periods).mean()

    # Calculate RVOL
    rvol = vol_numeric / rolling_avg

    # Replace inf/-inf with NaN
    rvol = rvol.replace([np.inf, -np.inf], np.nan)

    return rvol


def compute_analysis_table(
    ticker: str,
    intraday_rth: pd.DataFrame,
    daily_data: pd.DataFrame,
    rvol_window: int = 20,
) -> pd.DataFrame:
    """
    Compute complete analysis table with Mike and RVOL metrics.

    Args:
        ticker: Stock symbol
        intraday_rth: RTH-filtered intraday data
        daily_data: Daily data for previous close calculation
        rvol_window: Window size for RVOL calculation

    Returns:
        DataFrame with columns: time, price, mike, rvol

    Raises:
        ValueError: If computation fails due to data issues
    """
    if intraday_rth is None or intraday_rth.empty:
        raise ValueError(
            "No RTH intraday data available. "
            "Try a wider date range or different interval."
        )

    logger.info(f"Computing analysis table for {ticker}")

    # Build previous close lookup
    trading_dates, close_prices = build_previous_close_lookup(daily_data)

    data = intraday_rth.copy()

    # Extract NY trading date
    data["trading_date"] = data["time_ny"].dt.date

    # Get previous close for each trading date
    data["prev_close"] = data["trading_date"].map(
        lambda d: get_previous_close(trading_dates, close_prices, d)
    )

    # Calculate Mike
    data["mike"] = data.apply(
        lambda row: calculate_mike(
            safe_float_convert(row.get("Close")),
            safe_float_convert(row.get("prev_close")),
        ),
        axis=1,
    )




    # Calculate RVOL
    data["rvol"] = data.groupby("trading_date")["Volume"].transform(
    lambda s: calculate_relative_volume(s, window=rvol_window)
).astype(float)

    # Create output table
    output = pd.DataFrame(
        {
            "time": data["time_ny"].dt.strftime("%Y-%m-%d %H:%M"),
            "price": pd.to_numeric(data["Close"], errors="coerce").astype(float),
            "mike": pd.to_numeric(data["mike"], errors="coerce").astype(float),
            "rvol": pd.to_numeric(data["rvol"], errors="coerce").astype(float),
        }
    )

    # Clean output
    output = output.dropna(subset=["time", "price"]).reset_index(drop=True)

    logger.info(f"Computed {len(output)} rows with Mike and RVOL")

    return output


def calculate_statistics(data: pd.DataFrame) -> dict[str, Any]:
    """
    Calculate summary statistics for the analysis.

    Args:
        data: Analysis data with mike and rvol columns

    Returns:
        Dictionary of statistics
    """
    stats = {
        "total_rows": len(data),
        "mike": {
            "min": float(data["mike"].min()) if not data["mike"].isna().all() else None,
            "max": float(data["mike"].max()) if not data["mike"].isna().all() else None,
            "mean": float(data["mike"].mean()) if not data["mike"].isna().all() else None,
            "std": float(data["mike"].std()) if not data["mike"].isna().all() else None,
        },
        "rvol": {
            "min": float(data["rvol"].min()) if not data["rvol"].isna().all() else None,
            "max": float(data["rvol"].max()) if not data["rvol"].isna().all() else None,
            "mean": float(data["rvol"].mean()) if not data["rvol"].isna().all() else None,
            "std": float(data["rvol"].std()) if not data["rvol"].isna().all() else None,
        },
    }

    return stats


# -----------------------------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------------------------


def create_mike_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Create interactive Mike time series chart using Plotly.

    Args:
        data: Analysis data with 'time' and 'mike' columns
        ticker: Stock symbol for chart title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Add Mike line trace
    fig.add_trace(
        go.Scatter(
            x=data["time"],
            y=data["mike"],
            mode="lines",
            name="Mike",
            line=dict(
                color=MIKE_LINE_COLOR,
                width=MIKE_LINE_WIDTH,
            ),
            hovertemplate="<b>%{x}</b><br>Mike: %{y:.2f}<extra></extra>",
        )
    )

    # Add horizontal line at 10,000 (unchanged from previous close)
    fig.add_hline(
        y=10_000,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        annotation_text="Previous Close",
        annotation_position="right",
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{ticker} - Mike (Basis Points from Previous Close)",
            font=dict(size=20, color="#1f77b4"),
        ),
        xaxis=dict(
            title="Time (New York)",
            gridcolor="#e5e5e5",
            showgrid=True,
        ),
        yaxis=dict(
            title="Mike (basis points)",
            gridcolor="#e5e5e5",
            showgrid=True,
        ),
        hovermode="x unified",
        plot_bgcolor="white",
        height=500,
        margin=dict(l=60, r=40, t=80, b=60),
    )

    return fig


# -----------------------------------------------------------------------------
# MAIN APPLICATION
# -----------------------------------------------------------------------------


def main():
    """Main Streamlit application."""

    # Header
    st.title("📈 VolMike - Market Microstructure Analysis")
    st.markdown(
        """
        Analyze intraday price movements using **Mike** (basis points from previous close)
        and **Relative Volume** metrics during Regular Trading Hours (RTH).
        """
    )

    # Input section
    st.subheader("Analysis Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        ticker_input = st.text_input(
            "Ticker Symbol",
            value="SPY",
            help="Enter stock ticker (e.g., SPY, AAPL, TSLA)",
        )

    with col2:
        start_date = st.date_input(
            "Start Date",
            value=date.today() - timedelta(days=5),
            help="Analysis start date (inclusive)",
        )

    with col3:
        end_date = st.date_input(
            "End Date",
            value=date.today(),
            help="Analysis end date (inclusive)",
        )

    col4, col5 = st.columns([1, 2])

    with col4:
        interval_choice = st.selectbox(
            "Time Interval",
            options=list(INTERVAL_MAP.keys()),
            index=1,  # Default to 5m
            help="Intraday bar interval",
        )

    with col5:
        st.markdown("**RTH:** 09:30 - 16:00 ET | **RVOL Window:** 20 bars")

    # Run button
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    if run_analysis:
        try:
            # Validate inputs
            ticker = validate_ticker(ticker_input)
            validate_date_range(start_date, end_date)
            interval = INTERVAL_MAP[interval_choice]

            params = AnalysisParams(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                rvol_window=20,
            )

        except ValidationError as e:
            st.error(f"❌ Validation Error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected Error: {e}")
            logger.exception("Validation failed")
            st.stop()

        # Fetch data
        with st.spinner("📡 Fetching data from yfinance..."):
            try:
                intraday_data = fetch_intraday_data(
                    params.ticker,
                    params.start_date,
                    params.end_date,
                    params.interval,
                )
                daily_data = fetch_daily_data_with_backfill(
                    params.ticker,
                    params.start_date,
                    params.end_date,
                )
            except Exception as e:
                st.error(f"❌ Data Fetch Error: {e}")
                logger.exception("Data fetching failed")
                st.stop()

        # Validate fetched data
        if intraday_data.empty:
            st.error(
                "❌ No intraday data returned. "
                "Try a wider date range or different ticker/interval."
            )
            st.stop()

        if daily_data.empty:
            st.error(
                "❌ No daily data returned (required for previous close calculation). "
                "Please try again."
            )
            st.stop()

        # Process data
        with st.spinner("⚙️ Processing data..."):
            try:
                intraday_rth = filter_regular_trading_hours(intraday_data)

                if intraday_rth.empty:
                    st.error(
                        "❌ No RTH bars found for the selected date range. "
                        "Try different dates or interval."
                    )
                    st.stop()

                analysis_table = compute_analysis_table(
                    params.ticker,
                    intraday_rth,
                    daily_data,
                    params.rvol_window,
                )

                stats = calculate_statistics(analysis_table)

            except Exception as e:
                st.error(f"❌ Processing Error: {e}")
                logger.exception("Data processing failed")
                st.stop()

        # Display results
        st.success(f"✅ Analysis complete: {len(analysis_table)} RTH bars processed")

        # Mike chart
        st.subheader("📊 Mike Chart")
        mike_chart = create_mike_chart(analysis_table, params.ticker)
        st.plotly_chart(mike_chart, use_container_width=True)

        # Statistics
        st.subheader("📈 Summary Statistics")
        col_stat1, col_stat2 = st.columns(2)

        with col_stat1:
            st.metric("Total Bars", stats["total_rows"])
            if stats["mike"]["mean"] is not None:
                st.metric(
                    "Mike Mean",
                    f"{stats['mike']['mean']:.2f}",
                    help="Average Mike value across all bars",
                )
                st.metric(
                    "Mike Std Dev",
                    f"{stats['mike']['std']:.2f}",
                    help="Standard deviation of Mike",
                )

        with col_stat2:
            if stats["mike"]["min"] is not None and stats["mike"]["max"] is not None:
                st.metric("Mike Range", f"{stats['mike']['min']:.2f} - {stats['mike']['max']:.2f}")
            if stats["rvol"]["mean"] is not None:
                st.metric(
                    "RVOL Mean",
                    f"{stats['rvol']['mean']:.2f}",
                    help="Average relative volume",
                )

        # Data table
        st.subheader("📋 Data Table")

        # Format table for display
        display_table = analysis_table.copy()
        display_table["mike"] = display_table["mike"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        display_table["rvol"] = display_table["rvol"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        display_table["price"] = display_table["price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")

        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "time": st.column_config.TextColumn("Time (NY)"),
                "price": st.column_config.TextColumn("Price"),
                "mike": st.column_config.TextColumn("Mike (bps)"),
                "rvol": st.column_config.TextColumn("RVOL"),
            },
        )

        # Export section
        st.subheader("💾 Export Data")

        # Create JSON export
        export_data = {
            "metadata": {
                "ticker": params.ticker,
                "start_date": str(params.start_date),
                "end_date": str(params.end_date),
                "interval": interval_choice,
                "timezone": NY_TZ,
                "rvol_window": params.rvol_window,
                "computed_at_utc": get_utc_timestamp(),
            },
            "statistics": stats,
            "data": analysis_table.to_dict(orient="records"),
        }

        json_bytes = json.dumps(export_data, indent=2).encode("utf-8")

        st.download_button(
            label="📥 Download JSON",
            data=json_bytes,
            file_name=f"{params.ticker}_mike_rvol_{params.start_date}_{params.end_date}_{interval_choice}.json",
            mime="application/json",
            use_container_width=True,
        )

        # CSV export
        csv_bytes = analysis_table.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download CSV",
            data=csv_bytes,
            file_name=f"{params.ticker}_mike_rvol_{params.start_date}_{params.end_date}_{interval_choice}.csv",
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()