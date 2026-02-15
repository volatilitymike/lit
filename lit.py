"""
VOLMIKE.COM - Financial Analysis Dashboard
Production-ready Streamlit application for market analysis and visualization
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import date, timedelta, datetime
from typing import List, Dict, Tuple, Optional
import base64
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_TICKERS = [
    "ES=F", "NQ=F", "YM=F", "SPY", "VIXY", "SOXX", "NVDA", "AMZN", "MU",
    "AMD", "QCOM", "SMCI", "MSFT", "UBER", "AVGO", "MRVL", "QQQ", "PLTR",
    "AAPL", "GOOGL", "META", "XLY", "TSLA", "NKE", "GM", "C", "DKNG",
    "CHWY", "ETSY", "CART", "W", "KBE", "WFC", "HOOD", "PYPL", "COIN",
    "BAC", "JPM", "BTC-USD", "ETH-USD", "XRP-USD", "ADA-USD", "SOL-USD",
    "DOGE-USD", "MES=F", "MYM=F", "M6E=F", "MGC=F", "MNQ=F", "GC=F", "CL=F"
]

TIMEFRAME_OPTIONS = ["2m", "5m", "15m", "30m", "60m", "1d"]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names for consistency.
    - Lowercase
    - Replace spaces with underscores
    - Remove special characters
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[ $()]", "", regex=True)
        .str.replace(" ", "_", regex=True)
    )
    return df


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes for download."""
    df = clean_column_names(df)
    return df.to_csv(index=False).encode("utf-8")


# =============================================================================
# DATA FETCHING
# =============================================================================

@st.cache_data(ttl=300)
def fetch_daily_data(
    ticker: str,
    end_date: date
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Fetch previous day's OHLC data.
    
    Returns:
        Tuple of (prev_close, prev_high, prev_low)
    """
    try:
        daily_data = yf.download(
            ticker,
            end=end_date,
            interval="1d",
            progress=False,
            threads=False
        )
        
        if daily_data.empty:
            return None, None, None
            
        # Handle multi-index columns
        if isinstance(daily_data.columns, pd.MultiIndex):
            daily_data.columns = daily_data.columns.map(
                lambda x: x[0] if isinstance(x, tuple) else x
            )
        
        prev_close = float(daily_data["Close"].iloc[-1])
        prev_high = float(daily_data["High"].iloc[-1])
        prev_low = float(daily_data["Low"].iloc[-1])
        
        return prev_close, prev_high, prev_low
        
    except Exception as e:
        logger.error(f"Error fetching daily data for {ticker}: {e}")
        return None, None, None


@st.cache_data(ttl=300)
def fetch_intraday_data(
    ticker: str,
    start_date: date,
    end_date: date,
    timeframe: str
) -> Optional[pd.DataFrame]:
    """
    Fetch intraday data for the specified ticker and timeframe.
    
    Returns:
        DataFrame with intraday data or None if fetch fails
    """
    try:
        intraday = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=timeframe,
            progress=False
        )
        
        if intraday.empty:
            return None
        
        intraday.reset_index(inplace=True)
        
        # Handle multi-index columns
        if isinstance(intraday.columns, pd.MultiIndex):
            intraday.columns = intraday.columns.map(
                lambda x: x[0] if isinstance(x, tuple) else x
            )
        
        # Standardize datetime column name
        if "Datetime" in intraday.columns:
            intraday.rename(columns={"Datetime": "Date"}, inplace=True)
        
        # Convert to New York time
        if intraday["Date"].dtype == "datetime64[ns]":
            intraday["Date"] = (
                intraday["Date"]
                .dt.tz_localize("UTC")
                .dt.tz_convert("America/New_York")
            )
        else:
            intraday["Date"] = (
                intraday["Date"]
                .dt.tz_convert("America/New_York")
            )
        
        intraday["Date"] = intraday["Date"].dt.tz_localize(None)
        
        # Add Time column (12-hour format)
        intraday["Time"] = intraday["Date"].dt.strftime("%I:%M %p")
        
        # Keep only date in Date column
        intraday["Date"] = intraday["Date"].dt.strftime("%Y-%m-%d")
        
        # Add Range column
        intraday["Range"] = intraday["High"] - intraday["Low"]
        
        return intraday
        
    except Exception as e:
        logger.error(f"Error fetching intraday data for {ticker}: {e}")
        return None


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def calculate_f_percentage(
    df: pd.DataFrame,
    prev_close: float,
    col_name: str = "F_numeric"
) -> pd.DataFrame:
    """
    Calculate F% (percentage change from previous close in basis points).
    
    Args:
        df: DataFrame with Close prices
        prev_close: Previous day's closing price
        col_name: Name for the output column
        
    Returns:
        DataFrame with F% column added
    """
    df = df.copy()
    if prev_close and prev_close != 0:
        df[col_name] = ((df["Close"] - prev_close) / prev_close) * 10000
        df["F%"] = df[col_name].round(0).astype(int).astype(str) + "%"
    else:
        df[col_name] = 0
        df["F%"] = "N/A"
    return df


def calculate_rvol(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Calculate Relative Volume (RVOL).
    
    Args:
        df: DataFrame with Volume column
        window: Rolling window size
        
    Returns:
        DataFrame with RVOL_5 column added
    """
    df = df.copy()
    if len(df) >= window and "Volume" in df.columns:
        avg_vol = df["Volume"].rolling(window=window).mean()
        df["RVOL_5"] = safe_div(df["Volume"], avg_vol, default=np.nan)
    else:
        df["RVOL_5"] = np.nan
    return df


def calculate_bollinger_bands(
    df: pd.DataFrame,
    price_col: str = "F_numeric",
    window: int = 20,
    num_std: float = 2.0
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands for F%.
    
    Args:
        df: DataFrame with price data
        price_col: Column to calculate bands for
        window: Rolling window size
        num_std: Number of standard deviations
        
    Returns:
        DataFrame with Bollinger Band columns added
    """
    df = df.copy()
    if price_col in df.columns:
        df["F% MA"] = df[price_col].rolling(window=window, min_periods=1).mean()
        df["F% Std"] = df[price_col].rolling(window=window, min_periods=1).std()
        df["F% Upper"] = df["F% MA"] + (num_std * df["F% Std"])
        df["F% Lower"] = df["F% MA"] - (num_std * df["F% Std"])
    return df


def calculate_ichimoku(
    df: pd.DataFrame,
    prev_close: float,
    tenkan_period: int = 9,
    kijun_period: int = 26
) -> pd.DataFrame:
    """
    Calculate Ichimoku Cloud indicators in F% space.
    
    Args:
        df: DataFrame with OHLC data
        prev_close: Previous day's close for F% conversion
        tenkan_period: Tenkan-sen period
        kijun_period: Kijun-sen period
        
    Returns:
        DataFrame with Ichimoku indicators added
    """
    df = df.copy()
    
    # Tenkan-sen (Conversion Line)
    tenkan_high = df["High"].rolling(window=tenkan_period, min_periods=1).max()
    tenkan_low = df["Low"].rolling(window=tenkan_period, min_periods=1).min()
    df["Tenkan_sen"] = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line)
    kijun_high = df["High"].rolling(window=kijun_period, min_periods=1).max()
    kijun_low = df["Low"].rolling(window=kijun_period, min_periods=1).min()
    df["Kijun_sen"] = (kijun_high + kijun_low) / 2
    
    # Convert to F% scale
    if prev_close and prev_close != 0:
        df["Tenkan_F"] = ((df["Tenkan_sen"] - prev_close) / prev_close) * 10000
        df["Kijun_F"] = ((df["Kijun_sen"] - prev_close) / prev_close) * 10000
    else:
        df["Tenkan_F"] = 0
        df["Kijun_F"] = 0
    
    return df


# =============================================================================
# MARKET PROFILE
# =============================================================================

def compute_value_area(
    df: pd.DataFrame,
    price_col: str = "F_numeric",
    target_bins: int = 20,
    min_bin_width: float = 0.5
) -> Tuple[float, float, pd.DataFrame]:
    """
    Compute Market Profile Value Area.
    
    Args:
        df: DataFrame with price and time data
        price_col: Column containing price data
        target_bins: Target number of price bins
        min_bin_width: Minimum bin width
        
    Returns:
        Tuple of (va_min, va_max, profile_df)
    """
    import string
    
    if price_col not in df.columns:
        raise ValueError(f"Column {price_col} not found in DataFrame")
    
    # Create price bins
    lo, hi = df[price_col].min(), df[price_col].max()
    price_range = max(hi - lo, 1e-6)
    step = max(price_range / target_bins, min_bin_width)
    f_bins = np.arange(lo - step, hi + step, step)
    
    df = df.copy()
    df["F_Bin"] = pd.cut(
        df[price_col],
        bins=f_bins,
        labels=[str(x) for x in f_bins[:-1]]
    )
    
    # Letter assignment (15-min intervals)
    if "Time" in df.columns:
        df = df[df["Time"].notna()]
        df["TimeIndex"] = pd.to_datetime(
            df["Time"],
            format="%I:%M %p",
            errors="coerce"
        )
        df = df[df["TimeIndex"].notna()]
        df["LetterIndex"] = (
            (df["TimeIndex"].dt.hour * 60 + df["TimeIndex"].dt.minute") // 15
        ).astype(int)
        df["LetterIndex"] -= df["LetterIndex"].min()
        
        letters = string.ascii_uppercase
        df["Letter"] = df["LetterIndex"].apply(
            lambda n: letters[n] if n < 26
            else letters[(n // 26) - 1] + letters[n % 26]
        )
    else:
        df["Letter"] = "X"
    
    # Build Market Profile
    profile = {}
    for b in f_bins[:-1]:
        key = str(b)
        lets = df.loc[df["F_Bin"] == key, "Letter"].dropna().unique()
        if len(lets):
            profile[key] = "".join(sorted(lets))
    
    profile_df = pd.DataFrame(
        profile.items(),
        columns=["F% Level", "Letters"]
    ).astype({"F% Level": float})
    profile_df["Letter_Count"] = profile_df["Letters"].str.len().fillna(0)
    
    # Calculate 70% Value Area
    total = profile_df["Letter_Count"].sum()
    target = total * 0.7
    poc_sorted = profile_df.sort_values("Letter_Count", ascending=False)
    
    cumulative = 0
    va_levels = []
    for _, row in poc_sorted.iterrows():
        cumulative += row["Letter_Count"]
        va_levels.append(row["F% Level"])
        if cumulative >= target:
            break
    
    va_min = min(va_levels) if va_levels else 0
    va_max = max(va_levels) if va_levels else 0
    
    return va_min, va_max, profile_df


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_f_percent_plot(
    df: pd.DataFrame,
    prev_close: float,
    prev_high: float,
    prev_low: float,
    ticker: str
) -> go.Figure:
    """
    Create main F% visualization plot.
    
    Args:
        df: DataFrame with processed indicators
        prev_close: Previous day's close
        prev_high: Previous day's high
        prev_low: Previous day's low
        ticker: Ticker symbol
        
    Returns:
        Plotly figure object
    """
    fig = make_subplots(
        rows=1,
        cols=1,
        vertical_spacing=0.03,
        shared_xaxes=True
    )
    
    # Main F% line
    scatter_f = go.Scatter(
        x=df["Time"],
        y=df["F_numeric"],
        mode="lines+markers",
        customdata=df["Close"],
        line=dict(color="#57c7ff", width=1),
        hovertemplate="Time: %{x}<br>F%: %{y:.2f}<br>Close: $%{customdata:.2f}<extra></extra>",
        name="F%"
    )
    fig.add_trace(scatter_f, row=1, col=1)
    
    # Zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        row=1, col=1,
        annotation_text="0%",
        annotation_position="top left"
    )
    
    # Bollinger Bands
    if all(col in df.columns for col in ["F% Upper", "F% Lower", "F% MA"]):
        fig.add_trace(go.Scatter(
            x=df["Time"],
            y=df["F% Upper"],
            mode="lines",
            line=dict(dash="solid", color="#d3d3d3", width=1),
            name="Upper Band"
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df["Time"],
            y=df["F% Lower"],
            mode="lines",
            line=dict(dash="solid", color="#d3d3d3", width=1),
            name="Lower Band"
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df["Time"],
            y=df["F% MA"],
            mode="lines",
            line=dict(dash="dash", color="#d3d3d3", width=2),
            name="Middle Band"
        ), row=1, col=1)
    
    # Ichimoku lines
    if "Kijun_F" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Time"],
            y=df["Kijun_F"],
            mode="lines",
            line=dict(color="#2ECC71", width=1.4),
            name="Kijun (F%)"
        ), row=1, col=1)
    
    if "Tenkan_F" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Time"],
            y=df["Tenkan_F"],
            mode="lines",
            line=dict(color="#E63946", width=0.6, dash="solid"),
            name="Tenkan (F%)"
        ), row=1, col=1)
    
    # RVOL markers
    if "RVOL_5" in df.columns:
        mask_extreme = df["RVOL_5"] > 1.8
        mask_strong = (df["RVOL_5"] >= 1.5) & (df["RVOL_5"] < 1.8)
        
        if mask_extreme.any():
            fig.add_trace(go.Scatter(
                x=df.loc[mask_extreme, "Time"],
                y=df.loc[mask_extreme, "F_numeric"] + 3,
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="red"),
                name="RVOL > 1.8",
                hovertemplate="Time: %{x}<br>F%: %{y}<br>Extreme Volume"
            ), row=1, col=1)
        
        if mask_strong.any():
            fig.add_trace(go.Scatter(
                x=df.loc[mask_strong, "Time"],
                y=df.loc[mask_strong, "F_numeric"] + 3,
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="orange"),
                name="RVOL 1.5-1.79",
                hovertemplate="Time: %{x}<br>F%: %{y}<br>Strong Volume"
            ), row=1, col=1)
    
    # Layout
    fig.update_layout(
        title=f"{ticker} - F% Analysis",
        margin=dict(l=30, r=30, t=50, b=30),
        height=800,
        showlegend=True,
        hovermode="x unified"
    )
    
    return fig


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    """Main application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title="Volmike.com",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("VOLMIKE.COM - Market Analysis Dashboard")
    
    # Sidebar controls
    st.sidebar.header("Input Options")
    
    tickers = st.sidebar.multiselect(
        "Select Tickers",
        options=DEFAULT_TICKERS,
        default=["NVDA"]
    )
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=date(2025, 10, 1)
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=date.today()
    )
    
    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        options=TIMEFRAME_OPTIONS,
        index=1
    )
    
    gap_threshold = st.sidebar.slider(
        "Gap Threshold (%)",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.1
    )
    
    # Run analysis button
    if st.sidebar.button("Run Analysis"):
        
        if not tickers:
            st.warning("Please select at least one ticker.")
            return
        
        # Create tabs for each ticker
        ticker_tabs = st.tabs([f"📊 {t}" for t in tickers])
        
        for idx, ticker in enumerate(tickers):
            with ticker_tabs[idx]:
                
                st.subheader(f"Analysis for {ticker}")
                
                # Fetch daily data
                with st.spinner(f"Fetching daily data for {ticker}..."):
                    prev_close, prev_high, prev_low = fetch_daily_data(
                        ticker,
                        start_date
                    )
                
                if prev_close is None:
                    st.error(f"Could not fetch daily data for {ticker}")
                    continue
                
                # Display previous day stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Previous Close", f"${prev_close:.2f}")
                with col2:
                    st.metric("Previous High", f"${prev_high:.2f}")
                with col3:
                    st.metric("Previous Low", f"${prev_low:.2f}")
                
                # Fetch intraday data
                with st.spinner(f"Fetching intraday data for {ticker}..."):
                    intraday = fetch_intraday_data(
                        ticker,
                        start_date,
                        end_date,
                        timeframe
                    )
                
                if intraday is None or intraday.empty:
                    st.error(f"No intraday data available for {ticker}")
                    continue
                
                # Calculate indicators
                with st.spinner("Calculating indicators..."):
                    intraday = calculate_f_percentage(intraday, prev_close)
                    intraday = calculate_rvol(intraday)
                    intraday = calculate_bollinger_bands(intraday)
                    intraday = calculate_ichimoku(intraday, prev_close)
                
                # Create visualization
                fig = create_f_percent_plot(
                    intraday,
                    prev_close,
                    prev_high,
                    prev_low,
                    ticker
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                with st.expander("📊 View Data Table", expanded=False):
                    display_cols = [
                        "Time", "Close", "F_numeric", "F%", "Range",
                        "Volume", "RVOL_5", "Kijun_F", "Tenkan_F"
                    ]
                    available_cols = [c for c in display_cols if c in intraday.columns]
                    st.dataframe(
                        intraday[available_cols],
                        use_container_width=True
                    )
                
                # Download data
                csv_bytes = to_csv_bytes(intraday)
                csv_b64 = base64.b64encode(csv_bytes).decode("utf-8")
                st.markdown(
                    f'<a href="data:text/csv;base64,{csv_b64}" '
                    f'download="{ticker}_data.csv">⬇️ Download Data (CSV)</a>',
                    unsafe_allow_html=True
                )


if __name__ == "__main__":
    main()