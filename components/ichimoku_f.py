# components/ichimoku_f.py
from __future__ import annotations

import numpy as np
import pandas as pd


def add_tenkan_kijun_f(
    df: pd.DataFrame,
    *,
    high_col: str = "High",
    low_col: str = "Low",
    prev_close_col: str = "prev_close",
    group_col: str = "trading_date",
    tenkan_period: int = 9,
    kijun_period: int = 26,
    tenkan_name: str = "tenkan_f",
    kijun_name: str = "kijun_f",
) -> pd.DataFrame:
    """
    Adds Tenkan/Kijun in F-space (bps displacement from prev close):
      F = ((value - prev_close) / prev_close) * 10000

    By default, computes per trading day (group_col) so periods don't bleed across days.
    Requires High/Low and prev_close columns.
    """
    if df is None or df.empty:
        return df

    d = df.copy()

    if high_col not in d.columns or low_col not in d.columns:
        raise ValueError(f"Missing required columns: {high_col}/{low_col}")

    if prev_close_col not in d.columns:
        raise ValueError(f"Missing required column: {prev_close_col}")

    def _calc(g: pd.DataFrame) -> pd.DataFrame:
        hi = pd.to_numeric(g[high_col], errors="coerce").astype(float)
        lo = pd.to_numeric(g[low_col], errors="coerce").astype(float)
        prev = pd.to_numeric(g[prev_close_col], errors="coerce").astype(float)

        ten_hi = hi.rolling(tenkan_period, min_periods=tenkan_period).max()
        ten_lo = lo.rolling(tenkan_period, min_periods=tenkan_period).min()
        ten_px = (ten_hi + ten_lo) / 2.0

        kij_hi = hi.rolling(kijun_period, min_periods=kijun_period).max()
        kij_lo = lo.rolling(kijun_period, min_periods=kijun_period).min()
        kij_px = (kij_hi + kij_lo) / 2.0

        with np.errstate(divide="ignore", invalid="ignore"):
            ten_f = ((ten_px - prev) / prev) * 10_000.0
            kij_f = ((kij_px - prev) / prev) * 10_000.0

        g[tenkan_name] = pd.Series(ten_f).replace([np.inf, -np.inf], np.nan)
        g[kijun_name] = pd.Series(kij_f).replace([np.inf, -np.inf], np.nan)
        return g

    if group_col and group_col in d.columns:
        return d.groupby(group_col, group_keys=False).apply(_calc)

    return _calc(d)
