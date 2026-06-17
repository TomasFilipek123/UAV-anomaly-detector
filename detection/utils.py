from __future__ import annotations

import numpy as np
import pandas as pd


def ensure_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def circular_diff(series: pd.Series) -> pd.Series:
    raw = series.diff()
    return ((raw + 180) % 360) - 180


def haversine_step(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    R = 6_371_000.0
    lat1 = np.radians(lat[:-1])
    lat2 = np.radians(lat[1:])
    dlat = lat2 - lat1
    dlon = np.radians(lon[1:] - lon[:-1])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    d = 2 * R * np.arcsin(np.sqrt(a))
    return np.concatenate([[np.nan], d])


def rolling_by_case(
    series: pd.Series,
    case_id: pd.Series,
    window: int,
    op: str,
    min_periods: int | None = None,
) -> pd.Series:
    if min_periods is None:
        min_periods = max(1, window // 2)
    grouped = series.groupby(case_id, sort=False, group_keys=False)
    if op == "mean":
        return grouped.transform(lambda s: s.rolling(window=window, min_periods=min_periods).mean())
    if op == "std":
        return grouped.transform(lambda s: s.rolling(window=window, min_periods=min_periods).std())
    if op == "max":
        return grouped.transform(lambda s: s.rolling(window=window, min_periods=min_periods).max())
    if op == "min":
        return grouped.transform(lambda s: s.rolling(window=window, min_periods=min_periods).min())
    raise ValueError(f"Unknown rolling operation: {op}")
