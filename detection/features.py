from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import circular_diff, ensure_numeric_columns, haversine_step

NUMERIC_INPUT_COLS = ["altitude", "speed", "heading", "latitude", "longitude"]


def _features_one_case(case_df: pd.DataFrame, window: int) -> pd.DataFrame:
    feats = pd.DataFrame(index=case_df.index)
    min_p = max(2, window // 2)

    for col, prefix in [("altitude", "alt"), ("speed", "spd")]:
        s = case_df[col]
        roll = s.rolling(window=window, min_periods=min_p)
        feats[f"{prefix}_mean"] = roll.mean()
        feats[f"{prefix}_std"] = roll.std()
        feats[f"{prefix}_range"] = roll.max() - roll.min()
        feats[f"{prefix}_slope"] = (s - s.shift(window - 1)) / window

    hdg_diff = circular_diff(case_df["heading"])
    feats["hdg_diff_mean"] = hdg_diff.rolling(window=window, min_periods=min_p).mean()
    feats["hdg_diff_std"] = hdg_diff.rolling(window=window, min_periods=min_p).std()
    feats["hdg_diff_abs_max"] = hdg_diff.abs().rolling(window=window, min_periods=min_p).max()

    gps_step = pd.Series(
        haversine_step(case_df["latitude"].values, case_df["longitude"].values),
        index=case_df.index,
    )
    feats["gps_step_mean"] = gps_step.rolling(window=window, min_periods=min_p).mean()
    feats["gps_step_std"] = gps_step.rolling(window=window, min_periods=min_p).std()
    feats["gps_step_max"] = gps_step.rolling(window=window, min_periods=min_p).max()

    dt_raw = case_df["timestamp"].diff().dt.total_seconds()
    dt = dt_raw.replace(0, np.nan)

    gps_speed = gps_step / dt
    feats["speed_vs_gps"] = (case_df["speed"] - gps_speed).abs()
    feats["speed_vs_gps"] = feats["speed_vs_gps"].rolling(window=window, min_periods=min_p).mean()

    dt_roll = dt_raw.rolling(window=window, min_periods=min_p)
    feats["dt_mean"] = dt_roll.mean()
    feats["dt_std"] = dt_roll.std()
    feats["dt_max"] = dt_roll.max()
    feats["dt_min"] = dt_roll.min()

    feats["sample_var_score"] = (
        feats["alt_std"].fillna(0)
        + feats["spd_std"].fillna(0)
        + feats["hdg_diff_std"].fillna(0)
        + feats["gps_step_std"].fillna(0)
    )

    return feats


def compute_features(df: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    df = ensure_numeric_columns(df, NUMERIC_INPUT_COLS)

    parts = [
        _features_one_case(case_df, window)
        for _, case_df in df.groupby("case_id", sort=False)
    ]
    feats = pd.concat(parts).sort_index()
    return feats.bfill().fillna(0)
