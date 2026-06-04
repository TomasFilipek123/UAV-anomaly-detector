"""
Warstwa 2: detekcja naglych zmian przez statystyki kroczace (rolling z-score).

Idea: dla kazdego parametru liczymy pochodna (diff miedzy probkami),
a nastepnie z-score tej pochodnej w oknie kroczacym.

|z| > threshold oznacza, ze biezaca zmiana jest nietypowa wzgledem
ostatnich N probek -> "nagla zmiana".

WAZNE: wszystkie operacje (diff, rolling) wykonywane sa per case_id,
zeby nie liczyc roznicy miedzy ostatnia probka jednego lotu i pierwsza
nastepnego (dalo by to ogromne, falszywe skoki).
"""

import numpy as np
import pandas as pd


DEFAULT_CONFIG = {
    "window": 30,              # okno z-score (w probkach, nie sekundach!)
    "z_threshold": 3.5,
    "min_std": 0.05,
    "freeze_window": 6,
    "freeze_speed_std": 0.05,
    "freeze_heading_std": 0.3,
    "freeze_gps_std": 1e-6,    # std lat/lon ponizej tego = brak ruchu GPS
}

PARAM_CHANNELS = ["altitude", "speed", "heading", "gps_step"]


def _circular_diff(series: pd.Series) -> pd.Series:
    """Pochodna kursu z poprawnym wrap-around (359 -> 1 = 2deg, nie -358deg)."""
    raw = series.diff()
    return ((raw + 180) % 360) - 180


def _haversine_step(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Odleglosc Haversine miedzy kolejnymi probkami [m].
    Pierwsza wartosc to NaN (brak poprzednika).
    """
    R = 6_371_000.0  # promien Ziemi [m]
    lat1 = np.radians(lat[:-1])
    lat2 = np.radians(lat[1:])
    dlat = lat2 - lat1
    dlon = np.radians(lon[1:] - lon[:-1])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    d = 2 * R * np.arcsin(np.sqrt(a))
    return np.concatenate([[np.nan], d])


def _per_case_diffs(df: pd.DataFrame) -> dict:
    """Liczy pochodne (z poprawnym groupby case_id) dla wszystkich kanalow."""
    g = df.groupby("case_id", sort=False, group_keys=False)
    diffs = {
        "altitude": g["altitude"].diff(),
        "speed": g["speed"].diff(),
        "heading": g["heading"].apply(_circular_diff),
    }
    # GPS step liczymy per case
    gps_step = g.apply(
        lambda x: pd.Series(_haversine_step(x["latitude"].values, x["longitude"].values),
                             index=x.index)
    )
    diffs["gps_step"] = gps_step.droplevel(0) if isinstance(gps_step.index, pd.MultiIndex) else gps_step
    return diffs


def _per_case_rolling(series: pd.Series, case_id: pd.Series, window: int, op: str):
    """Rolling op (mean / std) wykonywany niezaleznie per case_id."""
    g = series.groupby(case_id, sort=False, group_keys=False)
    if op == "mean":
        return g.transform(lambda s: s.rolling(window=window, min_periods=window // 2).mean())
    if op == "std":
        return g.transform(lambda s: s.rolling(window=window, min_periods=window // 2).std())
    raise ValueError(op)


def detect_sudden_changes(
    df: pd.DataFrame,
    config: dict = None,
) -> pd.DataFrame:
    """
    Wykrywa nagle zmiany w altitude, speed, heading oraz GPS (haversine step).
    Plus drugi mechanizm: detekcja "freeze" (nietypowo niska wariancja).

    Wymaga w df kolumn: case_id, altitude, speed, heading, latitude, longitude.

    Zwraca df z dodanymi kolumnami:
      - z_altitude, z_speed, z_heading, z_gps_step
      - alert_change   : bool, ogolny alert tej warstwy
      - change_reasons : str, ktore parametry alarmuja
    """
    if config is None:
        config = DEFAULT_CONFIG

    df = df.copy()
    w = config["window"]
    th = config["z_threshold"]
    min_std = config["min_std"]

    diffs = _per_case_diffs(df)

    z_scores = {}
    for name, d in diffs.items():
        rmean = _per_case_rolling(d, df["case_id"], w, "mean")
        rstd = _per_case_rolling(d, df["case_id"], w, "std").clip(lower=min_std)
        z = (d - rmean) / rstd
        z_scores[name] = z
        df[f"z_{name}"] = z

    df["alert_change"] = False
    reasons_list = [[] for _ in range(len(df))]
    for name, z in z_scores.items():
        mask = (z.abs() > th).fillna(False).values
        df["alert_change"] = df["alert_change"] | mask
        for pos in range(len(df)):
            if mask[pos]:
                reasons_list[pos].append(f"sudden_{name}")

    # --- Detekcja freeze: nietypowo niska wariancja w krotkim oknie ---
    fw = config["freeze_window"]
    speed_std = _per_case_rolling(df["speed"], df["case_id"], fw, "std")
    heading_std = _per_case_rolling(df["heading"], df["case_id"], fw, "std")
    gps_std = _per_case_rolling(diffs["gps_step"], df["case_id"], fw, "std")

    freeze_mask = (
        (speed_std < config["freeze_speed_std"])
        & (heading_std < config["freeze_heading_std"])
        & (gps_std < config["freeze_gps_std"])
    ).fillna(False).values

    df["alert_change"] = df["alert_change"] | freeze_mask
    for pos in range(len(df)):
        if freeze_mask[pos]:
            reasons_list[pos].append("freeze_detected")

    df["change_reasons"] = ["|".join(r) if r else "" for r in reasons_list]
    return df


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.loader import load_dataset

    df = load_dataset()
    result = detect_sudden_changes(df)
    n_alerts = result["alert_change"].sum()
    print(f"Probek z alertem naglej zmiany: {n_alerts:,}/{len(result):,}")
    print(f"\nRozklad alertow vs label:")
    print(pd.crosstab(result["alert_change"], result["label"]))
