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

from .utils import circular_diff, ensure_numeric_columns, haversine_step, rolling_by_case


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


def _per_case_diffs(df: pd.DataFrame) -> dict[str, pd.Series]:
    g = df.groupby("case_id", sort=False, group_keys=False)
    diffs = {
        "altitude": g["altitude"].diff(),
        "speed": g["speed"].diff(),
        "heading": g["heading"].apply(circular_diff),
    }
    # GPS step liczymy per case. Petla zamiast g.apply(): apply() przy POJEDYNCZYM
    # case_id (np. bufor strumieniowy z jednym lotem) zwija wynik do dlugosci 1
    # zamiast N i nie tworzy MultiIndex -> blad "No objects to concatenate" w
    # kolejnym rolling. Petla daje identyczny wynik dla 1 i wielu grup.
    gps_step = pd.Series(np.nan, index=df.index, dtype=float)
    for _, idx in g.groups.items():
        sub = df.loc[idx]
        gps_step.loc[idx] = haversine_step(sub["latitude"].values, sub["longitude"].values)
    diffs["gps_step"] = gps_step
    return diffs


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
    df = ensure_numeric_columns(df, ["altitude", "speed", "heading", "latitude", "longitude"])

    w = config["window"]
    th = config["z_threshold"]
    min_std = config["min_std"]

    diffs = _per_case_diffs(df)

    z_scores = {}
    for name, diff in diffs.items():
        rmean = rolling_by_case(diff, df["case_id"], w, "mean")
        rstd = rolling_by_case(diff, df["case_id"], w, "std").clip(lower=min_std)
        z = (diff - rmean) / rstd
        z_scores[name] = z
        df[f"z_{name}"] = z

    reason_df = pd.DataFrame(
        {f"sudden_{name}": z.abs().gt(th).fillna(False) for name, z in z_scores.items()},
        index=df.index,
    )

    fw = config["freeze_window"]
    speed_std = rolling_by_case(df["speed"], df["case_id"], fw, "std")
    heading_std = rolling_by_case(df["heading"], df["case_id"], fw, "std")
    gps_std = rolling_by_case(diffs["gps_step"], df["case_id"], fw, "std")

    reason_df["freeze_detected"] = (
        (speed_std < config["freeze_speed_std"])
        & (heading_std < config["freeze_heading_std"])
        & (gps_std < config["freeze_gps_std"])
    ).fillna(False)

    df["alert_change"] = reason_df.any(axis=1)
    df["change_reasons"] = reason_df.apply(
        lambda row: "|".join(row.index[row].tolist()), axis=1
    )
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
