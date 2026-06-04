"""
Warstwa 1: detekcja przez progi fizyczne (rule-based).

Najprostsza i najszybsza warstwa. Lapie oczywiste naruszenia zakresow:
  - wysokosc ujemna albo absurdalnie wysoka
  - predkosc przekraczajaca maksimum drona
  - kurs poza [0, 360)
  - GPS poza zakresem geograficznym

To jest baseline - warstwa 2 i 3 musza byc lepsze niz to.
"""

import pandas as pd


DEFAULT_THRESHOLDS = {
    "altitude_min": -5.0,      # ponizej -5 m = blad sensora albo crash
    "altitude_max": 500.0,     # 500 m - typowy gorny zakres dla cywilnych dronow
    "speed_max": 30.0,         # m/s - typowe maks. dla srednich dronow
    "heading_min": 0.0,
    "heading_max": 360.0,
    "latitude_min": -90.0,
    "latitude_max": 90.0,
    "longitude_min": -180.0,
    "longitude_max": 180.0,
}


def detect_threshold_violations(
    df: pd.DataFrame,
    thresholds: dict = None,
) -> pd.DataFrame:
    """
    Aplikuje progi fizyczne i zwraca DataFrame z kolumnami detekcji.

    Dodaje:
      - alert_threshold : bool, czy ktorykolwiek prog naruszony
      - alert_reasons   : str, lista naruszonych progow rozdzielona '|'
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    df = df.copy()
    n = len(df)
    reasons = [[] for _ in range(n)]

    checks = [
        (df["altitude"] < thresholds["altitude_min"], "altitude_below_min"),
        (df["altitude"] > thresholds["altitude_max"], "altitude_above_max"),
        (df["speed"] < 0,                              "speed_negative"),
        (df["speed"] > thresholds["speed_max"],        "speed_above_max"),
        (df["heading"] < thresholds["heading_min"],    "heading_below_min"),
        (df["heading"] >= thresholds["heading_max"],   "heading_above_max"),
        (df["latitude"] < thresholds["latitude_min"],  "latitude_out_of_range"),
        (df["latitude"] > thresholds["latitude_max"],  "latitude_out_of_range"),
        (df["longitude"] < thresholds["longitude_min"], "longitude_out_of_range"),
        (df["longitude"] > thresholds["longitude_max"], "longitude_out_of_range"),
    ]

    for mask, label in checks:
        mask = mask.fillna(False).values
        for pos in range(n):
            if mask[pos]:
                reasons[pos].append(label)

    df["alert_threshold"] = [len(r) > 0 for r in reasons]
    df["alert_reasons"] = ["|".join(r) if r else "" for r in reasons]
    return df


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.loader import load_dataset

    df = load_dataset()
    result = detect_threshold_violations(df)
    n_alerts = result["alert_threshold"].sum()
    print(f"Probek z alertem progowym: {n_alerts:,}/{len(result):,}")
    print(f"\nRozklad alertow vs label:")
    print(pd.crosstab(result["alert_threshold"], result["label"]))
