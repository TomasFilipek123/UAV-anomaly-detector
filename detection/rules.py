"""
Warstwa 1: detekcja przez progi fizyczne (rule-based).

Najprostsza i najszybsza warstwa. Łapie oczywiste naruszenia:
  - bateria poniżej krytycznego poziomu
  - wysokość ujemna albo absurdalnie wysoka
  - prędkość przekraczająca maksimum drona
  - kurs poza [0, 360)

To jest baseline — warstwa 2 i 3 muszą być lepsze niż to.
"""

import pandas as pd


# Domyślne progi (do dostrojenia per dron)
DEFAULT_THRESHOLDS = {
    "altitude_min_m": -1.0,       # poniżej zera = błąd sensora lub crash
    "altitude_max_m": 120.0,      # limit zgodnie z przepisami UAV w PL
    "speed_max_mps": 20.0,        # typowe maks. dla małego drona
    "battery_critical_pct": 15.0, # krytyczny poziom baterii
}


def detect_threshold_violations(
    df: pd.DataFrame,
    thresholds: dict = None,
) -> pd.DataFrame:
    """
    Aplikuje progi fizyczne i zwraca DataFrame z kolumnami detekcji.

    Zwraca kopię df z dodatkowymi kolumnami:
      - alert_threshold   : bool, czy jakikolwiek próg naruszony
      - alert_reasons     : str, lista naruszonych progów oddzielona '|'
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    df = df.copy()
    reasons = [[] for _ in range(len(df))]

    # Wysokość
    too_low = df["altitude_m"] < thresholds["altitude_min_m"]
    too_high = df["altitude_m"] > thresholds["altitude_max_m"]
    for i in df.index[too_low]:
        reasons[df.index.get_loc(i)].append("altitude_below_min")
    for i in df.index[too_high]:
        reasons[df.index.get_loc(i)].append("altitude_above_max")

    # Prędkość
    too_fast = df["speed_mps"] > thresholds["speed_max_mps"]
    for i in df.index[too_fast]:
        reasons[df.index.get_loc(i)].append("speed_above_max")

    # Bateria
    low_batt = df["battery_pct"] < thresholds["battery_critical_pct"]
    for i in df.index[low_batt]:
        reasons[df.index.get_loc(i)].append("battery_critical")

    df["alert_threshold"] = [len(r) > 0 for r in reasons]
    df["alert_reasons"] = ["|".join(r) if r else "" for r in reasons]
    return df


if __name__ == "__main__":
    from pathlib import Path
    csv_path = Path(__file__).resolve().parent.parent / "data" / "flight_with_anomalies.csv"
    df = pd.read_csv(csv_path)
    result = detect_threshold_violations(df)

    n_alerts = result["alert_threshold"].sum()
    print(f"Próbek z alertem progowym: {n_alerts}/{len(result)}")
    print(f"\nPrzykładowe alerty:")
    alerts = result[result["alert_threshold"]].head(10)
    print(alerts[["timestamp", "altitude_m", "speed_mps", "battery_pct", "alert_reasons"]])
