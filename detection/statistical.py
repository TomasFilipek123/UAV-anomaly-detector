"""
Warstwa 2: detekcja nagłych zmian przez statystyki kroczące (rolling z-score).

Idea: dla każdego parametru liczymy pochodną (diff między próbkami),
a następnie z-score tej pochodnej w oknie kroczącym.

Z-score > threshold oznacza, że bieżąca zmiana jest nietypowa względem
ostatnich N sekund — czyli mamy "nagłą zmianę".

To jest sedno wymagania z tabeli: "mechanizm wykrywania nagłych zmian".
"""

import numpy as np
import pandas as pd


# Domyślne parametry detektora
DEFAULT_CONFIG = {
    "window_s": 30,         # długość okna dla z-score (sekundy)
    "z_threshold": 3.5,     # próg z-score uznawany za anomalię
    "min_std": 0.05,        # minimalne std żeby uniknąć dzielenia przez ~0
    "freeze_window_s": 6,   # okno do detekcji "freeze" (brak zmienności)
    "freeze_speed_std": 0.05,  # próg std prędkości poniżej którego = zamrożone
    "freeze_heading_std": 0.3,  # próg std kursu poniżej którego = zamrożone
}


def _circular_diff(series: pd.Series) -> pd.Series:
    """
    Pochodna kursu uwzględniająca cykliczność [0, 360).
    Zmiana z 359° na 1° to 2°, nie -358°.
    """
    raw = series.diff()
    # Mapujemy na [-180, 180]
    wrapped = ((raw + 180) % 360) - 180
    return wrapped


def detect_sudden_changes(
    df: pd.DataFrame,
    config: dict = None,
) -> pd.DataFrame:
    """
    Wykrywa nagłe zmiany w altitude, speed, heading, battery.

    Dla każdego parametru:
      1. Liczymy diff (pochodną dyskretną).
      2. Rolling mean i std tej diff w oknie window_s.
      3. Z-score = (diff - rolling_mean) / rolling_std.
      4. Alert jeśli |z-score| > z_threshold.

    Zwraca df z kolumnami:
      - z_<parametr> : wartości z-score
      - alert_change : bool, ogólny alert tej warstwy
      - change_reasons : str, które parametry alarmują
    """
    if config is None:
        config = DEFAULT_CONFIG

    df = df.copy()
    w = config["window_s"]
    th = config["z_threshold"]
    min_std = config["min_std"]

    # Pochodne — uwaga: kurs cyklicznie
    diffs = {
        "altitude": df["altitude_m"].diff(),
        "speed": df["speed_mps"].diff(),
        "heading": _circular_diff(df["heading_deg"]),
        "battery": df["battery_pct"].diff(),
    }

    z_scores = {}
    for name, d in diffs.items():
        rolling_mean = d.rolling(window=w, min_periods=w // 2).mean()
        rolling_std = d.rolling(window=w, min_periods=w // 2).std()
        # Floor na std żeby uniknąć dzielenia przez ~0 (np. kurs stały)
        rolling_std = rolling_std.clip(lower=min_std)
        z = (d - rolling_mean) / rolling_std
        z_scores[name] = z
        df[f"z_{name}"] = z

    # Alert jeśli którykolwiek |z| > threshold
    above_thresh = {name: z.abs() > th for name, z in z_scores.items()}

    df["alert_change"] = False
    reasons_list = [[] for _ in range(len(df))]
    for name, mask in above_thresh.items():
        mask = mask.fillna(False)
        df["alert_change"] = df["alert_change"] | mask
        for i in df.index[mask]:
            reasons_list[df.index.get_loc(i)].append(f"sudden_{name}")

    # --- Detekcja "freeze": nietypowo niska wariancja w krótkim oknie ---
    # To przeciwieństwo nagłej zmiany — parametry "zamarzają", co w normalnym
    # locie się nie zdarza (zawsze jest jakiś szum sensora).
    fw = config["freeze_window_s"]
    speed_rolling_std = df["speed_mps"].rolling(window=fw, min_periods=fw).std()
    heading_rolling_std = df["heading_deg"].rolling(window=fw, min_periods=fw).std()

    freeze_mask = (
        (speed_rolling_std < config["freeze_speed_std"])
        & (heading_rolling_std < config["freeze_heading_std"])
    ).fillna(False)

    df["alert_change"] = df["alert_change"] | freeze_mask
    for i in df.index[freeze_mask]:
        reasons_list[df.index.get_loc(i)].append("freeze_detected")

    df["change_reasons"] = ["|".join(r) if r else "" for r in reasons_list]
    return df


if __name__ == "__main__":
    from pathlib import Path
    csv_path = Path(__file__).resolve().parent.parent / "data" / "flight_with_anomalies.csv"
    df = pd.read_csv(csv_path)
    result = detect_sudden_changes(df)

    n_alerts = result["alert_change"].sum()
    print(f"Próbek z alertem nagłej zmiany: {n_alerts}/{len(result)}")

    print(f"\nPrzykładowe alerty:")
    alerts = result[result["alert_change"]].head(15)
    cols = ["timestamp", "altitude_m", "speed_mps", "heading_deg",
            "battery_pct", "change_reasons", "anomaly_type"]
    print(alerts[cols].to_string())
