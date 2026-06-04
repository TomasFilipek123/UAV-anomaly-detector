"""
Wstrzykiwanie anomalii do telemetrii drona.

Każda funkcja przyjmuje "czysty" DataFrame i zwraca jego kopię z wstrzykniętą
anomalią oraz kolumną `is_anomaly` (ground truth do ewaluacji).

Scenariusze:
  1. engine_failure   — gwałtowny spadek wysokości (awaria silnika)
  2. gps_spoofing     — skokowa zmiana kursu o ~180 stopni
  3. battery_drain    — nienaturalnie szybki spadek baterii (potencjalny cyberatak)
  4. control_freeze   — zacięcie sterów: parametry "zamarzają" na chwilę
  5. sensor_jamming   — silny szum w danych (zakłócanie sygnału)
"""

from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


def _ensure_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje kolumnę is_anomaly=False jeśli jej nie ma."""
    df = df.copy()
    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = False
        df["anomaly_type"] = "none"
    return df


def inject_engine_failure(
    df: pd.DataFrame,
    start_s: int = 300,
    duration_s: int = 15,
    drop_rate: float = 4.0,
) -> pd.DataFrame:
    """
    Awaria silnika: wysokość gwałtownie spada, prędkość spada do zera.

    drop_rate : m/s o ile wysokość spada na sekundę
    """
    df = _ensure_label_column(df)
    end_s = start_s + duration_s
    mask = (df["timestamp"] >= start_s) & (df["timestamp"] < end_s)
    idx = df.index[mask]

    # Wysokość spada liniowo od aktualnej wartości
    start_alt = df.loc[idx[0], "altitude_m"]
    seconds_into = df.loc[idx, "timestamp"].values - start_s
    df.loc[idx, "altitude_m"] = np.maximum(start_alt - drop_rate * seconds_into, 0)

    # Prędkość spada do prawie zera
    df.loc[idx, "speed_mps"] = df.loc[idx, "speed_mps"] * 0.1

    df.loc[idx, "is_anomaly"] = True
    df.loc[idx, "anomaly_type"] = "engine_failure"
    return df


def inject_gps_spoofing(
    df: pd.DataFrame,
    start_s: int = 200,
    duration_s: int = 20,
    heading_jump_deg: float = 170.0,
) -> pd.DataFrame:
    """
    GPS spoofing: kurs skokowo zmienia się o ~180 stopni.
    To anomalia typu "krok" — wartość się przesuwa, nie wraca.
    """
    df = _ensure_label_column(df)
    end_s = start_s + duration_s
    mask = (df["timestamp"] >= start_s) & (df["timestamp"] < end_s)
    idx = df.index[mask]

    df.loc[idx, "heading_deg"] = (df.loc[idx, "heading_deg"] + heading_jump_deg) % 360

    df.loc[idx, "is_anomaly"] = True
    df.loc[idx, "anomaly_type"] = "gps_spoofing"
    return df


def inject_battery_drain(
    df: pd.DataFrame,
    start_s: int = 400,
    duration_s: int = 30,
    drain_rate_pct: float = 1.5,
) -> pd.DataFrame:
    """
    Nienaturalnie szybki spadek baterii (możliwy cyberatak na BMS).

    drain_rate_pct : ile % traci bateria na sekundę (normalnie ~0.125 %/s)
    """
    df = _ensure_label_column(df)
    end_s = start_s + duration_s
    mask = (df["timestamp"] >= start_s) & (df["timestamp"] < end_s)
    idx = df.index[mask]

    start_batt = df.loc[idx[0], "battery_pct"]
    seconds_into = df.loc[idx, "timestamp"].values - start_s
    new_batt = start_batt - drain_rate_pct * seconds_into
    df.loc[idx, "battery_pct"] = np.maximum(new_batt, 0)

    # Wpływa też na wszystko po anomalii — bateria jest niżej do końca
    after_mask = df["timestamp"] >= end_s
    after_idx = df.index[after_mask]
    if len(after_idx) > 0:
        # Przesuwamy baterię w dół o całkowity dodatkowy ubytek
        extra_drain = drain_rate_pct * duration_s - 0.125 * duration_s
        df.loc[after_idx, "battery_pct"] = np.maximum(
            df.loc[after_idx, "battery_pct"] - extra_drain, 0
        )

    df.loc[idx, "is_anomaly"] = True
    df.loc[idx, "anomaly_type"] = "battery_drain"
    return df


def inject_control_freeze(
    df: pd.DataFrame,
    start_s: int = 350,
    duration_s: int = 10,
) -> pd.DataFrame:
    """
    Zacięcie sterów: kurs i prędkość "zamarzają" — żadnej zmiany przez okres.
    Wysokość zachowuje się normalnie (bo to akurat działa).
    """
    df = _ensure_label_column(df)
    end_s = start_s + duration_s
    mask = (df["timestamp"] >= start_s) & (df["timestamp"] < end_s)
    idx = df.index[mask]

    frozen_heading = df.loc[idx[0], "heading_deg"]
    frozen_speed = df.loc[idx[0], "speed_mps"]
    df.loc[idx, "heading_deg"] = frozen_heading
    df.loc[idx, "speed_mps"] = frozen_speed

    df.loc[idx, "is_anomaly"] = True
    df.loc[idx, "anomaly_type"] = "control_freeze"
    return df


def inject_sensor_jamming(
    df: pd.DataFrame,
    start_s: int = 450,
    duration_s: int = 12,
    noise_amplitude: float = 8.0,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Zakłócanie sygnału: silny szum nakłada się na wszystkie sensory.
    """
    rng = np.random.default_rng(seed)
    df = _ensure_label_column(df)
    end_s = start_s + duration_s
    mask = (df["timestamp"] >= start_s) & (df["timestamp"] < end_s)
    idx = df.index[mask]
    n = len(idx)

    df.loc[idx, "altitude_m"] += rng.normal(0, noise_amplitude, n)
    df.loc[idx, "speed_mps"] += rng.normal(0, noise_amplitude * 0.3, n)
    df.loc[idx, "speed_mps"] = df.loc[idx, "speed_mps"].clip(lower=0)
    df.loc[idx, "heading_deg"] = (
        df.loc[idx, "heading_deg"] + rng.normal(0, noise_amplitude * 4, n)
    ) % 360

    df.loc[idx, "is_anomaly"] = True
    df.loc[idx, "anomaly_type"] = "sensor_jamming"
    return df


# Rejestr scenariuszy — wygodne do ewaluacji
SCENARIOS = {
    "engine_failure": inject_engine_failure,
    "gps_spoofing": inject_gps_spoofing,
    "battery_drain": inject_battery_drain,
    "control_freeze": inject_control_freeze,
    "sensor_jamming": inject_sensor_jamming,
}


if __name__ == "__main__":
    from generate import generate_normal_flight

    df = generate_normal_flight()
    # Wstrzykujemy wszystkie 5 scenariuszy w różnych momentach lotu
    df = inject_gps_spoofing(df, start_s=180, duration_s=20)
    df = inject_engine_failure(df, start_s=280, duration_s=12)
    df = inject_control_freeze(df, start_s=340, duration_s=10)
    df = inject_battery_drain(df, start_s=400, duration_s=25)
    df = inject_sensor_jamming(df, start_s=460, duration_s=12)

    print(f"Łącznie próbek: {len(df)}")
    print(f"Próbek anomalnych: {df['is_anomaly'].sum()}")
    print(f"\nLiczba próbek per typ anomalii:")
    print(df["anomaly_type"].value_counts())

    output_path = SCRIPT_DIR / "flight_with_anomalies.csv"
    df.to_csv(output_path, index=False)
    print(f"\nZapisano: {output_path}")
