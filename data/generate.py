"""
Generator syntetycznej telemetrii drona.

Symuluje typowy lot:
  1. Faza wznoszenia (takeoff)  — wysokość rośnie liniowo
  2. Faza przelotu (cruise)     — wysokość stała, kurs się zmienia
  3. Faza lądowania (landing)   — wysokość spada liniowo

Wszystkie parametry mają realistyczny szum gaussowski.
Bateria rozładowuje się liniowo przez cały lot.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Katalog tego pliku — używamy ścieżek względnych, działa na każdym OS
SCRIPT_DIR = Path(__file__).resolve().parent


def generate_normal_flight(
    duration_s: int = 600,
    sample_rate_hz: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generuje normalny lot drona bez anomalii.

    Parametry
    ---------
    duration_s : długość lotu w sekundach (domyślnie 10 minut)
    sample_rate_hz : częstotliwość próbkowania (domyślnie 1 Hz)
    seed : ziarno dla powtarzalności

    Zwraca
    ------
    DataFrame z kolumnami:
      timestamp, altitude_m, speed_mps, heading_deg, battery_pct, phase
    """
    rng = np.random.default_rng(seed)
    n_samples = duration_s * sample_rate_hz
    t = np.arange(n_samples) / sample_rate_hz  # czas w sekundach

    # --- Definicja faz lotu (proporcje całego lotu) ---
    takeoff_end = int(0.15 * n_samples)   # pierwsze 15% to wznoszenie
    landing_start = int(0.85 * n_samples)  # ostatnie 15% to lądowanie

    # --- WYSOKOŚĆ ---
    cruise_altitude = 80.0  # docelowa wysokość przelotu w metrach
    altitude = np.zeros(n_samples)

    # Wznoszenie: 0 -> cruise_altitude
    altitude[:takeoff_end] = np.linspace(0, cruise_altitude, takeoff_end)
    # Przelot: cruise_altitude ze szumem
    altitude[takeoff_end:landing_start] = cruise_altitude
    # Lądowanie: cruise_altitude -> 0
    altitude[landing_start:] = np.linspace(
        cruise_altitude, 0, n_samples - landing_start
    )
    # Szum wysokościomierza (~±0.5m typowo dla GPS+barometr)
    altitude += rng.normal(0, 0.5, n_samples)

    # --- PRĘDKOŚĆ ---
    cruise_speed = 12.0  # m/s typowo dla małego drona
    speed = np.zeros(n_samples)
    speed[:takeoff_end] = np.linspace(0, cruise_speed, takeoff_end)
    speed[takeoff_end:landing_start] = cruise_speed
    speed[landing_start:] = np.linspace(cruise_speed, 0, n_samples - landing_start)
    speed += rng.normal(0, 0.3, n_samples)
    speed = np.clip(speed, 0, None)  # prędkość nieujemna

    # --- KURS (heading) ---
    # Stopniowe, łagodne zakręty podczas przelotu — kilka zmian kierunku
    heading = np.zeros(n_samples)
    heading[0] = 90.0  # start lecąc na wschód
    # Składamy z kilku sinusoid o różnej częstotliwości — symuluje manewry
    cruise_indices = np.arange(takeoff_end, landing_start)
    heading_changes = (
        20 * np.sin(2 * np.pi * cruise_indices / 200)
        + 15 * np.sin(2 * np.pi * cruise_indices / 80)
    )
    heading[takeoff_end:landing_start] = 90.0 + heading_changes
    heading[:takeoff_end] = 90.0  # podczas startu kurs stały
    heading[landing_start:] = heading[landing_start - 1]  # podczas lądowania kurs stały
    heading += rng.normal(0, 1.0, n_samples)
    heading = heading % 360  # normalizacja do [0, 360)

    # --- BATERIA ---
    # Liniowy spadek od 100% do ~25% przez cały lot (realistyczne zużycie)
    battery = np.linspace(100, 25, n_samples)
    battery += rng.normal(0, 0.2, n_samples)
    battery = np.clip(battery, 0, 100)

    # --- FAZA (etykieta pomocnicza, nie do detekcji) ---
    phase = np.array(["cruise"] * n_samples, dtype=object)
    phase[:takeoff_end] = "takeoff"
    phase[landing_start:] = "landing"

    df = pd.DataFrame({
        "timestamp": t,
        "altitude_m": altitude,
        "speed_mps": speed,
        "heading_deg": heading,
        "battery_pct": battery,
        "phase": phase,
    })
    return df


if __name__ == "__main__":
    df = generate_normal_flight()
    print("Wygenerowano lot — pierwsze 5 próbek:")
    print(df.head())
    print(f"\nKształt: {df.shape}")
    print(f"Czas trwania: {df['timestamp'].max():.0f}s")
    print(f"\nStatystyki opisowe:")
    print(df[["altitude_m", "speed_mps", "heading_deg", "battery_pct"]].describe())

    # Zapis do CSV w katalogu tego skryptu
    output_path = SCRIPT_DIR / "normal_flight.csv"
    df.to_csv(output_path, index=False)
    print(f"\nZapisano: {output_path}")
