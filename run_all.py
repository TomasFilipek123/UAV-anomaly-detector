"""
Uruchamia cały pipeline w odpowiedniej kolejności:
  1. Generuje normalny lot
  2. Wstrzykuje anomalie
  3. Tworzy wykres z alertami

Uruchom z katalogu projektu:
    python run_all.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.generate import generate_normal_flight
from data.inject_anomalies import (
    inject_gps_spoofing,
    inject_engine_failure,
    inject_control_freeze,
    inject_battery_drain,
    inject_sensor_jamming,
)
from notebooks.visualize import plot_telemetry


def main():
    print("=" * 60)
    print("KROK 1: Generowanie normalnego lotu")
    print("=" * 60)
    df = generate_normal_flight(duration_s=600, sample_rate_hz=1, seed=42)
    normal_path = PROJECT_ROOT / "data" / "normal_flight.csv"
    df.to_csv(normal_path, index=False)
    print(f"Zapisano: {normal_path}")
    print(f"Liczba próbek: {len(df)}")

    print()
    print("=" * 60)
    print("KROK 2: Wstrzykiwanie anomalii")
    print("=" * 60)
    df = inject_gps_spoofing(df, start_s=180, duration_s=20)
    df = inject_engine_failure(df, start_s=280, duration_s=12)
    df = inject_control_freeze(df, start_s=340, duration_s=10)
    df = inject_battery_drain(df, start_s=400, duration_s=25)
    df = inject_sensor_jamming(df, start_s=460, duration_s=12)
    anomaly_path = PROJECT_ROOT / "data" / "flight_with_anomalies.csv"
    df.to_csv(anomaly_path, index=False)
    print(f"Zapisano: {anomaly_path}")
    print(f"Próbek anomalnych: {df['is_anomaly'].sum()}/{len(df)}")
    print("\nLiczba próbek per typ anomalii:")
    print(df["anomaly_type"].value_counts())

    print()
    print("=" * 60)
    print("KROK 3: Generowanie wykresu z detekcją")
    print("=" * 60)
    png_path = PROJECT_ROOT / "data" / "telemetry_plot.png"
    plot_telemetry(df, str(png_path))

    print()
    print("=" * 60)
    print("KROK 4: Podsumowanie skuteczności (3 warstwy)")
    print("=" * 60)
    from detection.rules import detect_threshold_violations
    from detection.statistical import detect_sudden_changes
    from detection.ml import detect_ml_anomalies

    df_eval = detect_threshold_violations(df)
    df_eval = detect_sudden_changes(df_eval)
    df_eval = detect_ml_anomalies(df_eval)

    # Każdy typ anomalii — ile próbek wykryła każda warstwa
    print(f"\n{'Scenariusz':<18} {'Próbki':<8} {'W1 (próg)':<10} {'W2 (zmiany)':<12} {'W3 (ML)':<9}")
    print("-" * 60)
    for atype in ["engine_failure", "gps_spoofing", "battery_drain",
                  "control_freeze", "sensor_jamming"]:
        subset = df_eval[df_eval["anomaly_type"] == atype]
        if len(subset) == 0:
            continue
        n = len(subset)
        n1 = subset["alert_threshold"].sum()
        n2 = subset["alert_change"].sum()
        n3 = subset["alert_ml"].sum()
        print(f"{atype:<18} {n:<8} {n1}/{n:<8} {n2}/{n:<10} {n3}/{n}")

    clean = df_eval[df_eval["anomaly_type"] == "none"]
    fp1 = clean["alert_threshold"].sum()
    fp2 = clean["alert_change"].sum()
    fp3 = clean["alert_ml"].sum()
    print(f"\n{'False positives':<18} {len(clean):<8} {fp1}/{len(clean):<8} "
          f"{fp2}/{len(clean):<10} {fp3}/{len(clean)}")

    print()
    print("=" * 60)
    print("KROK 5: Pełna ewaluacja (precision/recall/F1, ROC, macierze)")
    print("=" * 60)
    from evaluation.metrics import run_full_evaluation
    run_full_evaluation(df_eval)

    print()
    print("=" * 60)
    print("GOTOWE")
    print("=" * 60)
    print(f"Wykres telemetrii:  {png_path}")
    print(f"Macierze pomyłek:   {PROJECT_ROOT / 'data' / 'confusion_matrices.png'}")
    print(f"Krzywa ROC:         {PROJECT_ROOT / 'data' / 'roc_curve.png'}")


if __name__ == "__main__":
    main()
