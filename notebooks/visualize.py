"""
Wizualizacja telemetrii z zaznaczonymi alertami i ground truth.

Pokazuje na czterech panelach (altitude, speed, heading, battery):
  - przebieg parametru w czasie
  - czerwone pasy w tle: ground truth (anomalie wstrzyknięte)
  - pomarańczowe kropki: alerty warstwy 1 (progi)
  - niebieskie kropki: alerty warstwy 2 (nagłe zmiany)
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# Dodajemy katalog projektu (rodzic notebooks/) do sys.path
# żeby działały importy `from detection.rules import ...`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection.rules import detect_threshold_violations
from detection.statistical import detect_sudden_changes
from detection.ml import detect_ml_anomalies


def plot_telemetry(df: pd.DataFrame, output_path: str = None):
    """Rysuje 4-panelowy wykres telemetrii z alertami wszystkich 3 warstw."""
    # Aplikujemy wszystkie 3 warstwy detekcji
    df = detect_threshold_violations(df)
    df = detect_sudden_changes(df)
    df = detect_ml_anomalies(df)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        "Telemetria drona — detekcja anomalii\n"
        "(warstwa 1: progi  •  warstwa 2: nagłe zmiany / freeze  •  warstwa 3: Isolation Forest)",
        fontsize=12, fontweight="bold",
    )

    params = [
        ("altitude_m", "Wysokość [m]", "tab:blue"),
        ("speed_mps", "Prędkość [m/s]", "tab:green"),
        ("heading_deg", "Kurs [°]", "tab:purple"),
        ("battery_pct", "Bateria [%]", "tab:orange"),
    ]

    # Identyfikujemy ciągłe segmenty anomalii (do narysowania pasów tła)
    anomaly_segments = []
    if "is_anomaly" in df.columns:
        in_anomaly = False
        seg_start = None
        seg_type = None
        for i, row in df.iterrows():
            if row["is_anomaly"] and not in_anomaly:
                seg_start = row["timestamp"]
                seg_type = row["anomaly_type"]
                in_anomaly = True
            elif (not row["is_anomaly"] or row["anomaly_type"] != seg_type) and in_anomaly:
                anomaly_segments.append((seg_start, row["timestamp"], seg_type))
                in_anomaly = row["is_anomaly"]
                if in_anomaly:
                    seg_start = row["timestamp"]
                    seg_type = row["anomaly_type"]
        if in_anomaly:
            anomaly_segments.append((seg_start, df["timestamp"].iloc[-1], seg_type))

    for ax, (col, label, color) in zip(axes, params):
        # Linia podstawowa
        ax.plot(df["timestamp"], df[col], color=color, linewidth=1.0, label=label)

        # Pasy tła = ground truth
        for start, end, atype in anomaly_segments:
            ax.axvspan(start, end, alpha=0.18, color="red")
            # Etykieta typu anomalii nad pierwszym panelem
            if col == "altitude_m":
                ax.text((start + end) / 2, ax.get_ylim()[1] * 0.95,
                        atype, ha="center", fontsize=8,
                        color="darkred", fontweight="bold")

        # Alerty warstwy 1 (progi)
        l1 = df[df["alert_threshold"]]
        if len(l1) > 0:
            ax.scatter(l1["timestamp"], l1[col],
                       color="orange", s=15, zorder=5, marker="o",
                       edgecolors="darkorange", linewidth=0.5)

        # Alerty warstwy 2 (nagłe zmiany / freeze)
        l2 = df[df["alert_change"]]
        if len(l2) > 0:
            ax.scatter(l2["timestamp"], l2[col],
                       color="blue", s=30, zorder=6, marker="x", linewidth=1.5)

        # Alerty warstwy 3 (Isolation Forest)
        l3 = df[df["alert_ml"]]
        if len(l3) > 0:
            ax.scatter(l3["timestamp"], l3[col],
                       color="green", s=40, zorder=7, marker="^",
                       edgecolors="darkgreen", linewidth=0.5, alpha=0.7)

        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Czas [s]")

    # Legenda zbiorcza
    legend_elements = [
        mpatches.Patch(color="red", alpha=0.18, label="Anomalia (ground truth)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="orange",
                   markersize=8, label="Warstwa 1: próg fizyczny"),
        plt.Line2D([0], [0], marker="x", color="blue",
                   markersize=8, label="Warstwa 2: nagła zmiana / freeze", linewidth=0),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="green",
                   markersize=9, label="Warstwa 3: Isolation Forest"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.01), fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"Zapisano wykres: {output_path}")
    return fig


if __name__ == "__main__":
    csv_path = PROJECT_ROOT / "data" / "flight_with_anomalies.csv"
    png_path = PROJECT_ROOT / "data" / "telemetry_plot.png"
    df = pd.read_csv(csv_path)
    plot_telemetry(df, str(png_path))
