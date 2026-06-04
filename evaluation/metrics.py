"""
Moduł ewaluacji — precision, recall, F1, confusion matrix, ROC curve.

Funkcje:
  - compute_layer_metrics(df, layer_col)  → metryki dla jednej warstwy
  - compute_full_evaluation(df)           → tabela dla wszystkich warstw + per scenariusz
  - plot_confusion_matrices(df)           → 3 macierze pomyłek obok siebie
  - plot_roc_curve(df)                    → krzywa ROC dla warstwy 3 (ML)
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection.rules import detect_threshold_violations
from detection.statistical import detect_sudden_changes
from detection.ml import detect_ml_anomalies


# Mapowanie nazw kolumn alertów na czytelne nazwy warstw
LAYERS = {
    "alert_threshold": "Warstwa 1 (progi)",
    "alert_change":    "Warstwa 2 (statystyki)",
    "alert_ml":        "Warstwa 3 (Isolation Forest)",
}


def compute_layer_metrics(df: pd.DataFrame, alert_col: str) -> dict:
    """
    Dla jednej warstwy: TP, FP, TN, FN, precision, recall, F1.
    Wymaga w df kolumn 'is_anomaly' (bool) i alert_col (bool).
    """
    y_true = df["is_anomaly"].astype(int).values
    y_pred = df[alert_col].astype(int).values

    # zero_division=0 chroni przed dzieleniem przez 0 (gdy warstwa nie alarmuje
    # ani razu albo nie ma żadnej anomalii)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "precision": precision, "recall": recall, "f1": f1,
    }


def compute_full_evaluation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Zwraca dwie tabele:
      1. global_metrics — precision/recall/F1 per warstwa (po wszystkich anomaliach)
      2. per_scenario   — recall per warstwa per scenariusz
    """
    # 1. Metryki globalne per warstwa
    global_rows = []
    for col, name in LAYERS.items():
        m = compute_layer_metrics(df, col)
        global_rows.append({
            "warstwa": name,
            "TP": m["TP"], "FP": m["FP"], "FN": m["FN"], "TN": m["TN"],
            "precision": round(m["precision"], 3),
            "recall": round(m["recall"], 3),
            "F1": round(m["f1"], 3),
        })
    global_df = pd.DataFrame(global_rows)

    # 2. Recall per scenariusz per warstwa
    # (precision per scenariusz nie ma sensu — fałszywe alarmy nie należą
    # do żadnego scenariusza)
    scenarios = sorted(
        [s for s in df["anomaly_type"].unique() if s != "none"]
    )
    per_scen_rows = []
    for atype in scenarios:
        subset = df[df["anomaly_type"] == atype]
        row = {"scenariusz": atype, "próbek": len(subset)}
        for col, name in LAYERS.items():
            detected = subset[col].sum()
            row[name] = f"{detected}/{len(subset)} ({100*detected/len(subset):.0f}%)"
        per_scen_rows.append(row)
    per_scen_df = pd.DataFrame(per_scen_rows)

    return global_df, per_scen_df


def plot_confusion_matrices(df: pd.DataFrame, output_path: str = None):
    """Rysuje 3 macierze pomyłek obok siebie (jedna na warstwę)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        "Macierze pomyłek — porównanie warstw detekcji",
        fontsize=13, fontweight="bold",
    )

    for ax, (col, name) in zip(axes, LAYERS.items()):
        m = compute_layer_metrics(df, col)
        cm = np.array([
            [m["TN"], m["FP"]],
            [m["FN"], m["TP"]],
        ])

        # Heatmapa — zielone TP/TN, czerwone FP/FN
        # Robimy własną mapę kolorów: po prostu rysujemy 4 prostokąty
        colors = np.array([
            ["#c8e6c9", "#ffcdd2"],   # TN zielony, FP czerwony
            ["#ffcdd2", "#c8e6c9"],   # FN czerwony, TP zielony
        ])
        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle((j, 1-i), 1, 1,
                                           facecolor=colors[i, j],
                                           edgecolor="black", linewidth=1))
                # Etykieta wartości
                ax.text(j + 0.5, 1.7 - i, str(cm[i, j]),
                        ha="center", va="center",
                        fontsize=18, fontweight="bold")
                # Etykieta typu (TN/FP/FN/TP)
                labels = [["TN", "FP"], ["FN", "TP"]]
                ax.text(j + 0.5, 1.25 - i, labels[i][j],
                        ha="center", va="center",
                        fontsize=10, color="gray")

        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels(["Brak alarmu", "Alarm"])
        ax.set_yticklabels(["Anomalia", "Normalne"])
        ax.set_xlabel("Predykcja warstwy")
        ax.set_ylabel("Stan rzeczywisty")
        ax.set_aspect("equal")
        ax.set_title(
            f"{name}\n"
            f"P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}",
            fontsize=11,
        )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"Zapisano macierze pomyłek: {output_path}")
    return fig


def plot_roc_curve(df: pd.DataFrame, output_path: str = None):
    """
    ROC curve dla warstwy 3 (Isolation Forest).
    Warstwy 1 i 2 dają tylko binarne decyzje — dla nich rysujemy pojedynczy
    punkt operacyjny zamiast krzywej.
    """
    y_true = df["is_anomaly"].astype(int).values

    # Warstwa 3 — ma ciągły score, więc rysujemy pełną krzywą.
    # ml_score jest "niższe = bardziej anomalna", więc odwracamy znak
    # żeby było "wyższe = bardziej anomalna" (standardowa konwencja ROC)
    ml_score_pos = -df["ml_score"].values
    fpr, tpr, _ = roc_curve(y_true, ml_score_pos)
    auc = roc_auc_score(y_true, ml_score_pos)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="green", linewidth=2,
            label=f"Warstwa 3 (Isolation Forest), AUC = {auc:.3f}")

    # Diagonalna referencyjna — losowy klasyfikator
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5,
            label="Losowy klasyfikator (AUC = 0.5)")

    # Punkty operacyjne dla warstw 1 i 2 (są binarne)
    for col, name, color, marker in [
        ("alert_threshold", "Warstwa 1 (progi)", "orange", "o"),
        ("alert_change",    "Warstwa 2 (statystyki)", "blue", "x"),
    ]:
        m = compute_layer_metrics(df, col)
        fpr_pt = m["FP"] / (m["FP"] + m["TN"]) if (m["FP"] + m["TN"]) > 0 else 0
        tpr_pt = m["TP"] / (m["TP"] + m["FN"]) if (m["TP"] + m["FN"]) > 0 else 0
        ax.scatter(fpr_pt, tpr_pt, color=color, s=150, marker=marker,
                   linewidth=2.5, zorder=5,
                   label=f"{name} (punkt operacyjny)")

    ax.set_xlabel("False Positive Rate (1 - specyficzność)")
    ax.set_ylabel("True Positive Rate (recall / czułość)")
    ax.set_title("Krzywa ROC — porównanie warstw detekcji",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"Zapisano krzywą ROC: {output_path}")
    return fig


def run_full_evaluation(df: pd.DataFrame, output_dir: Path = None):
    """
    Pełna ewaluacja: aplikuje wszystkie 3 warstwy, oblicza metryki,
    wypisuje tabele i generuje wykresy.
    """
    # Aplikujemy wszystkie warstwy (jeśli już nie są)
    if "alert_threshold" not in df.columns:
        df = detect_threshold_violations(df)
    if "alert_change" not in df.columns:
        df = detect_sudden_changes(df)
    if "alert_ml" not in df.columns:
        df = detect_ml_anomalies(df)

    # Liczymy metryki
    global_df, per_scen_df = compute_full_evaluation(df)

    print("=" * 70)
    print("METRYKI GLOBALNE (wszystkie anomalie razem)")
    print("=" * 70)
    print(global_df.to_string(index=False))

    print()
    print("=" * 70)
    print("RECALL PER SCENARIUSZ (jaki % próbek danego typu wykryto)")
    print("=" * 70)
    print(per_scen_df.to_string(index=False))

    # Generujemy wykresy
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data"
    output_dir = Path(output_dir)

    print()
    cm_path = output_dir / "confusion_matrices.png"
    plot_confusion_matrices(df, str(cm_path))
    plt.close()

    roc_path = output_dir / "roc_curve.png"
    plot_roc_curve(df, str(roc_path))
    plt.close()

    # Zapis tabel do CSV (do wstawienia w sprawozdaniu)
    global_df.to_csv(output_dir / "metrics_global.csv", index=False)
    per_scen_df.to_csv(output_dir / "metrics_per_scenario.csv", index=False)
    print(f"\nZapisano tabele CSV: {output_dir}\\metrics_global.csv,")
    print(f"                     {output_dir}\\metrics_per_scenario.csv")

    return global_df, per_scen_df


if __name__ == "__main__":
    csv_path = PROJECT_ROOT / "data" / "flight_with_anomalies.csv"
    if not csv_path.exists():
        print(f"BŁĄD: nie ma {csv_path}. Uruchom najpierw `python run_all.py`.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    run_full_evaluation(df)
