"""
Modul ewaluacji - precision, recall, F1, confusion matrix, ROC.

Funkcje:
  - compute_layer_metrics(df, alert_col)      -> metryki dla jednej warstwy
  - compute_full_evaluation(df)                -> tabele: global / per scenariusz / per profile
  - plot_confusion_matrices(df, layers)        -> macierze pomylek obok siebie
  - plot_roc_curves(df, score_columns)         -> krzywe ROC dla algorytmow ML
  - run_full_evaluation(df, ...)               -> kompletny raport + zapis wykresow
"""

from pathlib import Path

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


# Domyslna mapa warstw - mozna nadpisac w run_full_evaluation
DEFAULT_LAYERS = {
    "alert_threshold": "Warstwa 1 (progi)",
    "alert_change":    "Warstwa 2 (statystyki)",
    "alert_ml":        "Warstwa 3 (ML)",
}


def compute_layer_metrics(df: pd.DataFrame, alert_col: str) -> dict:
    """TP, FP, TN, FN + precision, recall, F1 dla jednej kolumny alertu."""
    y_true = df["label"].astype(int).values
    y_pred = df[alert_col].astype(int).values

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }


def compute_full_evaluation(
    df: pd.DataFrame,
    layers: dict = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Zwraca trzy tabele:
      1. global       - precision/recall/F1 per warstwa
      2. per_scenario - recall per warstwa per tamper_type
      3. per_profile  - recall per warstwa per profile intensywnosci
    """
    if layers is None:
        layers = DEFAULT_LAYERS

    # 1. Globalne
    rows = []
    for col, name in layers.items():
        m = compute_layer_metrics(df, col)
        rows.append({
            "warstwa": name,
            "TP": m["TP"], "FP": m["FP"], "FN": m["FN"], "TN": m["TN"],
            "precision": round(m["precision"], 3),
            "recall":    round(m["recall"], 3),
            "F1":        round(m["f1"], 3),
        })
    global_df = pd.DataFrame(rows)

    # 2. Per tamper_type (recall - bo precision per scenariusz nie ma sensu)
    rows = []
    scenarios = sorted(t for t in df["tamper_type"].unique() if t != "normal")
    for atype in scenarios:
        sub = df[df["tamper_type"] == atype]
        row = {"tamper_type": atype, "probek": len(sub)}
        for col, name in layers.items():
            d = sub[col].sum()
            row[name] = f"{d}/{len(sub)} ({100*d/len(sub):.0f}%)"
        rows.append(row)
    per_scen_df = pd.DataFrame(rows)

    # 3. Per profile (subtle / balanced / strong) - tylko dla anomalii
    rows = []
    for profile in ["subtle", "balanced", "strong"]:
        sub = df[(df["profile"] == profile) & (df["label"] == 1)]
        if len(sub) == 0:
            continue
        row = {"profile": profile, "probek": len(sub)}
        for col, name in layers.items():
            d = sub[col].sum()
            row[name] = f"{d}/{len(sub)} ({100*d/len(sub):.0f}%)"
        rows.append(row)
    per_profile_df = pd.DataFrame(rows)

    return global_df, per_scen_df, per_profile_df


def plot_confusion_matrices(
    df: pd.DataFrame,
    layers: dict = None,
    output_path: str = None,
):
    if layers is None:
        layers = DEFAULT_LAYERS

    n = len(layers)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Macierze pomylek - porownanie warstw detekcji",
                 fontsize=13, fontweight="bold")

    for ax, (col, name) in zip(axes, layers.items()):
        m = compute_layer_metrics(df, col)
        cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        colors = np.array([["#c8e6c9", "#ffcdd2"], ["#ffcdd2", "#c8e6c9"]])
        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle((j, 1-i), 1, 1,
                                           facecolor=colors[i, j],
                                           edgecolor="black", linewidth=1))
                ax.text(j + 0.5, 1.7 - i, f"{cm[i, j]:,}",
                        ha="center", va="center", fontsize=14, fontweight="bold")
                labels = [["TN", "FP"], ["FN", "TP"]]
                ax.text(j + 0.5, 1.25 - i, labels[i][j],
                        ha="center", va="center", fontsize=10, color="gray")
        ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5]); ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels(["Brak alarmu", "Alarm"])
        ax.set_yticklabels(["Anomalia", "Normalne"])
        ax.set_xlabel("Predykcja"); ax.set_ylabel("Rzeczywistosc")
        ax.set_aspect("equal")
        ax.set_title(f"{name}\nP={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}",
                     fontsize=11)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"Zapisano macierze pomylek: {output_path}")
    return fig


def plot_roc_curves(
    df: pd.DataFrame,
    score_columns: dict,
    binary_layers: dict = None,
    output_path: str = None,
):
    """
    score_columns : dict {nazwa_legendy: (kolumna_score, kierunek)}
        kierunek = "high" jesli wyzszy score = bardziej anomalna (supervised proba)
        kierunek = "low"  jesli nizszy  score = bardziej anomalna (IF score_samples)
    binary_layers : warstwy binarne - pokazane jako punkty operacyjne (P/R)
    """
    if binary_layers is None:
        binary_layers = {"alert_threshold": "Warstwa 1", "alert_change": "Warstwa 2"}

    y_true = df["label"].astype(int).values

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5,
            label="Losowy klasyfikator (AUC = 0.50)")

    colors = ["green", "purple", "brown", "teal", "red"]
    for (legend, (col, direction)), color in zip(score_columns.items(), colors):
        if col not in df.columns:
            continue
        scores = df[col].values
        if direction == "low":
            scores = -scores
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{legend} (AUC = {auc:.3f})")

    markers = ["o", "x", "s", "^"]
    bin_colors = ["orange", "blue", "magenta", "olive"]
    for (col, name), marker, c in zip(binary_layers.items(), markers, bin_colors):
        if col not in df.columns:
            continue
        m = compute_layer_metrics(df, col)
        fpr_pt = m["FP"] / (m["FP"] + m["TN"]) if (m["FP"] + m["TN"]) > 0 else 0
        tpr_pt = m["TP"] / (m["TP"] + m["FN"]) if (m["TP"] + m["FN"]) > 0 else 0
        ax.scatter(fpr_pt, tpr_pt, color=c, s=150, marker=marker,
                   linewidth=2.5, zorder=5, label=f"{name} (operating point)")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Krzywe ROC - porownanie warstw / algorytmow",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"Zapisano krzywe ROC: {output_path}")
    return fig


def run_full_evaluation(
    df: pd.DataFrame,
    layers: dict = None,
    score_columns: dict = None,
    output_dir: Path = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Kompletny raport: drukuje 3 tabele, generuje 2 wykresy, zapisuje CSV.

    df musi miec kolumny: label, tamper_type, profile + kolumny alert_* per warstwa.
    """
    if layers is None:
        layers = DEFAULT_LAYERS
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_df, per_scen_df, per_profile_df = compute_full_evaluation(df, layers)

    print("=" * 70)
    print("METRYKI GLOBALNE")
    print("=" * 70)
    print(global_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("RECALL PER TAMPER_TYPE")
    print("=" * 70)
    print(per_scen_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("RECALL PER PROFILE (intensywnosc anomalii)")
    print("=" * 70)
    print(per_profile_df.to_string(index=False))

    cm_path = output_dir / "confusion_matrices.png"
    plot_confusion_matrices(df, layers, str(cm_path))
    plt.close()

    if score_columns:
        roc_path = output_dir / "roc_curves.png"
        plot_roc_curves(df, score_columns, layers, str(roc_path))
        plt.close()

    global_df.to_csv(output_dir / "metrics_global.csv", index=False)
    per_scen_df.to_csv(output_dir / "metrics_per_scenario.csv", index=False)
    per_profile_df.to_csv(output_dir / "metrics_per_profile.csv", index=False)
    print(f"\nZapisano tabele CSV w: {output_dir}")

    return global_df, per_scen_df, per_profile_df
