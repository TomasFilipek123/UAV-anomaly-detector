"""Rysowanie wykresow ewaluacji detekcji anomalii."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .metrics import compute_layer_metrics


def plot_confusion_matrices(
    df: pd.DataFrame,
    layers: dict = None,
    output_path: str = None,
):
    if layers is None:
        from .metrics import DEFAULT_LAYERS
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
