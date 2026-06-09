"""
Modul ewaluacji - precision, recall, F1, confusion matrix.

Funkcje:
  - compute_layer_metrics(df, alert_col) -> metryki dla jednej warstwy
  - compute_full_evaluation(df)         -> tabele: global / per scenariusz / per profile
"""

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


# Domyslna mapa warstw - mozna nadpisac w report.py
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

    rows = []
    scenarios = sorted(t for t in df["tamper_type"].unique() if t != "normal")
    for atype in scenarios:
        sub = df[df["tamper_type"] == atype]
        row = {"tamper_type": atype, "probek": len(sub)}
        for col, name in layers.items():
            d = int(sub[col].sum())
            row[name] = f"{d}/{len(sub)} ({100*d/len(sub):.0f}%)"
        rows.append(row)
    per_scen_df = pd.DataFrame(rows)

    rows = []
    for profile in ["subtle", "balanced", "strong"]:
        sub = df[(df["profile"] == profile) & (df["label"] == 1)]
        if len(sub) == 0:
            continue
        row = {"profile": profile, "probek": len(sub)}
        for col, name in layers.items():
            d = int(sub[col].sum())
            row[name] = f"{d}/{len(sub)} ({100*d/len(sub):.0f}%)"
        rows.append(row)
    per_profile_df = pd.DataFrame(rows)

    return global_df, per_scen_df, per_profile_df
