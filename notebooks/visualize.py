"""
Wizualizacja telemetrii dla pojedynczego lotu (case_id).

Rysuje 5 paneli:
  - altitude  (m)
  - speed     (m/s)
  - heading   (deg)
  - lat / lon (trajektoria 2D - mini-mapa w osobnym panelu)
  - lat / lon vs czas (alternatywa)

Plus zaznacza:
  - pasy tla = ground truth (label == 1) z etykieta tamper_type
  - kropki   = alerty z kazdej warstwy detekcji
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _anomaly_segments(
    case_df: pd.DataFrame,
    merge_gap: float = 0.2,
    min_duration: float = 0.15,
) -> list[tuple[float, float, str]]:
    """Zwraca ciagle segmenty (t_start, t_end, tamper_type).

    Scalmy krótkie przerwy i filtrujemy bardzo krótkie segmenty,
    aby wykres nie był przesadnie zagęszczony dla przypadków
    z wieloma szybkim przełączeniami label/normal.
    """
    segs = []
    in_seg = False
    seg_start = None
    seg_type = None
    for _, row in case_df.iterrows():
        if row["label"] == 1 and not in_seg:
            seg_start = row["t_rel"]
            seg_type = row["tamper_type"]
            in_seg = True
        elif (row["label"] == 0 or row["tamper_type"] != seg_type) and in_seg:
            segs.append((seg_start, row["t_rel"], seg_type))
            if row["label"] == 1:
                seg_start = row["t_rel"]
                seg_type = row["tamper_type"]
                in_seg = True
            else:
                in_seg = False
    if in_seg:
        segs.append((seg_start, case_df["t_rel"].iloc[-1], seg_type))

    merged = []
    for start, end, typ in segs:
        if not merged:
            merged.append((start, end, typ))
            continue
        prev_start, prev_end, prev_typ = merged[-1]
        gap = start - prev_end
        if typ == prev_typ and gap <= merge_gap:
            merged[-1] = (prev_start, end, prev_typ)
        else:
            merged.append((start, end, typ))

    filtered = []
    for start, end, typ in merged:
        if end - start >= min_duration:
            filtered.append((start, end, typ))
    return filtered


def _binary_segments(
    case_df: pd.DataFrame,
    bool_col: str,
    merge_gap: float = 0.2,
    min_duration: float = 0.15,
) -> list[tuple[float, float]]:
    """Zwraca zgrupowane segmenty dla boolowskiej kolumny alertów."""
    segs = []
    in_seg = False
    seg_start = None
    for _, row in case_df.iterrows():
        if row[bool_col] and not in_seg:
            seg_start = row["t_rel"]
            in_seg = True
        elif not row[bool_col] and in_seg:
            segs.append((seg_start, row["t_rel"]))
            in_seg = False
    if in_seg:
        segs.append((seg_start, case_df["t_rel"].iloc[-1]))

    merged = []
    for start, end in segs:
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        gap = start - prev_end
        if gap <= merge_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return [(start, end) for start, end in merged if end - start >= min_duration]


def plot_case(
    df: pd.DataFrame,
    case_id,
    output_path: str = None,
) -> plt.Figure:
    """
    Rysuje telemetrie dla jednego case_id. Wymaga w df kolumn:
      timestamp, t_rel, latitude, longitude, altitude, speed, heading,
      label, tamper_type,
      oraz opcjonalnie alert_threshold, alert_change, alert_ml
    """
    case_df = df[df["case_id"] == case_id].copy().sort_values("t_rel").reset_index(drop=True)
    if len(case_df) == 0:
        raise ValueError(f"Brak danych dla case_id={case_id}")

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(4, 2, width_ratios=[3, 1], hspace=0.35, wspace=0.2)

    ax_alt = fig.add_subplot(gs[0, 0])
    ax_spd = fig.add_subplot(gs[1, 0], sharex=ax_alt)
    ax_hdg = fig.add_subplot(gs[2, 0], sharex=ax_alt)
    ax_lat = fig.add_subplot(gs[3, 0], sharex=ax_alt)
    ax_map = fig.add_subplot(gs[:, 1])

    fig.suptitle(
        f"Telemetria drona - case_id = {case_id}  "
        f"(profile = {case_df['profile'].iloc[0]})",
        fontsize=13, fontweight="bold",
    )

    panels = [
        (ax_alt, "altitude",  "Wysokosc [m]",  "tab:blue"),
        (ax_spd, "speed",     "Predkosc [m/s]", "tab:green"),
        (ax_hdg, "heading",   "Kurs [deg]",    "tab:purple"),
        (ax_lat, "latitude",  "Lat [deg]",     "tab:brown"),
    ]

    segs = _anomaly_segments(case_df)

    has = {col: col in case_df.columns for col in ["alert_threshold", "alert_change", "alert_ml"]}

    ml_segs = _binary_segments(case_df, "alert_ml") if has["alert_ml"] else []

    for ax, col, label, color in panels:
        ax.plot(case_df["t_rel"], case_df[col], color=color, linewidth=1.0)
        for start, end, atype in segs:
            ax.axvspan(start, end, alpha=0.18, color="red")
            if ax is ax_alt:
                ax.text((start + end) / 2, ax.get_ylim()[1] * 0.95,
                        atype, ha="center", fontsize=7,
                        color="darkred", fontweight="bold", rotation=15)

        for alert_col, c, marker, s in [
            ("alert_threshold", "orange", "o", 15),
            ("alert_change",    "blue",   "x", 30),
        ]:
            if has[alert_col]:
                sub = case_df[case_df[alert_col]]
                if len(sub) > 0:
                    ax.scatter(sub["t_rel"], sub[col], color=c, s=s, zorder=5, marker=marker,
                               alpha=0.7, linewidths=0.5)

        if ml_segs:
            for start, end in ml_segs:
                ax.axvspan(start, end, alpha=0.12, color="limegreen")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    ax_lat.set_xlabel("Czas od startu [s]")

    # Mini-mapa trajektorii
    ax_map.plot(case_df["longitude"], case_df["latitude"],
                color="gray", linewidth=0.8, alpha=0.6, label="trasa")
    anom = case_df[case_df["label"] == 1]
    if len(anom) > 0:
        ax_map.scatter(anom["longitude"], anom["latitude"],
                       color="red", s=8, alpha=0.6, label="anomalia (GT)")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_title("Trajektoria GPS")
    ax_map.legend(loc="best", fontsize=8)
    ax_map.grid(True, alpha=0.3)
    ax_map.set_aspect("equal", adjustable="datalim")

    # Legenda zbiorcza
    legend_elems = [
        mpatches.Patch(color="red", alpha=0.18, label="Anomalia (ground truth)"),
    ]
    if has["alert_threshold"]:
        legend_elems.append(plt.Line2D([0], [0], marker="o", color="w",
                                        markerfacecolor="orange", markersize=8,
                                        label="W1: prog"))
    if has["alert_change"]:
        legend_elems.append(plt.Line2D([0], [0], marker="x", color="blue",
                                        markersize=8, label="W2: nagla zmiana",
                                        linewidth=0))
    if has["alert_ml"]:
        legend_elems.append(plt.Line2D([0], [0], marker="^", color="w",
                                        markerfacecolor="green", markersize=9,
                                        label="W3: ML"))
    fig.legend(handles=legend_elems, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.01), fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.06)

    if output_path:
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"Zapisano wykres: {output_path}")
    return fig


def pick_case_with_anomalies(df: pd.DataFrame, min_anomalies: int = 50) -> int:
    """Wybiera pierwszy case_id z odpowiednia liczba anomalii (do demo)."""
    counts = df[df["label"] == 1].groupby("case_id").size()
    candidates = counts[counts >= min_anomalies]
    if len(candidates) == 0:
        return int(df["case_id"].iloc[0])
    return int(candidates.index[0])


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from data.loader import load_dataset

    df = load_dataset()
    case = pick_case_with_anomalies(df)
    print(f"Wybrany case_id: {case}")
    plots_dir = PROJECT_ROOT / "data" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    png_path = plots_dir / f"case_{case}_plot.png"
    plot_case(df, case, str(png_path))
