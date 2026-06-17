from pathlib import Path

from .metrics import compute_full_evaluation, DEFAULT_LAYERS


def run_full_evaluation(
    df,
    layers: dict = None,
    score_columns: dict = None,
    output_dir: Path = None,
):
    if layers is None:
        layers = DEFAULT_LAYERS
    if output_dir is None:
        from pathlib import Path as _Path
        output_dir = _Path("data")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from .plots import plot_confusion_matrices, plot_roc_curves

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

    if score_columns:
        roc_path = output_dir / "roc_curves.png"
        plot_roc_curves(df, score_columns, layers, str(roc_path))

    global_df.to_csv(output_dir / "metrics_global.csv", index=False)
    per_scen_df.to_csv(output_dir / "metrics_per_scenario.csv", index=False)
    per_profile_df.to_csv(output_dir / "metrics_per_profile.csv", index=False)
    print(f"\nZapisano tabele CSV w: {output_dir}")

    return global_df, per_scen_df, per_profile_df
