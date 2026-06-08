"""
Pelny pipeline na rzeczywistym datasecie Kaggle:
  1. Wczytuje dataset (data/drone_telemetry_v2.csv)
  2. Dzieli train/test po replicate
  3. Aplikuje 3 warstwy detekcji (rules / statistical / ML)
  4. Ewaluacja: precision/recall/F1, macierze pomylek, krzywe ROC
  5. Wizualizacja jednego wybranego case_id

Uruchom z katalogu projektu:
    python run_all.py [algorithm]

Domyslny algorytm warstwy 3 to isolation_forest. Inne opcje:
    one_class_svm, lof, random_forest, xgboost
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import load_dataset, split_by_replicate, summarize
from detection.rules import detect_threshold_violations
from detection.statistical import detect_sudden_changes
from detection.ml import AnomalyDetector, compute_features
from evaluation.metrics import run_full_evaluation
from notebooks.visualize import plot_case, pick_case_with_anomalies


def main(algorithm: str = "isolation_forest") -> None:
    print("=" * 60)
    print("KROK 1: Wczytywanie datasetu")
    print("=" * 60)
    df = load_dataset()
    summarize(df)

    print()
    print("=" * 60)
    print("KROK 2: Train/test split (po replicate)")
    print("=" * 60)
    train, test = split_by_replicate(df)
    print(f"Train (replicate 0,1,2): {len(train):,}")
    print(f"Test  (replicate 3):     {len(test):,}")

    print()
    print("=" * 60)
    print("KROK 3: Warstwa 1 - progi fizyczne")
    print("=" * 60)
    test = detect_threshold_violations(test)
    print(f"Alerty W1: {test['alert_threshold'].sum():,}/{len(test):,}")

    print()
    print("=" * 60)
    print("KROK 4: Warstwa 2 - statystyki kroczace")
    print("=" * 60)
    test = detect_sudden_changes(test)
    print(f"Alerty W2: {test['alert_change'].sum():,}/{len(test):,}")

    print()
    print("=" * 60)
    print(f"KROK 5: Warstwa 3 - ML ({algorithm})")
    print("=" * 60)
    # Cechy okienkowe liczymy raz dla train i raz dla test (gdyby trenowac
    # wiele algorytmow z rzedu - cache zaoszczedzilby kilka minut na pelnym datasecie).
    print("Liczenie cech okienkowych (train)...")
    train_features = compute_features(train, window=8)
    print("Liczenie cech okienkowych (test)...")
    test_features = compute_features(test, window=8)

    detector = AnomalyDetector(algorithm=algorithm).fit(train, features=train_features)
    test = detector.predict(test, features=test_features)
    print(f"Alerty W3: {test['alert_ml'].sum():,}/{len(test):,}")

    # Zapis modelu
    models_dir = PROJECT_ROOT / "models"
    detector.save(models_dir / f"{algorithm}.pkl")
    print(f"Zapisano model: {models_dir / f'{algorithm}.pkl'}")

    print()
    print("=" * 60)
    print("KROK 6: Pelna ewaluacja")
    print("=" * 60)
    score_dir = "low" if algorithm in {"isolation_forest", "one_class_svm", "lof"} else "high"
    run_full_evaluation(
        test,
        score_columns={f"W3 ({algorithm})": ("ml_score", score_dir)},
        output_dir=PROJECT_ROOT / "data",
    )

    print()
    print("=" * 60)
    print("KROK 7: Wizualizacja przykladowego case'a")
    print("=" * 60)
    case = pick_case_with_anomalies(test)
    png_path = PROJECT_ROOT / "data" / f"case_{case}_plot.png"
    plot_case(test, case, str(png_path))

    print()
    print("=" * 60)
    print("GOTOWE")
    print("=" * 60)
    print(f"Wykresy + tabele w: {PROJECT_ROOT / 'data'}")
    print(f"Model w:            {models_dir / f'{algorithm}.pkl'}")


if __name__ == "__main__":
    algo = sys.argv[1] if len(sys.argv) > 1 else "isolation_forest"
    main(algo)
