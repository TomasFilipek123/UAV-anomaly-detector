"""
Pelny pipeline na rzeczywistym datasecie Kaggle:
  1. Wczytuje dataset (data/drone_telemetry_v2.csv)
  2. Dzieli train/test po replicate
  3. Aplikuje 3 warstwy detekcji (rules / statistical / ML)
  4. Ewaluacja: precision/recall/F1, macierze pomylek, krzywe ROC
  5. Wizualizacja jednego wybranego case_id

Uruchom z katalogu projektu:
    python run_all.py [--algorithm ALGO] [--case CASE]

Domyslny algorytm warstwy 3 to isolation_forest. Inne opcje:
    one_class_svm, lof, random_forest, xgboost
"""

from argparse import ArgumentParser
from pathlib import Path

from data.loader import load_dataset, split_by_replicate, summarize
from detection.rules import detect_threshold_violations
from detection.statistical import detect_sudden_changes
from detection.ml import AnomalyDetector, compute_features
from evaluation.report import run_full_evaluation
from notebooks.visualize import plot_case

PROJECT_ROOT = Path(__file__).resolve().parent
SUPPORTED_ALGORITHMS = [
    "isolation_forest",
    "one_class_svm",
    "lof",
    "random_forest",
    "xgboost",
]


def parse_args() -> tuple[str, int, bool]:
    parser = ArgumentParser(description="Uruchom pipeline detekcji anomalii drona.")
    parser.add_argument("--algorithm", "-a", default="isolation_forest",
                        choices=SUPPORTED_ALGORITHMS,
                        help="Algorytm ML dla warstwy 3.")
    parser.add_argument("--case", "-c", type=int, default=550,
                        help="case_id do wizualizacji.")
    parser.add_argument("--load", "-l", action="store_true",
                        help="Wczytaj gotowy model zamiast trenować na nowo.")
    args = parser.parse_args()
    return args.algorithm, args.case, args.load


def _choose_case_id(test_df, requested_case_id: int) -> int:
    case_ids = set(test_df["case_id"].unique())
    if requested_case_id in case_ids:
        return requested_case_id
    if not case_ids:
        raise RuntimeError("Brak danych w zbiorze testowym. Nie można wygenerować wykresu.")
    chosen = int(sorted(case_ids)[0])
    print(f"UWAGA: case_id={requested_case_id} nie występuje w zestawie testowym.")
    print(f"       Używam case_id={chosen} do wizualizacji.")
    return chosen


def run_pipeline(algorithm: str, case_id: int, load_model: bool = False) -> None:
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
    
    models_dir = PROJECT_ROOT / "models"
    model_path = models_dir / f"{algorithm}.pkl"
    
    if load_model and model_path.exists():
        print(f"Wczytywanie gotowego modelu: {model_path}")
        detector = AnomalyDetector.load(model_path)
        print("Model załadowany pomyślnie.")
        print("Liczenie cech okienkowych (test)...")
        test_features = compute_features(test, window=8)
    else:
        if load_model:
            print(f"Model nie znaleziony: {model_path}")
            print("Będę trenować model na nowo...")
        # Cechy okienkowe liczymy raz dla train i raz dla test (gdyby trenowac
        # wiele algorytmow z rzedu - cache zaoszczedzilby kilka minut na pelnym datasecie).
        print("Liczenie cech okienkowych (train)...")
        train_features = compute_features(train, window=8)
        print("Liczenie cech okienkowych (test)...")
        test_features = compute_features(test, window=8)

        detector = AnomalyDetector(algorithm=algorithm).fit(train, features=train_features)
        # Zapis modelu
        models_dir.mkdir(parents=True, exist_ok=True)
        detector.save(model_path)
        print(f"Zapisano model: {model_path}")
    
    test = detector.predict(test, features=test_features)
    print(f"Alerty W3: {test['alert_ml'].sum():,}/{len(test):,}")

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
    case_id = _choose_case_id(test, case_id)
    png_path = PROJECT_ROOT / "data" / f"case_{case_id}_plot.png"
    plot_case(test, case_id, str(png_path))

    print()
    print("=" * 60)
    print("GOTOWE")
    print("=" * 60)
    print(f"Wykresy + tabele w: {PROJECT_ROOT / 'data'}")
    print(f"Model w:            {models_dir / f'{algorithm}.pkl'}")


def main() -> None:
    algorithm, case_id, load_model = parse_args()
    run_pipeline(algorithm, case_id, load_model)


if __name__ == "__main__":
    main()
