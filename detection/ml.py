"""
Warstwa 3: detekcja anomalii przez Isolation Forest na cechach okienkowych.

Idea:
  - Dla każdej próbki obliczamy cechy opisujące okno N sekund przed nią:
    mean, std, range, slope dla altitude/speed/battery
    + statystyki cyklicznej różnicy kursu.
  - Isolation Forest uczy się jak wyglądają "normalne" okna na czystym locie,
    a następnie zwraca anomaly score dla każdego okna w locie testowym.

Ta warstwa łapie anomalie kontekstowe, które warstwa 2 omija — np. zacięcie
sterów (control_freeze), bo zamrożone parametry powodują std=0 w oknie,
co w normalnym locie się nie zdarza.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# Domyślna konfiguracja
DEFAULT_CONFIG = {
    "window_s": 8,                # długość okna cech
    "n_estimators": 150,          # liczba drzew w lesie
    "random_state": 42,           # ziarno dla powtarzalności
    "n_train_flights": 3,         # ile czystych lotów użyć do treningu
    "threshold_percentile": 2.0,  # próg = ten percentyl scores z czystego lotu
    # ^ 2.0 oznacza: na czystym treningu chcemy ~2% false positive rate.
    #   Każda próbka testowa ze scorem poniżej tego progu = anomalia.
}


def _circular_diff(series: pd.Series) -> pd.Series:
    """Pochodna kursu z poprawnym wrap-around (359 -> 1 = 2°, nie -358°)."""
    raw = series.diff()
    return ((raw + 180) % 360) - 180


def compute_features(df: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """
    Wylicza cechy okienkowe dla każdej próbki w DataFrame.

    Zwraca DataFrame z 14 cechami:
      - alt_mean, alt_std, alt_range, alt_slope
      - spd_mean, spd_std, spd_range, spd_slope
      - bat_mean, bat_slope (bateria jest monotoniczna, std/range bezużyteczne)
      - hdg_diff_mean, hdg_diff_std, hdg_diff_abs_max
      - sample_var_score (zsumowane std — wykrywa "zamrożone" okna)
    """
    feats = pd.DataFrame(index=df.index)
    min_p = window // 2  # minimalna liczba próbek na początku serii

    # --- Cechy okienkowe per parametr ---
    for col, prefix in [
        ("altitude_m", "alt"),
        ("speed_mps", "spd"),
    ]:
        s = df[col]
        roll = s.rolling(window=window, min_periods=min_p)
        feats[f"{prefix}_mean"] = roll.mean()
        feats[f"{prefix}_std"] = roll.std()
        feats[f"{prefix}_range"] = roll.max() - roll.min()
        # Slope = liniowy gradient w oknie (proste przybliżenie: (last-first)/window)
        feats[f"{prefix}_slope"] = (s - s.shift(window - 1)) / window

    # Bateria — tylko mean i slope (monotoniczne zmienne, std/range nieinformatywne)
    bat_roll = df["battery_pct"].rolling(window=window, min_periods=min_p)
    feats["bat_mean"] = bat_roll.mean()
    feats["bat_slope"] = (
        df["battery_pct"] - df["battery_pct"].shift(window - 1)
    ) / window

    # Kurs — używamy cyklicznych różnic
    hdg_diff = _circular_diff(df["heading_deg"])
    feats["hdg_diff_mean"] = hdg_diff.rolling(window=window, min_periods=min_p).mean()
    feats["hdg_diff_std"] = hdg_diff.rolling(window=window, min_periods=min_p).std()
    feats["hdg_diff_abs_max"] = (
        hdg_diff.abs().rolling(window=window, min_periods=min_p).max()
    )

    # Dodatkowa cecha: zsumowane std — niskie wartości = "zamrożone okno"
    feats["sample_var_score"] = (
        feats["alt_std"].fillna(0)
        + feats["spd_std"].fillna(0)
        + feats["hdg_diff_std"].fillna(0)
    )

    # Wypełnij NaN-y na początku (przed pełnym oknem) — backfill, potem 0
    feats = feats.bfill().fillna(0)
    return feats


def train_isolation_forest(
    df_train: pd.DataFrame,
    config: dict = None,
) -> tuple[IsolationForest, float, list]:
    """
    Trenuje Isolation Forest na czystym locie i wylicza próg z percentyla scores.

    Zwraca (model, threshold, feature_names).
    """
    if config is None:
        config = DEFAULT_CONFIG

    train_feats = compute_features(df_train, window=config["window_s"])
    feature_names = list(train_feats.columns)

    model = IsolationForest(
        n_estimators=config["n_estimators"],
        contamination="auto",  # nie używamy do progowania — robimy to ręcznie
        random_state=config["random_state"],
        n_jobs=-1,
    )
    model.fit(train_feats.values)

    # Próg = określony percentyl scores na zbiorze treningowym
    # (niższy score = bardziej anomalna; bierzemy dolny ogon rozkładu)
    train_scores = model.score_samples(train_feats.values)
    threshold = np.percentile(train_scores, config["threshold_percentile"])

    return model, threshold, feature_names


def detect_ml_anomalies(
    df_test: pd.DataFrame,
    df_train: pd.DataFrame = None,
    config: dict = None,
) -> pd.DataFrame:
    """
    Wykrywa anomalie w df_test używając Isolation Forest wytrenowanego na df_train.

    Jeśli df_train nie podany — generuje N czystych lotów (różne seedy) i łączy
    je w jeden zbiór treningowy. Większa różnorodność = mniej false positives.

    Zwraca df_test z dodanymi kolumnami:
      - alert_ml   : bool, czy próbka oznaczona jako anomalia
      - ml_score   : ciągły anomaly score (niższy = bardziej anomalna)
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Jeśli nie podano zbioru treningowego, generujemy N czystych lotów
    if df_train is None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from data.generate import generate_normal_flight

        duration = int(df_test["timestamp"].max()) + 1
        train_flights = [
            generate_normal_flight(duration_s=duration, seed=1000 + i)
            for i in range(config["n_train_flights"])
        ]
        df_train = pd.concat(train_flights, ignore_index=True)

    # Trening
    model, threshold, _ = train_isolation_forest(df_train, config)

    # Inference: każda próbka ze score poniżej progu = anomalia
    test_feats = compute_features(df_test, window=config["window_s"])
    scores = model.score_samples(test_feats.values)

    result = df_test.copy()
    result["alert_ml"] = scores < threshold
    result["ml_score"] = scores
    return result


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parent.parent / "data" / "flight_with_anomalies.csv"
    df = pd.read_csv(csv_path)

    print("Trenowanie Isolation Forest na czystym locie...")
    result = detect_ml_anomalies(df)

    n_alerts = result["alert_ml"].sum()
    print(f"\nPróbek z alertem ML: {n_alerts}/{len(result)}")

    # Statystyki per typ anomalii
    print("\nSkuteczność per typ anomalii:")
    for atype in result["anomaly_type"].unique():
        if atype == "none":
            continue
        subset = result[result["anomaly_type"] == atype]
        detected = subset["alert_ml"].sum()
        print(f"  {atype:20s}: wykryto {detected}/{len(subset)} próbek")

    # False positives na "czystych" próbkach
    clean = result[result["anomaly_type"] == "none"]
    false_pos = clean["alert_ml"].sum()
    print(f"\nFałszywe alarmy na czystych próbkach: {false_pos}/{len(clean)}")
