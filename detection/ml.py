"""
Warstwa 3: detekcja anomalii przez modele ML na cechach okienkowych.

Wspierane algorytmy:
  unsupervised : Isolation Forest, One-Class SVM, Local Outlier Factor
  supervised   : Random Forest, XGBoost

Cechy okienkowe (15+) liczone niezaleznie per case_id:
  altitude  : mean, std, range, slope
  speed     : mean, std, range, slope
  heading   : diff_mean, diff_std, diff_abs_max  (na cyklicznych roznicach)
  gps_step  : mean, std, max                     (haversine miedzy probkami)
  speed_vs_gps : roznica miedzy speed sensora a predkoscia liczona z GPS
  sample_var_score : suma std-ow (sygnal "zamrozenia")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


# --- Cechy okienkowe ------------------------------------------------------

def _circular_diff(series: pd.Series) -> pd.Series:
    raw = series.diff()
    return ((raw + 180) % 360) - 180


def _haversine_step(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    R = 6_371_000.0
    lat1 = np.radians(lat[:-1])
    lat2 = np.radians(lat[1:])
    dlat = lat2 - lat1
    dlon = np.radians(lon[1:] - lon[:-1])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    d = 2 * R * np.arcsin(np.sqrt(a))
    return np.concatenate([[np.nan], d])


def _features_one_case(case_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Liczy cechy okienkowe dla pojedynczego lotu (case_id)."""
    feats = pd.DataFrame(index=case_df.index)
    min_p = max(2, window // 2)

    for col, prefix in [("altitude", "alt"), ("speed", "spd")]:
        s = case_df[col]
        roll = s.rolling(window=window, min_periods=min_p)
        feats[f"{prefix}_mean"] = roll.mean()
        feats[f"{prefix}_std"] = roll.std()
        feats[f"{prefix}_range"] = roll.max() - roll.min()
        feats[f"{prefix}_slope"] = (s - s.shift(window - 1)) / window

    hdg_diff = _circular_diff(case_df["heading"])
    feats["hdg_diff_mean"] = hdg_diff.rolling(window=window, min_periods=min_p).mean()
    feats["hdg_diff_std"] = hdg_diff.rolling(window=window, min_periods=min_p).std()
    feats["hdg_diff_abs_max"] = hdg_diff.abs().rolling(window=window, min_periods=min_p).max()

    gps_step = pd.Series(
        _haversine_step(case_df["latitude"].values, case_df["longitude"].values),
        index=case_df.index,
    )
    feats["gps_step_mean"] = gps_step.rolling(window=window, min_periods=min_p).mean()
    feats["gps_step_std"] = gps_step.rolling(window=window, min_periods=min_p).std()
    feats["gps_step_max"] = gps_step.rolling(window=window, min_periods=min_p).max()

    # Roznica miedzy predkoscia sensora a predkoscia z GPS
    # (zaklada sample rate ~1 Hz - jesli inny, korekta byc moze potrzebna)
    dt = case_df["timestamp"].diff().replace(0, np.nan)
    gps_speed = gps_step / dt
    feats["speed_vs_gps"] = (case_df["speed"] - gps_speed).abs()
    feats["speed_vs_gps"] = feats["speed_vs_gps"].rolling(window=window, min_periods=min_p).mean()

    feats["sample_var_score"] = (
        feats["alt_std"].fillna(0)
        + feats["spd_std"].fillna(0)
        + feats["hdg_diff_std"].fillna(0)
        + feats["gps_step_std"].fillna(0)
    )

    return feats


def compute_features(df: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """
    Liczy cechy okienkowe dla calego DataFrame, grupujac per case_id.
    Wypelnia NaN-y (bfill + 0) na koncu - kazda probka ma kompletny wektor cech.
    """
    parts = []
    for _, case_df in df.groupby("case_id", sort=False):
        parts.append(_features_one_case(case_df, window))
    feats = pd.concat(parts).sort_index()
    feats = feats.bfill().fillna(0)
    return feats


# --- AnomalyDetector ------------------------------------------------------

SUPPORTED_ALGOS = {"isolation_forest", "one_class_svm", "lof", "random_forest", "xgboost"}
UNSUPERVISED = {"isolation_forest", "one_class_svm", "lof"}
SUPERVISED = {"random_forest", "xgboost"}


@dataclass
class AnomalyDetector:
    """
    Jednolity interfejs dla 5 algorytmow detekcji anomalii.

    Unsupervised (trenowane na samych label==0):
      - 'isolation_forest'
      - 'one_class_svm'
      - 'lof'        (uwaga: trenuje sie inaczej - novelty=True)

    Supervised (wymagaja kolumny 'label' w treningu):
      - 'random_forest'
      - 'xgboost'    (opcjonalny - tylko jesli pakiet zainstalowany)

    Atrybuty po fit:
      - model         : wytrenowany estymator
      - threshold     : prog na score (tylko unsupervised)
      - feature_names : lista cech (kolejnosc istotna dla predict)
    """

    algorithm: str
    window: int = 8
    threshold_percentile: float = 2.0
    params: dict = field(default_factory=dict)

    model: object = field(default=None, init=False)
    threshold: float | None = field(default=None, init=False)
    feature_names: list[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        if self.algorithm not in SUPPORTED_ALGOS:
            raise ValueError(f"Nieznany algorytm: {self.algorithm}. Dostepne: {SUPPORTED_ALGOS}")

    def _build_estimator(self):
        a = self.algorithm
        if a == "isolation_forest":
            return IsolationForest(
                n_estimators=self.params.get("n_estimators", 150),
                contamination="auto",
                random_state=self.params.get("random_state", 42),
                n_jobs=-1,
            )
        if a == "one_class_svm":
            return OneClassSVM(
                kernel=self.params.get("kernel", "rbf"),
                gamma=self.params.get("gamma", "scale"),
                nu=self.params.get("nu", 0.05),
            )
        if a == "lof":
            return LocalOutlierFactor(
                n_neighbors=self.params.get("n_neighbors", 20),
                novelty=True,  # konieczne zeby moc wolac predict na nowych danych
                n_jobs=-1,
            )
        if a == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.params.get("n_estimators", 200),
                max_depth=self.params.get("max_depth", None),
                class_weight=self.params.get("class_weight", "balanced"),
                random_state=self.params.get("random_state", 42),
                n_jobs=-1,
            )
        if a == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as e:
                raise ImportError(
                    "Brak pakietu xgboost. Zainstaluj: pip install xgboost"
                ) from e
            return XGBClassifier(
                n_estimators=self.params.get("n_estimators", 200),
                max_depth=self.params.get("max_depth", 6),
                learning_rate=self.params.get("learning_rate", 0.1),
                eval_metric=self.params.get("eval_metric", "logloss"),
                random_state=self.params.get("random_state", 42),
                n_jobs=-1,
            )
        raise ValueError(self.algorithm)

    def fit(self, df_train: pd.DataFrame) -> "AnomalyDetector":
        """
        Trenuje detektor.
          - unsupervised : filtruje df_train do label==0 i uczy na czystych probkach
          - supervised   : uczy na calym df_train uzywajac kolumny 'label'
        """
        feats = compute_features(df_train, window=self.window)
        self.feature_names = list(feats.columns)
        X = feats.values

        self.model = self._build_estimator()

        if self.algorithm in UNSUPERVISED:
            mask = df_train["label"].values == 0
            X_train = X[mask]
            self.model.fit(X_train)
            # Prog z percentyla scores na czystym zbiorze treningowym
            scores = self._raw_scores(X_train)
            self.threshold = float(np.percentile(scores, self.threshold_percentile))
        else:
            y = df_train["label"].astype(int).values
            self.model.fit(X, y)

        return self

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Zwraca surowe scores (nizsze = bardziej anomalna) dla unsupervised
        albo p(label=1) dla supervised.
        """
        if self.algorithm == "isolation_forest":
            return self.model.score_samples(X)
        if self.algorithm == "one_class_svm":
            return self.model.score_samples(X)
        if self.algorithm == "lof":
            return self.model.score_samples(X)
        if self.algorithm in SUPERVISED:
            return self.model.predict_proba(X)[:, 1]
        raise ValueError(self.algorithm)

    def predict(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """
        Zwraca df_test z dodanymi kolumnami:
          - alert_ml  : bool
          - ml_score  : float (semantyka zalezna od algorytmu)
            * unsupervised : nizszy = bardziej anomalna (jak w IsolationForest.score_samples)
            * supervised   : p(label=1) (wyzszy = bardziej anomalna)
        """
        if self.model is None:
            raise RuntimeError("Wywolaj fit() przed predict()")

        feats = compute_features(df_test, window=self.window)
        if list(feats.columns) != self.feature_names:
            warnings.warn("Kolejnosc cech zmieniona - poprawiam.")
            feats = feats[self.feature_names]
        X = feats.values

        scores = self._raw_scores(X)
        if self.algorithm in UNSUPERVISED:
            alerts = scores < self.threshold
        else:
            alerts = scores >= 0.5

        result = df_test.copy()
        result["alert_ml"] = alerts
        result["ml_score"] = scores
        return result

    # --- serializacja ---

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "AnomalyDetector":
        with open(path, "rb") as f:
            return pickle.load(f)


# --- Zachowanie kompatybilnosci z run_all.py ------------------------------

def detect_ml_anomalies(
    df_test: pd.DataFrame,
    df_train: pd.DataFrame,
    algorithm: str = "isolation_forest",
    **kwargs,
) -> pd.DataFrame:
    """Trenuje detektor na df_train i zwraca df_test z kolumnami alert_ml/ml_score."""
    det = AnomalyDetector(algorithm=algorithm, **kwargs).fit(df_train)
    return det.predict(df_test)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.loader import load_dataset, split_by_replicate

    print("Loading dataset...")
    df = load_dataset()
    train, test = split_by_replicate(df)
    print(f"Train: {len(train):,}  Test: {len(test):,}")

    print("\nIsolation Forest:")
    det = AnomalyDetector(algorithm="isolation_forest").fit(train)
    out = det.predict(test)
    print(f"  alerts: {out['alert_ml'].sum():,}/{len(out):,}")
    print(pd.crosstab(out["alert_ml"], out["label"]))
