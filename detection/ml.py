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
  dt        : mean, std, max, min                 (odstepy czasowe miedzy probkami)
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
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from .features import compute_features


# --- AnomalyDetector ------------------------------------------------------

SUPPORTED_ALGOS = {"isolation_forest", "one_class_svm", "lof", "random_forest", "xgboost"}
UNSUPERVISED = {"isolation_forest", "one_class_svm", "lof"}
SUPERVISED = {"random_forest", "xgboost"}

# Domyslny limit probek treningowych dla algorytmow O(n^2).
# Wartosc None oznacza "brak limitu" - algorytm dostaje wszystko.
DEFAULT_MAX_TRAIN_SAMPLES = {
    "one_class_svm": 50_000,
    "lof":           50_000,
    "isolation_forest": None,
    "random_forest":    None,
    "xgboost":          None,
}


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
    max_train_samples: int | None = None  # None -> uzyj DEFAULT_MAX_TRAIN_SAMPLES
    params: dict = field(default_factory=dict)

    model: object = field(default=None, init=False)
    threshold: float | None = field(default=None, init=False)
    optimal_threshold: float | None = field(default=None, init=False)
    feature_names: list[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        if self.algorithm not in SUPPORTED_ALGOS:
            raise ValueError(f"Nieznany algorytm: {self.algorithm}. Dostepne: {SUPPORTED_ALGOS}")
        if self.max_train_samples is None:
            self.max_train_samples = DEFAULT_MAX_TRAIN_SAMPLES.get(self.algorithm)

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
            # max_depth=None rosnie az do czystych lisci -> na milionach probek
            # pickle puchnie do dziesiatkow GB. Ograniczamy glebokosc i wielkosc
            # lisci: model schodzi do ~kilkudziesieciu MB przy minimalnej stracie AUC.
            return RandomForestClassifier(
                n_estimators=self.params.get("n_estimators", 200),
                max_depth=self.params.get("max_depth", 16),
                min_samples_leaf=self.params.get("min_samples_leaf", 20),
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
            # scale_pos_weight = liczba negatives / liczba positives
            # Bedzie ustawiony w fit() gdy znamy rozkład danych
            return XGBClassifier(
                n_estimators=self.params.get("n_estimators", 300),
                max_depth=self.params.get("max_depth", 8),
                learning_rate=self.params.get("learning_rate", 0.05),
                subsample=self.params.get("subsample", 0.9),
                colsample_bytree=self.params.get("colsample_bytree", 0.9),
                eval_metric=self.params.get("eval_metric", "logloss"),
                random_state=self.params.get("random_state", 42),
                n_jobs=-1,
                verbosity=0,
            )
        raise ValueError(self.algorithm)

    def fit(
        self,
        df_train: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> "AnomalyDetector":
        """
        Trenuje detektor.
          - unsupervised : filtruje df_train do label==0 i uczy na czystych probkach
          - supervised   : uczy na calym df_train uzywajac kolumny 'label'

        Jezeli `features` podane (wynik compute_features() dla df_train),
        nie liczymy ich ponownie - przyspiesza istotnie gdy ten sam df_train
        idzie do wielu detektorow z rzedu (cache w notebooku).

        Jezeli `max_train_samples` jest ustawione i zbior po filtrze jest wiekszy,
        losowo podsemplujemy (random_state z self.params).
        """
        if features is None:
            features = compute_features(df_train, window=self.window)
        self.feature_names = list(features.columns)
        X = features.values

        self.model = self._build_estimator()
        rng = np.random.default_rng(self.params.get("random_state", 42))

        if self.algorithm in UNSUPERVISED:
            mask = df_train["label"].values == 0
            X_train = X[mask]
            X_train = self._maybe_subsample(X_train, rng)
            self.model.fit(X_train)
            # Prog z percentyla scores na zbiorze uzytym do treningu
            scores = self._raw_scores(X_train)
            self.threshold = float(np.percentile(scores, self.threshold_percentile))
        else:
            y = df_train["label"].astype(int).values
            X_train, y_train = self._maybe_subsample_xy(X, y, rng)
            
            # Dla XGBoost: ustaw scale_pos_weight na podstawie rozkładu klas
            if self.algorithm == "xgboost":
                n_neg = (y_train == 0).sum()
                n_pos = (y_train == 1).sum()
                if n_pos > 0:
                    scale_pos_weight = n_neg / n_pos
                    self.model.set_params(scale_pos_weight=scale_pos_weight)
            
            self.model.fit(X_train, y_train)
            
            # Dla supervised: oblicz optymalny próg na zbiorze treningowym
            # na podstawie maksymalnego F1, aby nie przesadzić z false
            # positives i nie zalewać wykresu alertami.
            scores_train = self._raw_scores(X_train)
            precision, recall, thresholds = precision_recall_curve(y_train, scores_train)
            if len(thresholds) > 0:
                f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
                best_idx = int(np.nanargmax(f1_scores))
                self.optimal_threshold = float(thresholds[best_idx])
            else:
                self.optimal_threshold = 0.5

        return self

    def _maybe_subsample(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.max_train_samples and len(X) > self.max_train_samples:
            idx = rng.choice(len(X), size=self.max_train_samples, replace=False)
            return X[idx]
        return X

    def _maybe_subsample_xy(self, X, y, rng):
        if self.max_train_samples and len(X) > self.max_train_samples:
            idx = rng.choice(len(X), size=self.max_train_samples, replace=False)
            return X[idx], y[idx]
        return X, y

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

    def predict(
        self,
        df_test: pd.DataFrame,
        features: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Zwraca df_test z dodanymi kolumnami:
          - alert_ml  : bool
          - ml_score  : float (semantyka zalezna od algorytmu)
            * unsupervised : nizszy = bardziej anomalna (jak w IsolationForest.score_samples)
            * supervised   : p(label=1) (wyzszy = bardziej anomalna)

        Jezeli `features` podane, nie liczymy ich ponownie (cache).
        """
        if self.model is None:
            raise RuntimeError("Wywolaj fit() przed predict()")

        if features is None:
            features = compute_features(df_test, window=self.window)
        if list(features.columns) != self.feature_names:
            warnings.warn("Kolejnosc cech zmieniona - poprawiam.")
            features = features[self.feature_names]
        X = features.values

        scores = self._raw_scores(X)
        if self.algorithm in UNSUPERVISED:
            alerts = scores < self.threshold
        else:
            # Dla supervised: użyj optymalnego progu (jeśli dostępny)
            threshold = self.optimal_threshold if self.optimal_threshold is not None else 0.5
            alerts = scores >= threshold

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
