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

    # Odstep czasowy miedzy probkami (sekundy).
    # timestamp jest pd.Timestamp -> diff() to Timedelta, konwertujemy do sekund.
    dt_raw = case_df["timestamp"].diff().dt.total_seconds()
    dt = dt_raw.replace(0, np.nan)  # do gps_speed - unikamy dzielenia przez 0

    # Roznica miedzy predkoscia sensora a predkoscia z GPS.
    gps_speed = gps_step / dt
    feats["speed_vs_gps"] = (case_df["speed"] - gps_speed).abs()
    feats["speed_vs_gps"] = feats["speed_vs_gps"].rolling(window=window, min_periods=min_p).mean()

    # Cechy z odstepow czasowych - celuja w tampery, ktore zaburzaja regularnosc
    # probkowania, a nie ich wartosci fizyczne (recall ~0% bez tego):
    #   deletion_gap -> duza luka czasowa (dt_max)
    #   injection    -> zdublowany/wstrzyniety timestamp, dt ~ 0 (dt_min)
    #   timestamp_drift -> nieregularne dt (dt_std)
    # Uzywamy dt_raw (z zerami), bo dt==0 to wlasnie sygnal injection.
    dt_roll = dt_raw.rolling(window=window, min_periods=min_p)
    feats["dt_mean"] = dt_roll.mean()
    feats["dt_std"] = dt_roll.std()
    feats["dt_max"] = dt_roll.max()
    feats["dt_min"] = dt_roll.min()

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
    # Zabezpieczenie: kanaly numeryczne moga przyjsc jako object (string), jesli
    # df nie przeszedl przez loader - wtedy diff()/haversine wywala sie z
    # "unsupported operand -: 'str' and 'str'". Kopiujemy tylko gdy trzeba.
    obj_cols = [c for c in ("altitude", "speed", "heading", "latitude", "longitude")
                if c in df.columns and df[c].dtype == object]
    if obj_cols:
        df = df.copy()
        for c in obj_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

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
            return XGBClassifier(
                n_estimators=self.params.get("n_estimators", 200),
                max_depth=self.params.get("max_depth", 6),
                learning_rate=self.params.get("learning_rate", 0.1),
                eval_metric=self.params.get("eval_metric", "logloss"),
                random_state=self.params.get("random_state", 42),
                n_jobs=-1,
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
            self.model.fit(X_train, y_train)

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
