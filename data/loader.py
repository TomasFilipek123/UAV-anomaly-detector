"""
Loader datasetu "Drone Telemetry Tampering Dataset v2" (Kaggle).

Dataset zawiera 720 lotów (case_id), kazdy z 4 replikatami (replicate 0..3).
Etykiety:
  - label        : 0 / 1  (binarna)
  - tamper_type  : 10 klas (normal + 9 typow manipulacji)
  - profile      : subtle / balanced / strong  (intensywnosc anomalii)

Konwencje preprocessingu:
  - Timestamp jest w formacie ISO 8601 (np. 2024-09-15T20:05:41.726000+00:00) -
    parsujemy do pandas.Timestamp z timezone UTC.
  - Wiersze z row_idx == 0 odrzucamy (timestamp 1970, niespojny).
  - Train/test split: replicate in {0,1,2} -> train, replicate == 3 -> test.
  - Dla kazdego case_id dodajemy kolumne `t_rel` (czas od startu lotu w sekundach,
    float), bo absolutny timestamp jest niewygodny w rolling/diff.
"""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "drone_telemetry_v2.csv"

EXPECTED_COLUMNS = {
    "profile", "replicate", "case_id", "case_name", "row_idx",
    "label", "tamper_type", "timestamp",
    "latitude", "longitude", "altitude", "speed", "heading",
    "source", "original_row_idx",
}

TAMPER_TYPES = [
    "normal", "speed_inconsistency", "timestamp_drift", "injection",
    "deletion_gap", "heading_inconsistency", "precision_rounding",
    "combined", "coordinate_jump", "altitude_spike",
]

PROFILES = ["subtle", "balanced", "strong"]


def load_dataset(
    path: str | Path = DEFAULT_CSV,
    usecols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Wczytuje CSV, weryfikuje kolumny, odrzuca row_idx == 0, sortuje po (case_id, row_idx),
    dodaje kolumne `t_rel` (czas od startu lotu w sekundach).

    Parameters
    ----------
    path : sciezka do pliku CSV z Kaggle
    usecols : opcjonalna lista kolumn do wczytania (oszczednosc RAM przy 8M wierszy)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono datasetu: {path}\n"
            f"Pobierz z Kaggle (drone-telemetry-tampering-dataset-v2) "
            f"i umiesc plik w {path.parent}."
        )

    df = pd.read_csv(
        path,
        usecols=list(usecols) if usecols else None,
    )

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing and usecols is None:
        raise ValueError(f"Brakujace kolumny w datasecie: {sorted(missing)}")

    # Jawne parsowanie timestamp (parse_dates w read_csv bywa zawodny
    # przy mieszanym formacie ISO 8601 z offset strefy). utc=True normalizuje
    # do UTC i daje dtype datetime64[ns, UTC].
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # row_idx == 0 ma timestamp 1970 (epoch 0) - usuwamy z kazdego lotu
    df = df[df["row_idx"] > 0].copy()

    df = df.sort_values(["case_id", "row_idx"], kind="stable").reset_index(drop=True)

    # Czas od startu lotu w sekundach (float).
    # transform zwraca Timedelta - konwertujemy do total_seconds().
    df["t_rel"] = df.groupby("case_id")["timestamp"].transform(
        lambda s: (s - s.min()).dt.total_seconds()
    )

    return df


def split_by_replicate(
    df: pd.DataFrame,
    test_replicate: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardowy split: replicate != test_replicate -> train, replicate == test_replicate -> test.
    """
    train = df[df["replicate"] != test_replicate].reset_index(drop=True)
    test = df[df["replicate"] == test_replicate].reset_index(drop=True)
    return train, test


def clean_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Zwraca tylko wiersze bez anomalii (label == 0). Uzywane do treningu unsupervised."""
    return df[df["label"] == 0].reset_index(drop=True)


def sample_rate_per_case(df: pd.DataFrame) -> pd.Series:
    """
    Dla kazdego case_id zwraca median(diff(timestamp)) w sekundach - typowy
    odstep miedzy probkami. Pozwala oszacowac sample rate (1/diff).
    """
    return df.groupby("case_id")["timestamp"].apply(
        lambda s: float(np.median(np.diff(s.values).astype("timedelta64[ns]").astype(np.int64))) / 1e9
        if len(s) > 1 else np.nan
    )


def summarize(df: pd.DataFrame) -> None:
    """Drukuje krotkie podsumowanie wczytanego datasetu."""
    print(f"Wierszy: {len(df):,}")
    print(f"Case'ow: {df['case_id'].nunique()}")
    print(f"Replicate: {sorted(df['replicate'].unique())}")
    print(f"Anomalii (label=1): {df['label'].sum():,} ({100 * df['label'].mean():.1f}%)")
    print("\nLiczba probek per tamper_type:")
    print(df["tamper_type"].value_counts().to_string())
    print("\nLiczba probek per profile:")
    print(df["profile"].value_counts().to_string())

    dt = sample_rate_per_case(df)
    print(f"\nTypowy odstep timestamp miedzy probkami (median, sekundy):")
    print(f"  min:    {dt.min():.3f}")
    print(f"  median: {dt.median():.3f}")
    print(f"  max:    {dt.max():.3f}")


if __name__ == "__main__":
    df = load_dataset()
    summarize(df)
    train, test = split_by_replicate(df)
    print(f"\nTrain (replicate 0,1,2): {len(train):,}")
    print(f"Test  (replicate 3):     {len(test):,}")
