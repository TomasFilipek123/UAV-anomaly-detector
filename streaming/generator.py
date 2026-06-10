"""
TelemetryGenerator: odtwarza dataset probka po probce, symulujac strumien w czasie
rzeczywistym.

Zrodlem danych jest istniejacy CSV (data/drone_telemetry_v2.csv) wczytany loaderem -
dzieki temu mamy ground-truth (kolumna `label`), a wyniki detekcji online sa
porownywalne z trybem wsadowym (run_all.py).

Pacing: w obrebie jednego lotu (case_id) odstep miedzy probkami odtwarzamy z kolumny
`t_rel` (czas od startu w sekundach), skrocony o `speed_factor`. Miedzy lotami zegar
sie resetuje - nie odtwarzamy przerw miedzy roznymi case'ami.
"""

from __future__ import annotations

import queue
import time
from typing import Iterable

import pandas as pd

from data.loader import load_dataset, split_by_replicate


# Sentinel wrzucany na koniec strumienia - sygnal dla konsumenta, ze danych juz nie bedzie.
END_OF_STREAM = None


class TelemetryGenerator:
    """
    Parameters
    ----------
    out_queue : queue.Queue
        Kolejka, na ktora trafiaja kolejne probki (dict) i na koncu END_OF_STREAM.
    df : pd.DataFrame, optional
        Gotowy DataFrame (np. juz po split_by_replicate). Jesli None - wczytujemy CSV.
    replicate : int, optional
        Filtr po replikacie (domyslnie 3 = standardowy zbior testowy).
    case_ids : Iterable, optional
        Konkretne case_id do odtworzenia. Jesli None - bierzemy `n_cases` pierwszych.
    n_cases : int, optional
        Ile pierwszych lotow odtworzyc (gdy case_ids nie podano). None = wszystkie.
    speed_factor : float
        Przyspieszenie czasu. 1.0 = realny czas, 50.0 = 50x szybciej.
    max_sleep : float
        Sufit pojedynczej pauzy w sekundach (chroni przed dluga luka w danych).
    max_samples : int, optional
        Twardy limit liczby wyslanych probek (np. na potrzeby szybkiego demo).
        None = bez limitu.
    """

    def __init__(
        self,
        out_queue: "queue.Queue",
        df: pd.DataFrame | None = None,
        replicate: int | None = 3,
        case_ids: Iterable | None = None,
        n_cases: int | None = 5,
        speed_factor: float = 50.0,
        max_sleep: float = 0.5,
        max_samples: int | None = None,
    ):
        if df is None:
            df = load_dataset()
        if replicate is not None and "replicate" in df.columns:
            _, df = split_by_replicate(df, test_replicate=replicate)

        if case_ids is not None:
            df = df[df["case_id"].isin(list(case_ids))]
        elif n_cases is not None:
            keep = df["case_id"].drop_duplicates().head(n_cases)
            df = df[df["case_id"].isin(keep)]

        self.df = df.sort_values(["case_id", "row_idx"], kind="stable").reset_index(drop=True)
        if max_samples is not None:
            self.df = self.df.head(max_samples)

        self.out_queue = out_queue
        self.speed_factor = max(speed_factor, 1e-9)
        self.max_sleep = max_sleep

    def run(self) -> None:
        """Odtwarza wszystkie wybrane loty na kolejke, konczac sentinelem END_OF_STREAM."""
        prev_case = None
        prev_t = None

        for row in self.df.to_dict("records"):
            case = row["case_id"]
            t_rel = row.get("t_rel")

            # Pacing tylko w obrebie jednego lotu; przy zmianie case resetujemy zegar.
            if case == prev_case and prev_t is not None and t_rel is not None:
                delay = (t_rel - prev_t) / self.speed_factor
                if delay > 0:
                    time.sleep(min(delay, self.max_sleep))

            self.out_queue.put(row)
            prev_case, prev_t = case, t_rel

        self.out_queue.put(END_OF_STREAM)

    @property
    def n_samples(self) -> int:
        return len(self.df)
