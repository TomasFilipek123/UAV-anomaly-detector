"""
StreamConsumer: czyta probki ze strumienia i przepuszcza kazda przez pelny pipeline
detekcji (3 warstwy), emitujac alert gdy ktorakolwiek warstwa zaalarmuje.

Wyzwanie: warstwy 2 (statystyki kroczace, okno 30) i 3 (ML, okno 8) sa OKIENKOWE -
licza sie na ostatnich N probkach danego lotu. W trybie strumieniowym nie mamy calego
DataFrame, wiec utrzymujemy bufor kroczacy (deque) per case_id. Po kazdej nowej probce
budujemy mini-DataFrame z bufora, uruchamiamy na nim ISTNIEJACE funkcje detekcji bez
zmian i bierzemy wynik tylko dla ostatniego wiersza. Dzieki temu reuzywamy cala logike
rolling/groupby z detection/* zamiast ja przepisywac.
"""

from __future__ import annotations

import queue
from collections import deque

import pandas as pd

from detection.rules import detect_threshold_violations
from detection.statistical import detect_sudden_changes, DEFAULT_CONFIG
from detection.ml import compute_features, SUPERVISED

from streaming.alerts import Alert, AlertSink
from streaming.generator import END_OF_STREAM


# Kanaly numeryczne, ktore moga przyjsc jako string i wymagaja koercji.
_NUMERIC_COLS = ["altitude", "speed", "heading", "latitude", "longitude"]

# Dostepne warstwy detekcji (w kolejnosci pipeline).
VALID_LAYERS = ("rules", "statistical", "ml")


class StreamConsumer:
    """
    Parameters
    ----------
    detector : AnomalyDetector
        Wczytany (wytrenowany) model warstwy 3.
    in_queue : queue.Queue
        Kolejka z probkami (dict) i sentinelem END_OF_STREAM na koncu.
    sink : AlertSink
        Gdzie emitowac alerty.
    buffer_size : int, optional
        Dlugosc bufora kroczacego per case_id. Domyslnie max(okno_stat, okno_ml) + margines,
        tak by warstwa statystyczna miala pelne okno (30 probek).
    stat_config : dict, optional
        Konfiguracja warstwy statystycznej (domyslnie DEFAULT_CONFIG).
    enabled_layers : iterable[str], optional
        Ktore warstwy uruchamiac: podzbior {"rules","statistical","ml"}.
        Domyslnie wszystkie 3. Pozwala np. przepuscic strumien tylko przez ML.
    ml_threshold : float, optional
        Prog decyzyjny warstwy ML. None = domyslny modelu (supervised: 0.5 na
        p(anomalia); unsupervised: prog wyuczony z percentyla). Podanie wartosci
        nadpisuje decyzje: supervised -> alert gdy score >= prog; unsupervised ->
        alert gdy score < prog.
    """

    def __init__(
        self,
        detector,
        in_queue: "queue.Queue",
        sink: AlertSink,
        buffer_size: int | None = None,
        stat_config: dict | None = None,
        enabled_layers: "Iterable[str] | None" = None,
        ml_threshold: float | None = None,
    ):
        self.detector = detector
        self.in_queue = in_queue
        self.sink = sink
        self.stat_config = stat_config or DEFAULT_CONFIG
        self.ml_threshold = ml_threshold

        # Walidacja i ustalenie aktywnych warstw (zachowujemy kanoniczna kolejnosc).
        if enabled_layers is None:
            enabled_layers = VALID_LAYERS
        enabled = {str(x).lower() for x in enabled_layers}
        unknown = enabled - set(VALID_LAYERS)
        if unknown:
            raise ValueError(f"Nieznane warstwy: {sorted(unknown)}. Dozwolone: {VALID_LAYERS}")
        if not enabled:
            raise ValueError("Musisz wlaczyc co najmniej jedna warstwe.")
        if "ml" in enabled and detector is None:
            raise ValueError("Warstwa 'ml' wymaga wczytanego modelu (detector=None).")
        self.layers = tuple(l for l in VALID_LAYERS if l in enabled)

        if buffer_size is None:
            # Bufor dobieramy do najwiekszego okna wsrod WLACZONYCH warstw:
            # statistical=okno z-score (30), ml=okno cech, rules=bezstanowa (1).
            needed = [1]
            if "statistical" in enabled:
                needed.append(self.stat_config["window"])
            if "ml" in enabled:
                needed.append(detector.window)
            buffer_size = max(needed) + 10
        self.buffer_size = buffer_size

        self.buffers: dict[object, deque] = {}

        # Liczniki do ewaluacji online (wzgledem ground-truth label).
        self.n_samples = 0
        self.n_alerts = 0
        self.tp = self.fp = self.fn = self.tn = 0

    def _build_df(self, buf: deque) -> pd.DataFrame:
        """Buduje DataFrame z bufora i wymusza poprawne typy (timestamp + kanaly)."""
        df = pd.DataFrame(list(buf))
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        for col in _NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _process(self, row: dict) -> None:
        case = row["case_id"]
        buf = self.buffers.setdefault(case, deque(maxlen=self.buffer_size))
        buf.append(row)
        df = self._build_df(buf)

        reasons_threshold = ""
        reasons_change = ""
        ml_score = None
        layers = []

        # Warstwa 1: progi fizyczne (bezstanowa, ale liczymy na buforze dla jednolitosci).
        if "rules" in self.layers:
            r1 = detect_threshold_violations(df)
            if bool(r1["alert_threshold"].iloc[-1]):
                layers.append("rules")
                reasons_threshold = r1["alert_reasons"].iloc[-1]

        # Warstwa 2: statystyki kroczace (z-score + freeze) na buforze.
        if "statistical" in self.layers:
            r2 = detect_sudden_changes(df, self.stat_config)
            if bool(r2["alert_change"].iloc[-1]):
                layers.append("statistical")
                reasons_change = r2["change_reasons"].iloc[-1]

        # Warstwa 3: cechy okienkowe licza sie na CALYM buforze (rolling potrzebuje
        # okna), ale predykcje robimy tylko dla OSTATNIEGO wiersza - alert dotyczy
        # biezacej probki, a predict na 1 wierszu zamiast ~40 oszczedza czas modelu.
        if "ml" in self.layers:
            feats = compute_features(df, window=self.detector.window)
            r3 = self.detector.predict(df.iloc[[-1]], features=feats.iloc[[-1]])
            ml_score = float(r3["ml_score"].iloc[-1])
            if self.ml_threshold is None:
                is_ml_alert = bool(r3["alert_ml"].iloc[-1])
            elif self.detector.algorithm in SUPERVISED:
                # supervised: score = p(anomalia), wyzszy = bardziej anomalna.
                is_ml_alert = ml_score >= self.ml_threshold
            else:
                # unsupervised: nizszy score = bardziej anomalna.
                is_ml_alert = ml_score < self.ml_threshold
            if is_ml_alert:
                layers.append("ml")

        combined = bool(layers)
        label = row.get("label")
        label = int(label) if label is not None and pd.notna(label) else None

        # Ewaluacja online.
        self.n_samples += 1
        if combined:
            self.n_alerts += 1
        if label is not None:
            if combined and label == 1:
                self.tp += 1
            elif combined and label == 0:
                self.fp += 1
            elif not combined and label == 1:
                self.fn += 1
            else:
                self.tn += 1

        if combined:
            reasons = "|".join(p for p in (reasons_threshold, reasons_change) if p)
            self.sink.emit(Alert(
                timestamp=row.get("timestamp"),
                case_id=case,
                row_idx=row.get("row_idx"),
                layers=layers,
                reasons=reasons,
                ml_score=ml_score,
                label=label,
            ))

    def run(self) -> None:
        """Petla glowna: konsumuje kolejke az do sentinela END_OF_STREAM."""
        while True:
            row = self.in_queue.get()
            if row is END_OF_STREAM:
                break
            self._process(row)

    def summary(self) -> dict:
        """Zwraca liczniki + precision/recall/F1 dla alertow strumieniowych."""
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            "n_samples": self.n_samples,
            "n_alerts": self.n_alerts,
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "precision": precision, "recall": recall, "f1": f1,
        }
