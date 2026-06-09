"""
Reprezentacja alertu i miejsca docelowe jego emisji (sinki).

Alert powstaje w StreamConsumer, gdy ktorakolwiek z 3 warstw detekcji zaalarmuje
na biezacej probce. Sink decyduje, co z alertem zrobic (wydruk, zapis do pliku).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Alert:
    """
    Pojedynczy alert wykryty na strumieniu.

      timestamp : czas probki (str ISO lub Timestamp) - kiedy wystapila anomalia
      case_id   : identyfikator lotu
      row_idx   : numer probki w locie
      layers    : ktore warstwy zaalarmowaly, np. ["rules", "statistical", "ml"]
      reasons   : sklejone powody (alert_reasons | change_reasons), rozdzielone '|'
      ml_score  : surowy score modelu (semantyka zalezna od algorytmu)
      label     : ground-truth (0/1) jesli znane - tylko do ewaluacji online
    """

    timestamp: object
    case_id: object
    row_idx: object
    layers: list[str] = field(default_factory=list)
    reasons: str = ""
    ml_score: float | None = None
    label: int | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = str(self.timestamp)
        return d


class AlertSink:
    """Bazowy interfejs sinka. Podklasy nadpisuja emit()."""

    def emit(self, alert: Alert) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Domyslnie nic - sinki z zasobami (pliki) nadpisuja."""
        pass


class ConsoleAlertSink(AlertSink):
    """Drukuje alert na stdout w zwiezlej, czytelnej formie."""

    def emit(self, alert: Alert) -> None:
        score = f"{alert.ml_score:.3f}" if alert.ml_score is not None else "-"
        layers = "+".join(alert.layers) if alert.layers else "-"
        gt = ""
        if alert.label is not None:
            gt = "  [TP]" if alert.label == 1 else "  [FP]"
        print(
            f"[ALERT {alert.timestamp}] case={alert.case_id} row={alert.row_idx} "
            f"warstwy={layers} score={score} powody={alert.reasons or '-'}{gt}"
        )


class JsonlAlertSink(AlertSink):
    """Dopisuje kazdy alert jako linie JSON do pliku (append-only)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def emit(self, alert: Alert) -> None:
        self._fh.write(json.dumps(alert.to_dict(), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class MultiSink(AlertSink):
    """Rozsyla alert do kilku sinkow naraz (np. konsola + plik)."""

    def __init__(self, *sinks: AlertSink):
        self.sinks = list(sinks)

    def emit(self, alert: Alert) -> None:
        for s in self.sinks:
            s.emit(alert)

    def close(self) -> None:
        for s in self.sinks:
            s.close()
