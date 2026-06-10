"""
Demo strumieniowe: spina TelemetryGenerator (watek) z StreamConsumer i pokazuje
alerty wykrywane w czasie rzeczywistym, a na koncu podsumowanie skutecznosci.

Uruchom z katalogu projektu:
    python -m streaming.run_stream [algorithm] [--cases N] [--speed F] [--replicate R] [--jsonl]

Wymaga wczesniej zapisanego modelu (models/<algorithm>.pkl) - jesli go nie ma,
najpierw uruchom:  python run_all.py <algorithm>
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import load_dataset
from detection.ml import AnomalyDetector

from streaming.generator import TelemetryGenerator
from streaming.consumer import StreamConsumer, VALID_LAYERS
from streaming.alerts import ConsoleAlertSink, JsonlAlertSink, MultiSink


def main() -> None:
    parser = argparse.ArgumentParser(description="Strumieniowa detekcja anomalii w telemetrii drona")
    parser.add_argument("algorithm", nargs="?", default="random_forest",
                        help="algorytm warstwy 3 (model w models/<algorithm>.pkl)")
    parser.add_argument("--cases", type=int, default=5, help="ile lotow odtworzyc")
    parser.add_argument("--max-samples", type=int, default=800,
                        help="twardy limit liczby probek (0 = bez limitu). RF ~0.1s/probke.")
    parser.add_argument("--speed", type=float, default=50.0, help="przyspieszenie czasu (1.0 = realny)")
    parser.add_argument("--replicate", type=int, default=3, help="ktory replikat odtwarzac (zbior testowy)")
    parser.add_argument("--layers", default="all",
                        help="warstwy detekcji: 'all' lub lista po przecinku z {rules,statistical,ml}, "
                             "np. --layers ml albo --layers rules,statistical")
    parser.add_argument("--ml-threshold", type=float, default=None,
                        help="prog decyzyjny ML (supervised: alert gdy p>=prog, domyslnie 0.5; "
                             "unsupervised: alert gdy score<prog). Brak = domyslny modelu.")
    parser.add_argument("--jsonl", action="store_true", help="zapisuj alerty takze do data/alerts.jsonl")
    args = parser.parse_args()

    # Ustalenie warstw: 'all' -> wszystkie, w przeciwnym razie lista po przecinku.
    if args.layers.strip().lower() == "all":
        layers = list(VALID_LAYERS)
    else:
        layers = [x.strip().lower() for x in args.layers.split(",") if x.strip()]

    # Model warstwy 3 wczytujemy tylko gdy 'ml' jest aktywne.
    detector = None
    if "ml" in layers:
        model_path = PROJECT_ROOT / "models" / f"{args.algorithm}.pkl"
        if not model_path.exists():
            print(f"Brak modelu: {model_path}")
            print(f"Najpierw wytrenuj i zapisz model:  python run_all.py {args.algorithm}")
            sys.exit(1)
        print(f"Wczytuje model: {model_path}")
        detector = AnomalyDetector.load(model_path)

    print("Wczytuje dataset...")
    df = load_dataset()

    # Sink: konsola, opcjonalnie + plik JSONL.
    sink = ConsoleAlertSink()
    if args.jsonl:
        jsonl_path = PROJECT_ROOT / "data" / "alerts.jsonl"
        sink = MultiSink(sink, JsonlAlertSink(jsonl_path))
        print(f"Alerty zapisywane takze do: {jsonl_path}")

    q: "queue.Queue" = queue.Queue(maxsize=1000)
    generator = TelemetryGenerator(
        out_queue=q, df=df, replicate=args.replicate,
        n_cases=args.cases, speed_factor=args.speed,
        max_samples=args.max_samples or None,
    )
    consumer = StreamConsumer(detector=detector, in_queue=q, sink=sink,
                              enabled_layers=layers, ml_threshold=args.ml_threshold)

    algo_info = args.algorithm if "ml" in consumer.layers else "(bez ML)"
    print(f"Start strumienia: {args.cases} lotow, {generator.n_samples:,} probek, "
          f"speed={args.speed}x, warstwy={'+'.join(consumer.layers)}, algorytm={algo_info}")
    print("=" * 70)

    # Generator w osobnym watku, konsument w glownym.
    gen_thread = threading.Thread(target=generator.run, name="generator", daemon=True)
    gen_thread.start()
    consumer.run()
    gen_thread.join()

    sink.close()

    print("=" * 70)
    s = consumer.summary()
    print("PODSUMOWANIE (online, vs ground-truth label):")
    print(f"  Probki:    {s['n_samples']:,}")
    print(f"  Alerty:    {s['n_alerts']:,}")
    print(f"  TP={s['tp']:,}  FP={s['fp']:,}  FN={s['fn']:,}  TN={s['tn']:,}")
    print(f"  Precision: {s['precision']:.3f}")
    print(f"  Recall:    {s['recall']:.3f}")
    print(f"  F1:        {s['f1']:.3f}")


if __name__ == "__main__":
    main()
