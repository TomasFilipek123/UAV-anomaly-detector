"""
Demo strumieniowe: spina TelemetryGenerator (watek) z StreamConsumer i pokazuje
alerty wykrywane w czasie rzeczywistym, a na koncu podsumowanie skutecznosci.

Uruchom z katalogu projektu:
    python -m streaming.run_stream [algorithm] [--cases N] [--speed F] [--replicate R] [--jsonl] [--plot]

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

from streaming.generator import (
    TelemetryGenerator, make_synthetic_flights, DEFAULT_SYNTH_ANOMALIES, SYNTH_SHAPES,
)
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
    parser.add_argument("--synthetic", action="store_true",
                        help="tryb demo: zamiast datasetu generuj gladkie, stabilne loty "
                             "z 1-2 wyraznymi anomaliami (czytelne wykresy)")
    parser.add_argument("--syn-samples", type=int, default=200,
                        help="(tryb --synthetic) liczba probek na lot")
    parser.add_argument("--syn-anomalies", default=",".join(DEFAULT_SYNTH_ANOMALIES),
                        help="(tryb --synthetic) PULA typow anomalii po przecinku, z ktorej losujemy, "
                             "np. 'altitude_spike,coordinate_jump'; 'none' = bez anomalii")
    parser.add_argument("--syn-count", default="random",
                        help="(tryb --synthetic) ile anomalii na lot: 'random' (0-2 losowo) albo liczba")
    parser.add_argument("--syn-shape", default="cruise", choices=list(SYNTH_SHAPES),
                        help="(tryb --synthetic) profil lotu: 'cruise' (przelot) albo 'mission' "
                             "(start->przelot->ladowanie)")
    parser.add_argument("--syn-static", action="store_true",
                        help="(tryb --synthetic) wylacz losowy ksztalt per lot (wspolny baseline)")
    parser.add_argument("--syn-noise", type=float, default=1.0,
                        help="(tryb --synthetic) mnoznik szumu pomiarowego (0 = idealnie gladko)")
    parser.add_argument("--jsonl", action="store_true", help="zapisuj alerty takze do data/alerts/alerts.jsonl")
    parser.add_argument("--plot", action="store_true",
                        help="po zakonczeniu strumienia zapisz wykres PNG per lot "
                             "(data/plots/stream_case_<id>_plot.png) - telemetria + alerty warstw")
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

    if args.synthetic:
        syn_anoms = [] if args.syn_anomalies.strip().lower() == "none" else \
            [x.strip() for x in args.syn_anomalies.split(",") if x.strip()]
        syn_count = "random" if args.syn_count.strip().lower() == "random" else int(args.syn_count)
        print(f"Generuje {args.cases} syntetycznych lotow ({args.syn_samples} probek/lot), "
              f"shape={args.syn_shape}, anomalie/lot={syn_count}, pula={syn_anoms or 'brak'}...")
        df = make_synthetic_flights(
            n_cases=args.cases, n_samples=args.syn_samples,
            anomalies=syn_anoms, n_anomalies=syn_count, shape=args.syn_shape,
            vary=not args.syn_static, noise=args.syn_noise,
        )
    else:
        print("Wczytuje dataset...")
        df = load_dataset()

    # Sink: konsola, opcjonalnie + plik JSONL.
    sink = ConsoleAlertSink()
    if args.jsonl:
        alerts_dir = PROJECT_ROOT / "data" / "alerts"
        alerts_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = alerts_dir / "alerts.jsonl"
        sink = MultiSink(sink, JsonlAlertSink(jsonl_path))
        print(f"Alerty zapisywane takze do: {jsonl_path}")

    q: "queue.Queue" = queue.Queue(maxsize=1000)
    # W trybie synthetic df ma juz dokladnie wybrane loty (replicate=-1) i pelne
    # przebiegi - nie filtrujemy po replikacie ani nie ucinamy max_samples.
    generator = TelemetryGenerator(
        out_queue=q, df=df,
        replicate=None if args.synthetic else args.replicate,
        n_cases=None if args.synthetic else args.cases,
        speed_factor=args.speed,
        max_samples=None if args.synthetic else (args.max_samples or None),
    )
    consumer = StreamConsumer(detector=detector, in_queue=q, sink=sink,
                              enabled_layers=layers, ml_threshold=args.ml_threshold,
                              collect_rows=args.plot)

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

    # Wizualizacja: budujemy DataFrame ze zebranych probek i rysujemy ten sam
    # wykres co tryb wsadowy (plot_case), jeden PNG na lot.
    if args.plot:
        from notebooks.visualize import plot_case
        print("=" * 70)
        plot_df = consumer.to_dataframe()
        plots_dir = PROJECT_ROOT / "data" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        for case in plot_df["case_id"].drop_duplicates():
            png_path = plots_dir / f"stream_case_{case}_plot.png"
            plot_case(plot_df, case, str(png_path))


if __name__ == "__main__":
    main()
