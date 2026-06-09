# Detekcja anomalii w telemetrii drona

Projekt semestralny: trojwarstwowy detektor anomalii w telemetrii drona,
trenowany na **Drone Telemetry Tampering Dataset v2** (Kaggle).

## Architektura - 3 warstwy detekcji

| Warstwa | Plik | Idea |
|---|---|---|
| 1 - progi fizyczne | `detection/rules.py` | Twarde zakresy: altitude, speed, heading, lat/lon |
| 2 - statystyka | `detection/statistical.py` | Rolling z-score pochodnych + detekcja "freeze" (niska wariancja) |
| 3 - ML | `detection/ml.py` | 19 cech okienkowych (altitude/speed/heading/gps_step/dt) + jeden z 5 algorytmow (IF / OCSVM / LOF / RF / XGBoost) |

## Wymagania

- Python 3.10 lub nowszy
- Pakiety: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `xgboost`

```bash
pip install -r requirements.txt
```

> **Wazne przypiecia wersji** (w `requirements.txt`): `scikit-learn==1.6.1` (zgodne
> z zapisanymi modelami `.pkl`) oraz `pandas<3` (kod warstw 2/3 nie dziala na
> pandas 3.0). Na systemach z PEP668 / bez systemowego pip:
> `python3 -m pip install --user --break-system-packages -r requirements.txt`.

## Pobranie datasetu

1. Wejdz na [Kaggle - drone-telemetry-tampering-dataset-v2](https://www.kaggle.com/datasets/rasikaekanayakadevlk/drone-telemetry-tampering-dataset-v2)
2. Pobierz CSV i zapisz jako `data/drone_telemetry_v2.csv`

Alternatywnie przez Kaggle API:

```bash
kaggle datasets download -d rasikaekanayakadevlk/drone-telemetry-tampering-dataset-v2 -p data --unzip
```

## Uruchomienie - cala pipeline

```bash
python run_all.py                  # Isolation Forest (domyslnie)
python run_all.py one_class_svm    # albo inny algorytm
python run_all.py random_forest    # supervised
python run_all.py xgboost          # supervised (wymaga xgboost)
```

Wynik:
- `data/confusion_matrices.png` - macierze pomylek per warstwa
- `data/roc_curves.png` - ROC dla warstwy ML + punkty operacyjne W1/W2
- `data/case_<id>_plot.png` - telemetria przykladowego lotu
- `data/metrics_global.csv`, `metrics_per_scenario.csv`, `metrics_per_profile.csv`
- `models/<algorithm>.pkl` - wytrenowany model

### Uruchamianie pojedynczych modulow

```bash
python data/loader.py            # podsumowanie datasetu
python detection/rules.py        # tylko warstwa 1
python detection/statistical.py  # tylko warstwa 2
python detection/ml.py           # tylko warstwa 3 (IF)
python notebooks/visualize.py    # wykres jednego case'a
```

## Detekcja w czasie rzeczywistym (streaming)

Pakiet `streaming/` symuluje strumien telemetrii i prowadzi detekcje **online** -
probka po probce, tak jak wygladaloby to przy zywym dronie. Dwa moduly spiete
kolejka w pamieci (`queue.Queue`) + watki:

| Modul | Rola |
|---|---|
| `streaming/generator.py` | `TelemetryGenerator` - odtwarza dataset probka po probce na kolejke, z zachowaniem tempa z `t_rel` (skroconego o `--speed`) |
| `streaming/consumer.py` | `StreamConsumer` - bufor krok-po-kroku per `case_id`, na kazdej probce odpala **wszystkie 3 warstwy** i emituje alert gdy ktorakolwiek zaalarmuje |
| `streaming/alerts.py` | `Alert` + sinki: `ConsoleAlertSink`, `JsonlAlertSink`, `MultiSink` |
| `streaming/run_stream.py` | punkt wejscia / demo - spina generator (watek) z konsumentem |

**Jak to dziala (okienkowosc):** warstwy 2 i 3 sa okienkowe (z-score liczy okno 30
probek, ML - 8), wiec konsument utrzymuje **bufor kroczacy `deque` per `case_id`**.
Po kazdej nowej probce buduje mini-DataFrame z bufora, uruchamia na nim istniejace
funkcje detekcji (`detect_threshold_violations`, `detect_sudden_changes`,
`compute_features` + `AnomalyDetector.predict`) i bierze wynik **ostatniego wiersza**.

```bash
# demo: 2 loty, limit 800 probek, czas x200, alerty tez do pliku JSONL
python -m streaming.run_stream random_forest --cases 2 --max-samples 800 --speed 200 --jsonl

# tylko warstwa ML (model nie jest ladowany dla pozostalych trybow)
python -m streaming.run_stream random_forest --layers ml

# tylko tanie warstwy bezmodelowe (rules + statistical)
python -m streaming.run_stream --layers rules,statistical

# pelny strumien (bez limitu probek)
python -m streaming.run_stream random_forest --max-samples 0
```

Parametry CLI: `--cases N` (ile lotow), `--max-samples N` (limit probek, `0` = bez
limitu), `--speed F` (przyspieszenie czasu, `1.0` = realny), `--replicate R`
(domyslnie 3 = zbior testowy), `--layers` (`all` lub lista po przecinku z
`{rules,statistical,ml}` - przez ktore warstwy przepuscic strumien),
`--ml-threshold T` (prog decyzyjny warstwy ML), `--jsonl`
(zapis alertow do `data/alerts.jsonl`).

> **Prog ML (`--ml-threshold`):** steruje czuloscia warstwy 3. Dla modeli supervised
> (RF/XGBoost) `ml_score` to p(anomalia), a alert pada gdy `score >= prog`
> (domyslnie 0.5). Podniesienie progu = mniej alertow, wyzsza precyzja, nizszy recall.
> Przyklad (RF, ten sam wycinek 300 probek): `0.5` -> P=0.50 R=0.83; `0.7` -> P=0.77
> R=0.52; `0.9` -> P=1.00 R=0.19. Dla modeli unsupervised (IF/OCSVM/LOF) semantyka
> jest odwrotna (nizszy score = bardziej anomalna), wiec alert pada gdy `score < prog`.

> **Wybor warstw (`--layers`):** mozna przepuscic strumien tylko przez czesc
> pipeline'u, np. `--layers ml` (sam model) albo `--layers rules,statistical`
> (lekko, bez ladowania modelu). Bufor kroczacy dobiera sie automatycznie do
> najwiekszego okna wsrod aktywnych warstw (ml-only -> 18 probek, ze statistical -> 40),
> wiec wylaczenie warstwy statystycznej dodatkowo przyspiesza przetwarzanie.

Wymaga wczesniej zapisanego modelu (`models/<algorithm>.pkl`) - jesli go brak,
najpierw `python run_all.py <algorithm>`. Na koncu drukowane jest podsumowanie
(precision/recall/F1 wzgledem `label`), bo odtwarzamy etykietowany dataset.

> **Wydajnosc:** Random Forest to ~0.1 s/probke (200 drzew) - stad domyslny limit
> `--max-samples`. Dla szybszego real-time uzyj lzejszego modelu (`isolation_forest`,
> 1 MB). Metryki z malego `--max-samples` **nie sa reprezentatywne** (waski, gesty
> w anomaliach wycinek) - po porownywalne z offline liczby uruchom bez limitu.

## Trening na Google Colab

Notebook `notebooks/train_colab.ipynb` trenuje wszystkie 5 algorytmow i zapisuje
pickle'e do `models/`. Po pobraniu na lokalna maszyne mozna ich uzyc przez
`AnomalyDetector.load(path)`.

> **Uwaga o rozmiarze RF:** Random Forest jest ograniczony (`max_depth=16`,
> `min_samples_leaf=20`) - bez tego na milionach probek pickle puchnie do ~20 GB.
> Przy zmianie cech trzeba przetrenowac modele od nowa (stare `.pkl` maja zapisany
> stary zestaw `feature_names`).

## Wyniki (replicate 3 = test, profile mieszane)

Porownanie warstwy ML wg AUC (`data/roc_curves_all_algos.png`):

| Algorytm | AUC | rozmiar pkl |
|---|---|---|
| **random_forest** | **0.751** | 213 MB |
| xgboost | 0.743 | 0.9 MB |
| isolation_forest | 0.664 | 1 MB |
| lof | 0.592 | 17 MB |
| one_class_svm | 0.471 (≈ losowy) | 0.4 MB |

**Rekomendacja: Random Forest** - najlepszy rozdzial i przy progu 0.5 najwyzszy
recall (~70% per profile). XGBoost ma zblizone AUC, ale przy progu 0.5 odpala
zachowawczo (recall ~30%) - jego prawdopodobienstwa sa ostrozniejsze.

### Wplyw cech czasowych `dt_*`

Dodanie 4 cech z odstepow czasowych (`dt_mean/std/max/min`) bylo kluczowe dla
manipulacji zaburzajacych regularnosc probkowania. Recall RF dla warstwy 3,
przed dodaniem cech -> po:

| tamper_type | przed | po |
|---|---|---|
| `timestamp_drift` | 0% | **84%** |
| `injection` | 0% | **66%** |
| `speed_inconsistency` | 0% | 32% |
| `deletion_gap` | 0% | ~1% (wciaz slabo - patrz nizej) |

`deletion_gap` pozostaje slaby, ale to **artefakt etykietowania, nie brak cechy**.
Przy delecji `dt` faktycznie skacze (0.101 s -> ~2.2 s), wiec `dt_max` widzi sygnal -
problem w tym, ze delecja to zdarzenie **jednowierszowe**, a dataset oznacza jako
anomalie **szeroki pas** wierszy wokol niej. Wiekszosc tych oznaczonych probek jest
fizycznie identyczna z `normal`, wiec zaden uczciwy feature ich nie odroznii.
Kolumna `original_row_idx` rozwiazalaby to tylko przez **leakage** (to metadana
konstrukcji datasetu, nieobecna w realnej telemetrii) - swiadomie jej nie uzywamy.

## Struktura datasetu

| Kolumna | Opis |
|---|---|
| `case_id` | 720 unikalnych lotow |
| `replicate` | 0..3 - 4 niezalezne realizacje kazdego case'a (split: 0,1,2=train; 3=test) |
| `profile` | `subtle` / `balanced` / `strong` - intensywnosc anomalii |
| `row_idx` | krok w locie (row_idx=0 odrzucamy - timestamp 1970) |
| `label` | 0 / 1 - binarna etykieta anomalii |
| `tamper_type` | `normal` + 9 typow manipulacji (patrz nizej) |
| `timestamp` | ISO 8601 z offsetem UTC (np. `2024-09-15T20:05:41.726000+00:00`); parsowany z `format="ISO8601"` |
| `latitude`, `longitude` | GPS [deg] |
| `altitude` | wysokosc [m] |
| `speed` | predkosc [m/s] |
| `heading` | kurs [deg, 0..360) |
| `source` | DJI_FLIGHTRECORD_BIN_DJI_LOG (real) lub SYNTH_* (syntetyczne) |

### Klasy anomalii (tamper_type)

| Klasa | Charakterystyka | Spodziewana wykrywalnosc |
|---|---|---|
| `altitude_spike` | nagly skok wysokosci | W1 (jesli przekroczy prog) + W2 |
| `coordinate_jump` | teleport GPS | W2 (sudden_gps_step) + W3 |
| `heading_inconsistency` | niespojny kurs | W2 (sudden_heading) + W3 |
| `speed_inconsistency` | predkosc sensora != predkosc z GPS | W3 (cecha `speed_vs_gps`) |
| `timestamp_drift` | skok timestamp | W3 (cecha `dt_std`) |
| `injection` | dolozone falszywe wiersze | W2 / W3 (cecha `dt_min` ~ 0) |
| `deletion_gap` | luka w probkach | W3 (cecha `dt_max`) |
| `precision_rounding` | obciecie precyzji | W3 (subtelna anomalia kontekstowa) |
| `combined` | wiele typow razem | wszystkie warstwy |

## Format wynikow ewaluacji

`compute_full_evaluation` zwraca 3 tabele:

1. **Globalne**: precision/recall/F1 per warstwa
2. **Per `tamper_type`**: recall per warstwa per typ anomalii (jaki % danego typu zlapano)
3. **Per `profile`**: recall per warstwa per intensywnosc (subtle/balanced/strong)
