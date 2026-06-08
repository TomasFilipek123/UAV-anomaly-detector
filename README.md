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
- Pakiety: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `xgboost` (opcjonalny)

```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```

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

`deletion_gap` pozostaje nieuchwytny dla `dt_*` - usuniecie probek nie zostawia
skoku `dt` (timestampy sa przenumerowane). Kandydat na dalsza prace: cecha
`original_row_idx.diff()` (przy deletion_gap `original_row_idx` ma dziury,
a `row_idx` jest ciagle).

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
