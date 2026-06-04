# Detekcja anomalii w telemetrii drona

Projekt semestralny: trojwarstwowy detektor anomalii w telemetrii drona,
trenowany na **Drone Telemetry Tampering Dataset v2** (Kaggle).

## Architektura - 3 warstwy detekcji

| Warstwa | Plik | Idea |
|---|---|---|
| 1 - progi fizyczne | `detection/rules.py` | Twarde zakresy: altitude, speed, heading, lat/lon |
| 2 - statystyka | `detection/statistical.py` | Rolling z-score pochodnych + detekcja "freeze" (niska wariancja) |
| 3 - ML | `detection/ml.py` | Cechy okienkowe + jeden z 5 algorytmow (IF / OCSVM / LOF / RF / XGBoost) |

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

Notebook `notebooks/train_colab.ipynb` (do dodania) trenuje wszystkie 5 algorytmow
i zapisuje pickle'e do `models/`. Po pobraniu na lokalna maszyne mozna ich uzyc
przez `AnomalyDetector.load(path)`.

## Struktura datasetu

| Kolumna | Opis |
|---|---|
| `case_id` | 720 unikalnych lotow |
| `replicate` | 0..3 - 4 niezalezne realizacje kazdego case'a (split: 0,1,2=train; 3=test) |
| `profile` | `subtle` / `balanced` / `strong` - intensywnosc anomalii |
| `row_idx` | krok w locie (row_idx=0 odrzucamy - timestamp 1970) |
| `label` | 0 / 1 - binarna etykieta anomalii |
| `tamper_type` | `normal` + 9 typow manipulacji (patrz nizej) |
| `timestamp` | epoch (sekundy) |
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
| `timestamp_drift` | skok timestamp | W3 (przez `speed_vs_gps`) |
| `injection` | dolozone falszywe wiersze | W2 / W3 |
| `deletion_gap` | luka w probkach | W3 (cechy okienkowe) |
| `precision_rounding` | obciecie precyzji | W3 (subtelna anomalia kontekstowa) |
| `combined` | wiele typow razem | wszystkie warstwy |

## Format wynikow ewaluacji

`compute_full_evaluation` zwraca 3 tabele:

1. **Globalne**: precision/recall/F1 per warstwa
2. **Per `tamper_type`**: recall per warstwa per typ anomalii (jaki % danego typu zlapano)
3. **Per `profile`**: recall per warstwa per intensywnosc (subtle/balanced/strong)
