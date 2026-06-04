# Detekcja anomalii w telemetrii drona

Projekt semestralny: wykrywanie anomalii parametrów lotu (wysokość, prędkość, kurs, bateria)
oraz nagłych zmian wskazujących na awarię lub incydent cybernetyczny.

## Wymagania

- Python 3.9 lub nowszy
- Pakiety: `numpy`, `pandas`, `matplotlib`, `scikit-learn`

## Instalacja

1. Otwórz PowerShell w katalogu projektu (tam, gdzie jest ten README).
2. Zainstaluj pakiety:

   ```powershell
   python -m pip install numpy pandas matplotlib scikit-learn
   ```

## Uruchomienie

**Najprostsza opcja — odpal wszystko jednym poleceniem:**

```powershell
python run_all.py
```

To wykona pełny pipeline:
1. Generuje normalny lot (`data\normal_flight.csv`)
2. Wstrzykuje 5 typów anomalii (`data\flight_with_anomalies.csv`)
3. Tworzy wykres z detekcją (`data\telemetry_plot.png`)

**Krok po kroku (jeśli chcesz uruchamiać moduły osobno):**

```powershell
python data\generate.py            # tylko generator normalnego lotu
python data\inject_anomalies.py    # generator + wstrzyknięcie anomalii
python detection\rules.py          # test warstwy 1 (progi)
python detection\statistical.py    # test warstwy 2 (nagłe zmiany)
python notebooks\visualize.py      # wykres
```

> **Uwaga:** moduły z `detection\` i `notebooks\` zakładają, że plik
> `data\flight_with_anomalies.csv` już istnieje. Najpierw uruchom
> `python data\inject_anomalies.py` albo `python run_all.py`.

## Struktura

```
drone_anomaly\
├── run_all.py                  # uruchamia cały pipeline
├── README.md
├── data\
│   ├── generate.py             # generator normalnego lotu
│   ├── inject_anomalies.py     # wstrzykiwanie 5 scenariuszy anomalii
│   ├── normal_flight.csv       # (generowany)
│   ├── flight_with_anomalies.csv  # (generowany)
│   └── telemetry_plot.png      # (generowany)
├── detection\
│   ├── rules.py                # warstwa 1: progi fizyczne
│   └── statistical.py          # warstwa 2: rolling z-score
└── notebooks\
    └── visualize.py            # wykres telemetrii + alerty
```

## Scenariusze anomalii

| Scenariusz | Opis | Co powinno wykryć |
|---|---|---|
| `engine_failure` | Gwałtowny spadek wysokości i prędkości | Warstwa 2 (sudden_altitude, sudden_speed) |
| `gps_spoofing` | Skokowa zmiana kursu o ~180° | Warstwa 2 (sudden_heading) |
| `battery_drain` | Nienaturalnie szybki spadek baterii | Warstwa 2 (sudden_battery), potem warstwa 1 |
| `control_freeze` | Zacięcie sterów — parametry zamarzają | Warstwa 3 (do dodania) |
| `sensor_jamming` | Silny szum we wszystkich sensorach | Warstwa 2 (wiele kanałów) |

## Status projektu

- [x] Generator normalnego lotu
- [x] Wstrzykiwanie 5 scenariuszy anomalii
- [x] Warstwa 1 — progi fizyczne
- [x] Warstwa 2 — rolling z-score (wykrywanie nagłych zmian)
- [x] Wizualizacja
- [ ] Warstwa 3 — Isolation Forest (scikit-learn)
- [ ] Moduł ewaluacji: precision / recall / F1 per warstwa per scenariusz
- [ ] Raport końcowy (Jupyter notebook)
