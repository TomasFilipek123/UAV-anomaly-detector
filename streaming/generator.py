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

import numpy as np
import pandas as pd

from data.loader import load_dataset, split_by_replicate


# Sentinel wrzucany na koniec strumienia - sygnal dla konsumenta, ze danych juz nie bedzie.
END_OF_STREAM = None


# --------------------------------------------------------------------------- #
# Syntetyczny generator gladkich lotow (tryb demonstracyjny)
# --------------------------------------------------------------------------- #
# Loty z datasetu sa dosc niestabilne - warstwa 2 (rolling z-score) strzela na
# samym szumie, przez co wykresy demo sa nieczytelne. Ten generator buduje
# SZTUCZNE, gladkie loty (niski szum) z 0-2 wyraznie odseparowanymi anomaliami,
# zeby wykresy byly przejrzyste. Zwraca DataFrame w DOKLADNIE tym samym ukladzie
# kolumn co loader, dzieki czemu odtwarzamy go istniejacym TelemetryGenerator i
# rysujemy istniejacym plot_case bez zadnych zmian.
#
# Kazdy lot dostaje LOSOWE (per case_id, ale powtarzalne dla danego seed)
# parametry ksztaltu - inna wysokosc/predkosc przelotowa, kurs i amplitudy/okresy
# falowania - dzieki czemu loty nie wygladaja identycznie. Dwa profile:
#   - "cruise"  : caly czas przelot z lagodnym falowaniem (najczystsze wykresy),
#   - "mission" : start -> przelot -> ladowanie (rampa wysokosci; realistyczniej,
#                 ale rampy moga same wzbudzic kilka alertow W2 na brzegach).

# Anomalie wstrzykiwane domyslnie (po jednej z kazdej "rodziny" - widoczne na
# panelu wysokosci i na mini-mapie GPS).
DEFAULT_SYNTH_ANOMALIES = ("altitude_spike", "coordinate_jump")

# Obslugiwane typy anomalii (nazwy zgodne z tamper_type z datasetu).
SYNTH_ANOMALY_TYPES = (
    "altitude_spike", "speed_inconsistency", "heading_inconsistency", "coordinate_jump",
)

# Profile ksztaltu lotu.
SYNTH_SHAPES = ("cruise", "mission")

# Wartosci bazowe (uzywane gdy vary=False).
_CRUISE_ALT = 80.0       # m - wysokosc przelotowa
_CRUISE_SPEED = 12.0     # m/s - predkosc przelotowa
_BASE_HEADING = 90.0     # deg - kurs bazowy (wschod)
_LAT0, _LON0 = 52.2297, 21.0122   # punkt startu (Warszawa)
_M_PER_DEG = 111_320.0   # przyblizenie metr/stopien szerokosci


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """Wygladzone S (ease in-out) na [0,1]; pochodna zerowa na krancach."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _inject_anomaly(atype: str, sl: slice, alt, speed, heading, lat, lon) -> None:
    """Modyfikuje in-place wycinek lotu, wstrzykujac wybrana anomalie."""
    if atype == "altitude_spike":
        alt[sl] += 60.0                       # skok wysokosci (W2: z_altitude)
    elif atype == "speed_inconsistency":
        speed[sl] += 12.0                     # skok predkosci (W2: z_speed)
    elif atype == "heading_inconsistency":
        heading[sl] = (heading[sl] + 120.0) % 360.0   # skok kursu (W2: z_heading)
    elif atype == "coordinate_jump":
        lat[sl] += 0.002                      # teleport ~220 m (W2: z_gps_step)
        lon[sl] += 0.002
    else:
        raise ValueError(
            f"Nieznany typ anomalii: {atype!r}. Dozwolone: {SYNTH_ANOMALY_TYPES}"
        )


def _random_centers(k: int, lo: float, hi: float, min_gap: float,
                    rng: np.random.Generator) -> list[float]:
    """k losowych pozycji (frakcje lotu) w [lo, hi] z minimalnym odstepem min_gap.
    Po nieudanych probach wraca do rownomiernego rozlozenia."""
    if k <= 0:
        return []
    if k == 1:
        return [float(rng.uniform(lo, hi))]
    for _ in range(200):
        c = np.sort(rng.uniform(lo, hi, k))
        if np.all(np.diff(c) >= min_gap):
            return c.tolist()
    return np.linspace(lo, hi, k).tolist()


def _one_synthetic_flight(
    case_id: int, n: int, dt: float, pool, n_anomalies, shape: str,
    vary: bool, noise: float, rng: np.random.Generator, start_time: pd.Timestamp,
) -> pd.DataFrame:
    """Buduje jeden gladki lot (DataFrame) z wstrzyknietymi anomaliami."""
    t = np.arange(n, dtype=float) * dt
    frac = t / t[-1] if (n > 1 and t[-1] > 0) else np.zeros(n)

    # Parametry ksztaltu: losowe per lot (vary) albo stale (powtarzalny baseline).
    if vary:
        cruise_alt = rng.uniform(50.0, 120.0)
        cruise_speed = rng.uniform(8.0, 16.0)
        base_heading = rng.uniform(0.0, 360.0)
        a_amp, a_per, a_ph = rng.uniform(3, 10), rng.uniform(80, 160), rng.uniform(0, 2 * np.pi)
        s_amp, s_per, s_ph = rng.uniform(0.2, 0.6), rng.uniform(40, 90), rng.uniform(0, 2 * np.pi)
        h_amp, h_per, h_ph = rng.uniform(5, 20), rng.uniform(60, 120), rng.uniform(0, 2 * np.pi)
    else:
        cruise_alt, cruise_speed, base_heading = _CRUISE_ALT, _CRUISE_SPEED, _BASE_HEADING
        a_amp, a_per, a_ph = 5.0, 120.0, 0.0
        s_amp, s_per, s_ph = 0.3, 60.0, 0.0
        h_amp, h_per, h_ph = 8.0, 90.0, 0.0

    # Lagodne, plynne sinusy maja maly/rownomierny diff -> warstwa 2 milczy poza
    # anomaliami, a falowanie predkosci utrzymuje ciagly ruch GPS (brak "freeze").
    alt = cruise_alt + a_amp * np.sin(2 * np.pi * t / a_per + a_ph)
    speed = cruise_speed + s_amp * np.sin(2 * np.pi * t / s_per + s_ph)
    heading = (base_heading + h_amp * np.sin(2 * np.pi * t / h_per + h_ph)) % 360.0

    # Profil "mission": rampa wysokosci start->przelot->ladowanie. Predkosc
    # (ruch poziomy) zostaje ciagla, by GPS sie nie zatrzymal (brak "freeze").
    if shape == "mission":
        env = np.minimum(_smoothstep(frac / 0.25), 1.0 - _smoothstep((frac - 0.75) / 0.25))
        alt = cruise_alt * env + a_amp * np.sin(2 * np.pi * t / a_per + a_ph)

    # Maly szum pomiarowy - na tyle niski, ze |z| < prog W2 (3.5).
    alt = np.clip(alt + rng.normal(0.0, 0.05 * noise, n), 0.0, None)
    speed = np.clip(speed + rng.normal(0.0, 0.02 * noise, n), 0.0, None)
    heading = (heading + rng.normal(0.0, 0.05 * noise, n)) % 360.0

    # GPS: calkujemy pozycje z predkosci i kursu (0 deg = polnoc).
    lat = np.empty(n); lon = np.empty(n)
    lat[0], lon[0] = _LAT0, _LON0
    for i in range(1, n):
        d = speed[i] * dt
        th = np.radians(heading[i])
        lat[i] = lat[i - 1] + (d * np.cos(th)) / _M_PER_DEG
        lon[i] = lon[i - 1] + (d * np.sin(th)) / (_M_PER_DEG * np.cos(np.radians(lat[i - 1])))

    # Ile anomalii w tym locie: "random" -> 0..2, albo stala liczba.
    if isinstance(n_anomalies, str) and n_anomalies == "random":
        k = int(rng.integers(0, 3))
    else:
        k = int(n_anomalies)
    k = min(k, len(pool)) if pool else 0

    # Profil "mission" trzyma anomalie w strefie przelotu (z dala od ramp),
    # "cruise" rozklada je szerzej.
    lo, hi = (0.35, 0.65) if shape == "mission" else (0.20, 0.80)

    label = np.zeros(n, dtype=int)
    tamper = np.array(["normal"] * n, dtype=object)
    if k > 0:
        types = list(rng.choice(np.array(pool, dtype=object), size=k, replace=False))
        seg_len = max(2, int(round(2.0 / dt)))   # segment ~2 s
        centers = _random_centers(k, lo, hi, min_gap=0.20, rng=rng)
        for atype, c in zip(types, centers):
            i0 = int(c * n)
            sl = slice(i0, min(i0 + seg_len, n))
            _inject_anomaly(atype, sl, alt, speed, heading, lat, lon)
            label[sl] = 1
            tamper[sl] = atype

    idx = np.arange(1, n + 1)
    return pd.DataFrame({
        "profile": "demo",
        "replicate": -1,                          # nie koliduje z replicate 0..3
        "case_id": case_id,
        "case_name": f"synthetic_{case_id}",
        "row_idx": idx,
        "label": label,
        "tamper_type": tamper,
        "timestamp": start_time + pd.to_timedelta(t, unit="s"),
        "latitude": lat,
        "longitude": lon,
        "altitude": alt,
        "speed": speed,
        "heading": heading,
        "source": "synthetic",
        "original_row_idx": idx,
        "t_rel": t,
    })


def make_synthetic_flights(
    n_cases: int = 3,
    n_samples: int = 200,
    dt: float = 1.0,
    anomalies: Iterable[str] = DEFAULT_SYNTH_ANOMALIES,
    n_anomalies="random",
    shape: str = "cruise",
    vary: bool = True,
    noise: float = 1.0,
    seed: int = 42,
    start_case_id: int = 900_000,
) -> pd.DataFrame:
    """
    Generuje `n_cases` gladkich lotow do demo - kazdy o `n_samples` probkach co
    `dt` sekund, z 0-2 wyraznie odseparowanymi anomaliami. Loty roznia sie miedzy
    soba ksztaltem (gdy vary=True) oraz liczba/pozycja anomalii.

    Zwracany DataFrame ma te same kolumny co loader (label/tamper_type/profile/
    t_rel/...), wiec mozna go podac wprost do TelemetryGenerator(df=...) i
    narysowac plot_case.

    Parameters
    ----------
    n_cases : ile lotow wygenerowac (kazdy dostaje case_id = start_case_id + i).
    n_samples : liczba probek na lot.
    dt : odstep miedzy probkami w sekundach.
    anomalies : PULA typow anomalii do losowania; puste/None = loty bez anomalii.
        Dozwolone: SYNTH_ANOMALY_TYPES.
    n_anomalies : ile anomalii na lot - "random" (0..2 losowo per lot) albo stala
        liczba calkowita (przyciecie do rozmiaru puli).
    shape : profil lotu - "cruise" (przelot z falowaniem, najczystsze wykresy)
        albo "mission" (start->przelot->ladowanie). Patrz SYNTH_SHAPES.
    vary : True = losowe parametry ksztaltu per lot (rozne wysokosci/kursy/fale),
        False = wspolny, powtarzalny baseline.
    noise : mnoznik szumu pomiarowego (1.0 = domyslny, 0 = idealnie gladko).
    seed : ziarno RNG (powtarzalnosc calej partii).
    start_case_id : bazowy case_id (duza liczba, by nie kolidowac z datasetem).
    """
    pool = [str(a).strip() for a in (anomalies or []) if str(a).strip()]
    unknown = set(pool) - set(SYNTH_ANOMALY_TYPES)
    if unknown:
        raise ValueError(
            f"Nieznane typy anomalii: {sorted(unknown)}. Dozwolone: {SYNTH_ANOMALY_TYPES}"
        )
    if shape not in SYNTH_SHAPES:
        raise ValueError(f"Nieznany shape: {shape!r}. Dozwolone: {SYNTH_SHAPES}")
    if not (isinstance(n_anomalies, str) and n_anomalies == "random"):
        n_anomalies = int(n_anomalies)

    rng = np.random.default_rng(seed)
    start_time = pd.Timestamp("2024-09-15T12:00:00Z")
    frames = [
        _one_synthetic_flight(
            case_id=start_case_id + i, n=n_samples, dt=dt,
            pool=pool, n_anomalies=n_anomalies, shape=shape, vary=vary,
            noise=noise, rng=rng, start_time=start_time,
        )
        for i in range(n_cases)
    ]
    return pd.concat(frames, ignore_index=True)


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
