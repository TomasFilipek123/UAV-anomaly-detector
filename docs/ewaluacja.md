# Ewaluacja i ograniczenia

**Metodyka.** Skuteczność detektora oceniono na warstwie trzeciej (model Random
Forest), wybranej jako najlepsza spośród pięciu testowanych algorytmów (AUC 0.751
wobec 0.743 dla XGBoost i 0.471 dla One-Class SVM, który okazał się gorszy od
klasyfikatora losowego). Zbiór testowy stanowił replikat nr 3 (≈2,1 mln próbek),
nieużywany w treningu. Oprócz globalnych metryk (precision, recall, F1) raportowano
*recall* w rozbiciu na dziewięć typów manipulacji oraz na trzy profile intensywności
anomalii, a także krzywą Precision-Recall i miarę Average Precision (AP) — bardziej
wiarygodną od AUC przy 28,5% udziale klasy anomalnej.

**Wyniki.** Random Forest osiągnął na zbiorze testowym F1 = 0,558 (precision 0,461,
recall 0,706) przy progu decyzyjnym 0,5, oraz AP = 0,620 wobec poziomu odniesienia
0,294 (częstość klasy anomalnej) — tj. ponad dwukrotnie powyżej klasyfikatora
losowego. Kluczowy okazał się dobór cech: dodanie czterech cech opartych na
odstępach czasowych między próbkami (`dt_mean/std/max/min`) podniosło *recall* dla
manipulacji zaburzających regularność próbkowania z 0% do wartości użytecznych —
`timestamp_drift` z 0% do 84%, `injection` z 0% do 66%. Model radzi sobie bardzo
dobrze z anomaliami o wyraźnej sygnaturze fizycznej (`altitude_spike` 88%,
`heading_inconsistency` 87%, `coordinate_jump` 84%, `combined` 81%), słabiej
z subtelnymi (`speed_inconsistency` 31%).

**Weryfikacja generalizacji.** Ponieważ standardowy podział po `replicate` umieszcza
te same trajektorie lotów (`case_id`) zarówno w zbiorze treningowym, jak i testowym,
przeprowadzono dodatkową ewaluację na podziale **rozłącznym po `case_id`** (540 lotów
treningowych, 180 testowych — żaden lot nie występuje w obu zbiorach). Wyniki okazały
się niemal identyczne (F1 = 0,552, precision 0,455, recall 0,702), co dowodzi, że
model nie zapamiętuje sygnatur konkretnych lotów, lecz uczy się lokalnych, względnych
zależności — a raportowane metryki nie są obciążone przeciekiem danych.

**Ograniczenia.** Głównym ograniczeniem jest precyzja (0,46): około połowy
zgłaszanych alertów to fałszywe alarmy, co przy obecnym progu wyklucza zastosowanie
w trybie nienadzorowanym i wymagałoby fuzji warstw lub kalibracji progu pod docelowy
koszt FP/FN. Strojenie samego progu nie poprawia jakości — próg maksymalizujący F1
(0,393) daje tę samą wartość F1 = 0,558, co wskazuje na sufit wynikający z cech, nie
z punktu pracy. Manipulacja `deletion_gap` pozostaje praktycznie niewykrywalna
(recall ~1%), co zdiagnozowano jako **artefakt etykietowania**, a nie brak cechy:
usunięcie próbek jest zdarzeniem jednowierszowym (skok `dt` z 0,101 s do ~2,2 s,
widoczny dla modelu), natomiast zbiór danych oznacza jako anomalię szerokie pasmo
otaczających próbek, fizycznie nieodróżnialnych od danych normalnych. Świadomie
zrezygnowano z kolumny `original_row_idx`, która rozwiązałaby ten problem, lecz
wyłącznie przez przeciek danych (*leakage*) — jest to metadana konstrukcji zbioru,
nieobecna w rzeczywistej telemetrii.

## Podsumowanie metryk (Random Forest, warstwa 3)

| Metryka | replicate-split | case-split (nowe loty) |
|---|---|---|
| Precision | 0,461 | 0,455 |
| Recall | 0,706 | 0,702 |
| F1 | 0,558 | 0,552 |
| AP | 0,620 | — |
| AUC | 0,751 | — |

Baseline (częstość klasy anomalnej) = 0,294.

| tamper_type | recall (RF) |
|---|---|
| altitude_spike | 88% |
| heading_inconsistency | 87% |
| coordinate_jump | 84% |
| timestamp_drift | 82–84% |
| combined | 81% |
| injection | 66% |
| precision_rounding | 58% |
| speed_inconsistency | 31% |
| deletion_gap | ~1% (artefakt etykietowania) |
