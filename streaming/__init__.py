"""
Modul strumieniowy: symulacja telemetrii w czasie rzeczywistym + detekcja online.

  - generator.py : TelemetryGenerator   - odtwarza dataset probka po probce na queue
  - consumer.py  : StreamConsumer        - czyta queue, 3 warstwy detekcji, alerty
  - alerts.py    : Alert + AlertSink     - reprezentacja i emisja alertow
  - run_stream.py: punkt wejscia / demo  - spina generator i consumer w watkach
"""
