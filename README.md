# UAV-IDS-DroneBridge

Forschungsprojekt für eine Masterarbeit über **Intrusion Detection System (IDS) für UAVs**, das auf dem [DroneBridge](https://github.com/DroneBridge/DroneBridge)-Funkprotokoll aufsetzt. Die Pipeline reicht von der simulierten Datengenerierung (ArduPilot SITL + Gazebo) über automatisierte Angriffsinjektion bis hin zum Training und Export von Anomalie-Erkennungsmodellen für den Einsatz auf einem NVIDIA Jetson Orin.

---

## Überblick

```
Datengenerierung                ML-Pipeline
─────────────────────────────   ──────────────────────────────────
generate_missions.py            training/
        │                         ├── train.py
        ▼                         ├── evaluate.py
missions_manifest.csv             ├── export.py (→ ONNX → TensorRT)
+ dataset/waypoints/              ├── config.py
+ dataset/meta/                   ├── configs/default.yaml
        │                         ├── models/
        ▼                         │    ├── rssm/      (unsupervised)
simulation/ (Docker-Stack)        │    ├── mts_jepa/  (unsupervised)
 Gazebo + ArduPilot SITL          │    └── hassler/   (supervised)
 + DroneBridge + MAVProxy         └── data/
        │                              ├── dataset.py
        ▼                              ├── preprocessing.py
orchestrator.py                        ├── features.py
 ├── tcpdump → dataset/pcap/           └── windowing.py
 ├── telemetry_logger.py
 │    └── → dataset/telemetry/
 ├── attacks/*.py (optional)
 └── → dataset/phase_labels/
```

---

## Verzeichnisstruktur

```
UAV-IDS-DroneBridge/
├── data/                        # Datengenerierungs-Skripte
│   ├── generate_missions.py     # Erstellt alle Missionen 
│   ├── orchestrator.py          # Führt Missionen automatisiert aus
│   ├── telemetry_logger.py      # Loggt MAVLink-Sensor-Daten als CSV
│   ├── pcap_to_csv.py           # Konvertiert PCAP-Captures zu CSV
│   ├── validate_missions.py     # Validiert erzeugte Missionsdateien (TODO: delete?)
│   ├── visualize_mission.py     # Visualisiert Waypoint-Profile    (TODO: delete?)
│   ├── CPS_correlation_explorer.py  # Analysiert CPS-Inkonsistenzen    (TODO: delete)
│   └── attacks/                 # Angriffsskripte (vom Orchestrator gestartet)
│       ├── gnss_spoofing.py     # T1 – GPS-Position driften lassen
│       ├── gnss_jamming.py      # T2 – GPS komplett deaktivieren
│       ├── rf_jamming.py        # T6/T19 – 802.11-Kanal fluten
│       ├── frame_injection.py   # T7 – Crafted DroneBridge-Frames einschleusen
│       └── c2_hijacking.py      # T10 – MAVLink-Kommandos via DroneBridge
│
├── dataset/                     # Generierte Daten (gitignored, außer meta/)
│   ├── missions_manifest.csv    # Index aller Missionen + Status
│   ├── meta/                    # mission_XXXXX_params.json pro Mission
│   ├── waypoints/               # Waypoint-Dateien
│   ├── pcap/                    # Netzwerkaufzeichnungen (tcpdump)
│   ├── telemetry/               # CSV-Dateien pro Nachrichtentyp & Mission
│   ├── phase_labels/            # Angriffs-Zeitstempel als JSON
│   └── cache/                   # Vorverarbeitete .npy-Arrays (auto-generiert, erst nach Abschluss Simulation)
│
├── simulation/                  # Docker-basierter Simulationsstack
│   ├── docker-compose.yml       # Definiert alle Services
│   ├── ardupilot/Dockerfile     # ArduPilot SITL (Copter)
│   ├── gazebo/Dockerfile        # Gazebo 8 Physik-Simulator
│   ├── mavproxy/Dockerfile      # MAVProxy für UDP-Multiplexing
│   ├── orchestrator/Dockerfile  # Container für orchestrator.py
│   ├── dronebridge.Dockerfile   # DroneBridge-Proxy (GCS + UAV)
│   ├── dronebridge/             # Git-Submodul: DroneBridge C-Quellcode
│   └── start_sim.sh             # Startskript für die Simulation
│
├── training/                    # ML-Trainings-Pipeline
│   ├── train.py                 # Einheitlicher Trainingseinstiegspunkt
│   ├── evaluate.py              # Auswertung (unlabeled + labeled)
│   ├── export.py                # ONNX-Export + Paritätsprüfung
│   ├── config.py                # YAML-Config mit CLI-Overrides + Interpolation
│   ├── configs/
│   │   └── default.yaml         # Basis-Konfiguration für alle Modelle
│   ├── models/
│   │   ├── base.py              # AnomalyDetectionModel (abstrakte Basisklasse)
│   │   ├── rssm/model.py        # Recurrent State Space Model (DreamerV3-Variant)
│   │   ├── mts_jepa/model.py    # Joint Embedding Prediction (JEPA) für Zeitreihendaten
│   │   └── hassler/model.py     # CNN-LSTM Baseline nach Hassler et al. 2024
│   └── data/
│       ├── dataset.py           # Dataset-Fabrik + TelemetryWindowDataset
│       ├── preprocessing.py     # Mission-Laden, Zeitraster-Alignment, Normalizer
│       ├── features.py          # Feature-Definitionen (zentrale Wahrheitsquelle)
│       └── windowing.py         # Sliding-Window-Index-Generierung
│
├── requirements.txt             # Python-Abhängigkeiten (Datengenerierung, TODO: NICHT UP TO DATE )
└── training/requirements.txt    # Python-Abhängigkeiten (Training)
```

---

## Komponenten im Detail

### 1. Simulationsstack (`simulation/`)

Der Docker-Compose-Stack bildet die gesamte UAV-Kommunikationskette nach:

```
Gazebo (Physik)
    │  UDP Plugin
    ▼
ArduPilot SITL (Port 5760 TCP)
    │
    ▼ (socat-Bridge)
DroneBridge UAV-Proxy (wlan0, Port 5751)
    │  "RF"-Link (hwsim0 — simuliertes WLAN)
DroneBridge GCS-Proxy (wlan1, Port 5750)
    │
    ▼
MAVProxy ──► UDP 14552  (Orchestrator/GCS)
         └─► UDP 14553  (Telemetry Logger)
```

**Services:**
| Service | Zweck |
|---|---|
| `gazebo` | 3D-Physiksimulation mit ArduPilot-Plugin |
| `ardupilot` | ArduCopter SITL, empfängt Gazebo-Physik |
| `dronebridge_uav` | DroneBridge-Proxy auf UAV-Seite (wlan0) |
| `dronebridge_gcs` | DroneBridge-Proxy auf GCS-Seite (wlan1) |
| `socat` | Verbindet DroneBridge-UAV-Port mit ArduPilot SITL (beide erwarten TCP-Server zu sein, socat ist serverbridge) |
| `mavproxy` | Verteilt MAVLink auf UDP 14552 + 14553 |
| `orchestrator` | Führt `orchestrator.py` im Container aus |

`hwsim0` ist ein simuliertes 802.11-Interface (mac80211_hwsim). Angriffe auf den "RF-Link" injizieren Frames direkt auf dieses Interface.

---

### 2. Datengenerierung (`data/`)

#### Schritt 1: Missionen generieren

```bash
python3 data/generate_missions.py --output-dir ./dataset
python3 data/generate_missions.py --output-dir ./dataset --dry-run  # nur zählen
```

`generate_missions.py` erzeugt einen **Full-Factorial-Designraum** über alle Parameterkombinationen (bis auf Wind):

| Dimension | Varianten |
|---|---|
| Flugprofile | Rectangle, Figure-8, Star, Zigzag, Loiter |
| Drohnen-Preset | conservative, standard, dynamic, fast |
| Windprofil | calm, light, moderate, gusty |   => Wird später ergänzt
| Windrichtung | 0°, 45°, 90°, 135° |

Jede Kombination ergibt eine Mission mit:
- `.waypoints`-Datei (QGC WPL 110 Format) in `dataset/waypoints/`
- `mission_XXXXX_params.json` in `dataset/meta/`
- Eintrag in `dataset/missions_manifest.csv` (Status: `pending` / `invalid`)

#### Schritt 2: Missionen ausführen (Orchestrator)

```bash
python3 data/orchestrator.py \
    --manifest ./dataset/missions_manifest.csv \
    --output   ./dataset \
    --connection udpin:127.0.0.1:14552
```

Der Orchestrator iteriert über alle `pending`-Missionen und führt pro Mission aus:

1. MAVLink-Verbindung herstellen (GPS-Fix + EKF abwarten)
2. Drohnen-Parameter setzen (`PARAM_SET`)
3. Mission hochladen (Waypoint-Upload-Protokoll)
4. **tcpdump** starten → `dataset/pcap/mission_XXXXX.pcap`
5. **telemetry_logger** starten → `dataset/telemetry/mission_XXXXX_*.csv`
6. Takeoff (GUIDED) → AUTO-Modus
7. Optional: **Angriffsskript** starten (randomisierter Offset + Dauer)
8. Auf Mission-Ende warten (STATUSTEXT / HEARTBEAT)
9. Label schreiben → `dataset/phase_labels/mission_XXXXX.json`
10. Manifest-Status aktualisieren, SITL rebooten

Bei 5 aufeinanderfolgenden Fehlern stoppt der Orchestrator und schreibt `.stop_orchestrator`. Push-Benachrichtigungen via [ntfy.sh](https://ntfy.sh) (opt-in via `NTFY_TOPIC`-Umgebungsvariable).

#### Schritt 3: Telemetrie-Logging

`telemetry_logger.py` läuft parallel zum Orchestrator (als Subprocess) und loggt folgende MAVLink-Nachrichtentypen mit 50 Hz auf separate CSVs:

| Nachrichtentyp | Felder |
|---|---|
| `RAW_IMU` | xacc, yacc, zacc, xgyro, ygyro, zgyro, xmag, ymag, zmag |
| `GPS_RAW_INT` | fix_type, lat, lon, alt, eph, epv, vel, cog, satellites_visible |
| `SCALED_PRESSURE` | press_abs, press_diff, temperature |
| `SERVO_OUTPUT_RAW` | servo1_raw … servo8_raw |
| `GLOBAL_POSITION_INT` | lat, lon, alt, relative_alt, vx, vy, vz, hdg |

Jede CSV-Zeile enthält zusätzlich `host_timestamp_ns` (monotoner Systemclock) als gemeinsamen Synchronisationsanker.

#### Phase-Labels (Angriffsmarkierung)

`dataset/phase_labels/mission_XXXXX.json`:
```json
{
  "run_id": "mission_00042",
  "start_time": 1719000000.0,
  "end_time": 1719001800.0,
  "attack_type": "gnss_spoofing",
  "attack_start": 1719000180.5,
  "attack_end": 1719000220.1,
  "mission_success": true,
  "waypoints_reached": [1, 2, 3, 4]
}
```

---

### 3. Angriffsskripte (`data/attacks/`)

Alle Skripte können standalone getestet oder vom Orchestrator automatisch gestartet/gestoppt werden. Sie sind auf `SIGTERM` ausgelegt (sauberes Aufräumen).

| Skript | PASTAD-Kategorie | Mechanismus |
|---|---|---|
| `gnss_spoofing.py` | T1 | Setzt `SIM_GPS_GLITCH_X/Y` per PARAM_SET; graduelles Driften (~2,2 m/Schritt) |
| `gnss_jamming.py` | T2 | Setzt `SIM_GPS_DISABLE=1`; EKF fällt rein auf IMU zurück |
| `rf_jamming.py` | T6/T19 | Flutet `hwsim0`-Interface mit zufälligen 802.11-Frames (max. Rate) |
| `frame_injection.py` | T7 | Injectiert crafted DroneBridge-Frames mit `SET_MODE_LAND` / `SET_POSITION` |
| `c2_hijacking.py` | T10 | Zyklische Kommando-Injektion: LAND → RTL → FLIGHT_TERMINATION |

Die RF-basierten Angriffe (rf_jamming, frame_injection, c2_hijacking) benötigen Root und Scapy (`sudo`). Die GNSS-Angriffe nutzen eine zweite MAVLink-Verbindung zum SITL (UDP 14554).

---

### 4. ML-Trainings-Pipeline (`training/`)

#### Konfigurationssystem

```bash
# Basis-Config
python train.py --config configs/default.yaml

# Mit Experiment-Override
python train.py --config configs/default.yaml --experiment configs/exp_rssm.yaml

# CLI-Overrides (höchste Priorität)
python train.py model.name=mts_jepa training.lr=3e-4 data.window_length=128
```

`config.py` unterstützt:
- Deep-Merge mehrerer YAML-Dateien
- `${a.b.c}`-Interpolation (z.B. `telemetry_dir: "${data.dataset_dir}/telemetry"`)
- Automatisches Parsen von Typen (int, float, bool, list) aus CLI-Strings

#### Daten-Pipeline

```
dataset/telemetry/
    │
    ├── discover_missions()          # Findet vollständige Missionen (alle 5 CSV-Typen)
    ├── split_missions()             # Mission-level Split (70/15/15, seed-deterministisch)
    ├── load_mission()               # Lädt 5 CSVs, aligned auf gemeinsames Zeitraster
    │    └── merge_asof (LOCF)       # Last-observation-carried-forward, Toleranz 2 Perioden
    ├── Normalizer.fit(train)        # Z-Score per Feature, nur auf Trainings-Split
    ├── .npy-Cache                   # dataset/cache/{freq}hz_{feature_hash}/
    └── TelemetryWindowDataset       # Sliding Windows (B, T, F) mit build_window_index()
```

**Wichtig:** Der Split erfolgt auf Mission-Ebene, nicht auf Fenster-Ebene, um temporale Datenlecks zu verhindern.

Default-Parameter (in `configs/default.yaml`):
- Resampling: 10 Hz
- Fensterlänge: 64 Timesteps (6,4 Sekunden)
- Stride: 16 Timesteps
- Normalisierung: Z-Score
- Features: alle 5 Nachrichtentypen → 37 Features

#### Modelle

Alle Modelle implementieren die abstrakte `AnomalyDetectionModel`-Schnittstelle aus `models/base.py`:

| Methode | Bedeutung |
|---|---|
| `training_step(batch)` | Gibt Dict mit `"loss"` zurück (differenzierbar) |
| `anomaly_score(batch)` | Gibt `(B,)` Float-Tensor zurück (höher = anomaler) |
| `configure_optimizers(cfg)` | Standard: AdamW + CosineAnnealing |
| `on_train_batch_end()` | Hook z.B. für EMA-Updates |
| `requires_labels` | `True` für supervised Modelle |

**RSSM** (`models/rssm/`) — Unsupervised, produktionsbereit:
DreamerV3-Stil World Model ohne Aktionen. Lernt die normale Drohnendynamik im Latent Space.
- Deterministischer GRU-Pfad `h_t` + stochastischer Latent-Zustand `z_t`
- KL-Balancing (stop-gradient) wie in DreamerV3
- Anomalie-Score = mittlerer One-Step-Prior-Prediction-Error

**MTS-JEPA** (`models/mts_jepa/`) — Unsupervised, produktionsbereit:
Joint Embedding Predictive Architecture für multivariate Zeitreihen.
- Fenster → temporale Patches → Transformer-Encoder
- Predictor sagt Embeddings maskierter Patches vorher (kein Rekonstruktionsfehler in Eingabe-Space)
- EMA-Target-Branch (kein Kollaps-Problem)
- Anomalie-Score = mittlerer Embedding-Prediction-Error auf gemaskten Patches

**HasslerBaseline** (`models/hassler/`) — Supervised, noch nicht vollständig:
CNN-LSTM nach Hassler, Mughal & Ismail (IEEE TITS 2024). Braucht Angriffsbezeichnungen pro Fenster.
Architektur ist aktuell ein Platzhalter — muss gegen das Paper verifiziert werden (TODO).

#### Training

```bash
cd training/
python train.py model.name=rssm
python train.py model.name=mts_jepa training.lr=1e-3
```

Checkpoints landen in `checkpoints/{timestamp}_{model}/`:
- `best.pt` — bestes Validierungs-Loss
- `last.pt` — letzter Epoch
- `epoch_XXXX.pt` — periodische Snapshots (alle 10 Epochs)
- `config.json` — vollständige Konfiguration für Reproduzierbarkeit

#### Evaluation

```bash
python evaluate.py --checkpoint checkpoints/<run>/best.pt
```

Zwei Modi (automatisch erkannt):

- **Unlabeled** (aktuell): Keine Angriffsdaten vorhanden. Berechnet Anomalie-Score-Verteilung auf Normal-Daten, kalibriert Schwellwerte (Perzentile: p90/p95/p99/p99.5/p99.9). Der p99-Schwellwert entspricht ~1% FPR auf Normaldaten.
- **Labeled** (nach Angriffsdaten-Integration): Vollständige Metriken: Precision, Recall, F1, AUC-ROC, AUC-PR, FPR, Detection Latency.

Ausgabe: `scores_test.npz` + `eval_report_test.json` neben dem Checkpoint.

**Hinweis:** Die Zeitbasis-Abbildung von `phase_labels` (Epoch-Sekunden) auf Telemetrie-Timestamps (monotone Nanosekunden) ist noch nicht implementiert (`NotImplementedError` in `evaluate.py:load_window_labels`).

#### Export (Jetson Orin)

```bash
python export.py --checkpoint checkpoints/<run>/best.pt [--opset 18]
python export.py --checkpoint checkpoints/<run>/best.pt --export-calibration 1000
```

Exportiert den **Anomalie-Score-Pfad** (nicht den Trainingsgraphen) als ONNX mit dynamischer Batch-Achse. Optionale Paritätsprüfung via onnxruntime.

TensorRT-Engine-Bau muss auf dem Orin selbst erfolgen (hardware-spezifisch):
```bash
# FP16
trtexec --onnx=model.onnx --saveEngine=model_fp16.engine --fp16
# INT8 (benötigt Kalibrierungsdaten aus --export-calibration)
trtexec --onnx=model.onnx --saveEngine=model_int8.engine --int8 --calib=<cache>
```

---

## Schnellstart

### Voraussetzungen

```bash
# Python-Umgebung (Datengenerierung)
pip install -r requirements.txt

# Python-Umgebung (Training)
pip install -r training/requirements.txt

# Submodul initialisieren (DroneBridge-Quellcode)
git submodule update --init --recursive
```

### Missionen generieren

```bash
python3 data/generate_missions.py --output-dir ./dataset
# Ergebnis: missions_manifest.csv + waypoints/ + meta/
```

### Simulation starten + Daten sammeln

```bash
cd simulation/
./start_sim.sh
# Orchestrator läuft im Container und iteriert alle pending-Missionen
```

### Modell trainieren

```bash
cd training/
python train.py model.name=rssm
python train.py model.name=mts_jepa
```

### Auswerten und exportieren

```bash
cd training/
python evaluate.py --checkpoint checkpoints/<run>/best.pt
python export.py   --checkpoint checkpoints/<run>/best.pt --export-calibration 1000
```

---

## Datenfluss

```
generate_missions.py
        │
        ├──► dataset/missions_manifest.csv   (pending/completed/invalid)
        ├──► dataset/waypoints/              (.waypoints Dateien)
        └──► dataset/meta/                   (_params.json pro Mission)
                │
                ▼
        orchestrator.py (läuft gegen SITL)
                │
                ├──► dataset/pcap/           (.pcap Netzwerkaufnahmen)
                ├──► dataset/telemetry/      (_MSG_TYPE.csv pro Mission)
                ├──► dataset/phase_labels/   (.json Angriffs-Zeitfenster)
                └──► missions_manifest.csv   (Status-Update: completed/failed)
                │
                ▼
        training/data/preprocessing.py
                │
                ├── discover_missions()      (prüft auf vollständige CSV-Sätze)
                ├── load_mission()           (5 CSVs → gemeinsames 10Hz-Raster)
                ├── Normalizer.fit()         (Z-Score, nur auf train-Split)
                └── dataset/cache/          (.npy-Arrays + normalizer.json)
                        │
                        ▼
                TelemetryWindowDataset
                (B=64, T=64, F=37)
                        │
                        ▼
                train.py / evaluate.py / export.py
```

---

## Abhängigkeiten

**Datengenerierung** (`requirements.txt`):
`pymavlink`, `scapy`, `numpy`, `scipy`, `pandas`, `matplotlib`, `filterpy`

**Training** (`training/requirements.txt`):
`torch`, `numpy`, `pandas`, `pyyaml`, `wandb` (optional)

**Simulation**:
Docker + Docker Compose, mac80211_hwsim Kernel-Modul (für simuliertes WLAN), X11 (für Gazebo-GUI)

---

## Aktueller Entwicklungsstand

- [x] Simulationsstack (Docker Compose)
- [x] Missionsgeneration (Full-Factorial, ~50.000+ Missionen)
- [x] Orchestrator (automatisierte Datengenerierung)
- [x] Angriffsskripte (5 PASTAD-Angriffskategorien)
- [x] Datenpipeline (Preprocessing, Windowing, Normalisierung)
- [x] RSSM-Modell (unsupervised, ONNX-exportierbar)
- [x] MTS-JEPA-Modell (unsupervised, ONNX-exportierbar)
- [x] Evaluation (unlabeled-Modus)
- [x] ONNX-Export
- [ ] Zeitbasis-Mapping phase_labels → telemetry (für labeled Evaluation)
- [ ] HasslerBaseline-Architektur verifizieren (TODO im Code)
- [ ] TensorRT-Pipeline auf Jetson Orin
- [ ] Integration echter Angriffsdaten in Trainingsloop
