# data/

Scripts for automated dataset generation. These scripts coordinate the SITL simulation stack to produce the normal-flight training dataset.

## Scripts

| Script | Purpose | Run |
|---|---|---|
| `generate_missions.py` | Generates waypoint files, meta-JSONs, and `missions_manifest.csv` via full factorial design (10752 missions) | Once |
| `orchestrator.py` | Reads manifest, uploads waypoints to SITL, sets drone params, runs missions, writes status back | Continuous |
| `telemetry_logger.py` | Captures MAVLink telemetry (RAW_IMU, GPS_RAW_INT, SCALED_PRESSURE, SERVO_OUTPUT_RAW, GLOBAL_POSITION_INT) to per-mission CSVs | Called by orchestrator |
| `validate_missions.py` | Post-hoc validation: compares recorded telemetry against planned waypoints, checks takeoff/landing/duration/gaps | After flights |

## Workflow

```bash
# 1. Start simulation stack
./start_sim.sh

# 2. Generate missions (once)
python3 data/generate_missions.py --output-dir ./dataset

# 3. Run orchestrator (resumable — Ctrl+C to stop, restart to continue)
python3 data/orchestrator.py --manifest ./dataset/missions_manifest.csv --output ./dataset

# watch simulation progress with
docker logs -f orchestrator

# 4. Validate completed missions
python3 data/validate_missions.py --dataset ./dataset
```


## Output Structure

All outputs go to `dataset/` (gitignored except waypoints and meta):

```
dataset/
├── missions_manifest.csv
├── waypoints/*.waypoints
├── meta/*_params.json
├── pcap/*.pcap
├── telemetry/*_<MSG_TYPE>.csv
├── phase_labels/*_phases.csv
└── validation_report.csv
```