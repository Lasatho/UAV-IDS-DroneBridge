#!/bin/bash
# Read-only daily wellness check of the attack run (and of the tuning on TR).
# usage: run_check_attack.sh <host hp|tr> <repo> <python> "<since local YYYY-MM-DD HH:MM>" <since UTC ISO> [out dir]
# Output: <out dir>/run_<host>.log plus JSON details (default /tmp/wellness_<date>_atk/out).
# Procedure and interpretation: Vault note "täglicher wellness-check".
H=$1; R=$2; PY=$3; SINCE=$4; SINCE_UTC=$5
S=$(cd "$(dirname "$0")" && pwd); O=${6:-/tmp/wellness_$(date +%Y%m%d)_atk/out}; mkdir -p "$O"; cd "$R"
M=dataset/missions_manifest_attacks_v2_$H.csv
OFF=$(ls dataset/clock_offset_*_20261025.json)
{
echo "### health"
docker ps --format '{{.Names}} {{.Status}}'
for c in $(docker ps --format '{{.Names}}'); do echo "$c restarts $(docker inspect -f '{{.RestartCount}} {{.State.StartedAt}}' $c)"; done
p=$(pgrep -f "^/bin/bash simulation/sim_watchdog.sh"); echo "watchdog pid $p $(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep ^MANIFEST)"
pgrep -af "wmediumd -c" | grep -v pgrep || echo "no wmediumd"
for i in wlan0 wlan1 wlan2; do echo "$i $(iw dev $i info | grep -o 'type [a-z]*')"; done
df -h / | tail -1
docker exec uav_ids_orchestrator md5sum /opt/orchestrator/orchestrator.py
$PY -I -c "import json,time;d=json.load(open('$OFF'));print('clock offset drift s', round(((time.time_ns()-time.monotonic_ns())-d['epoch_minus_monotonic_ns'])/1e9,3))"
echo "### rtf_low (<0.75 since)"; $PY -I $S/rtf_low.py ~/sim_degradation.log "$SINCE" 0.75
echo "### watchdog log since"; $PY -I -c "
import sys
for l in open('$HOME/sim_watchdog.log', errors='replace'):
    if l[:1].isdigit() and l[:16] >= '$SINCE': print(l.rstrip())" | tail -60
echo "### progress"; $PY -I $S/attack_progress.py $R $M
echo "### orchestrator log since"
docker logs --since $SINCE_UTC uav_ids_orchestrator > $O/orch_$H.log 2>&1
echo "starts $(grep -c '=== Starting mission' $O/orch_$H.log), completed $(grep -c '=== mission_.* completed' $O/orch_$H.log), failed $(grep -c '=== mission_.* failed' $O/orch_$H.log)"
grep -E "ERROR|WARNING|=== mission_.* failed|Paused|attempt|Read-back|Crash|Timeout\." $O/orch_$H.log | grep -v "Progress" | cut -c1-200
echo "ground phase min/max: $(grep -o 'Ground phase sim/wall [0-9.]*' $O/orch_$H.log | awk '{print $4}' | sort -n | sed -n '1p;$p' | tr '\n' ' ')"
echo "start offset max: $(grep -o 'Start offset from spawn point [0-9.]*' $O/orch_$H.log | awk '{print $6}' | sort -n | tail -1)"
echo "### degraded/crashed new"; find dataset/_degraded_runs dataset/_crashed_runs -maxdepth 1 -mindepth 1 -newermt "$SINCE" | sort
echo "### attack check"; $PY -I $S/attack_check.py $R $M $OFF $O/${H}_attack.json "$SINCE"
echo "### in-flight RTF (10-s windows > 3 m, sim/wall < 0.9)"; $PY -I $S/flight_rtf.py $R $M "$SINCE" 2>&1 | tail -30
echo "### data check (sensor values, all completed attack missions; attack effects expected)"
$PY -I $S/data_check.py $R $M $O/${H}_data.json > $O/${H}_data.txt 2>&1; head -40 $O/${H}_data.txt
if [ "$H" = tr ]; then
  echo "### tuning"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
  pgrep -af "tune.py" | grep -v pgrep | grep -c "worker-id" | sed 's/^/tune processes (incl. loader workers): /'
  (cd training && $PY -I $S/tune_status.py optuna_studies rssm_v2 mts_jepa_v2)
  grep -hiE "Traceback|Error|nan|inf |diverg|failed" training/tune_v2_*.log | grep -v UserWarning | tail -20
  ls -la --time-style=+%F_%T training/tune_v2_*.log | awk '{print $6, $7}'
fi
echo "### DONE"
} > $O/run_$H.log 2>&1
