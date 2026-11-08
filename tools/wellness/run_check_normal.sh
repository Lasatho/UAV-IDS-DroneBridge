#!/bin/bash
# Read-only daily wellness check of the normal run v3 (GCS + on-board telemetry) and of the tuning on TR.
# usage: run_check_normal.sh <host hp|tr> <repo> <python> "<since local YYYY-MM-DD HH:MM>" <since UTC ISO> [out dir]
# Output: <out dir>/run_<host>.log plus JSON details (default /tmp/wellness_<date>_v3/out).
# Every simulation instance of the host is checked: instance 0 = docker-compose.yml
# (manifest missions_manifest_v3_<host>.csv), instance k >= 1 from
# simulation/instances/<k>.env if its orchestrator container uav_ids_<k>_orchestrator
# exists (manifest missions_manifest_v3_<host><k>.csv).
# Procedure and interpretation: Vault note "täglicher wellness-check".
H=$1; R=$(cd "$2" && pwd); PY=$(cd "$(dirname "$3")" 2>/dev/null && pwd)/$(basename "$3"); SINCE=$4; SINCE_UTC=$5
[ -x "$PY" ] || PY=$(command -v "$3")
S=$(cd "$(dirname "$0")" && pwd); O=${6:-/tmp/wellness_$(date +%Y%m%d)_v3/out}; mkdir -p "$O"; cd "$R"
OFF=$(ls dataset/clock_offset_*_20261025.json)
INST="0"
for f in simulation/instances/*.env; do
  [ -e "$f" ] || continue; k=$(basename $f .env)
  docker inspect uav_ids_${k}_orchestrator > /dev/null 2>&1 && INST="$INST $k"
done
# per instance: container prefix, manifest, radios, logs
inst_vars(){
  if [ "$1" = 0 ]; then P=uav_ids; M=dataset/missions_manifest_v3_$H.csv; IFS_="wlan0 wlan1 wlan2"; CH=6
    SL=$HOME/sim_degradation.log; WL=$HOME/sim_watchdog.log; TAG=$H
  else . simulation/instances/$1.env; P=uav_ids_$1; M=dataset/missions_manifest_v3_$H$1.csv
    IFS_="$IF_DRONE $IF_GCS $IF_ATTACK"; SL=$HOME/sim_degradation_$1.log; WL=$HOME/sim_watchdog_$1.log; TAG=$H$1; fi
}
{
echo "### health (host)"
echo "instances: $INST"
docker ps --format '{{.Names}} {{.Status}}' | sort
for c in $(docker ps --format '{{.Names}}' | sort); do echo "$c restarts $(docker inspect -f '{{.RestartCount}} {{.State.StartedAt}}' $c)"; done
for p in $(pgrep -f "^/bin/bash simulation/sim_watchdog.sh"); do
  echo "watchdog pid $p $(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep ^MANIFEST)"; done
echo "watchdogs $(pgrep -fc "^/bin/bash simulation/sim_watchdog.sh") for $(echo $INST | wc -w) instances"
pgrep -af "wmediumd -c" | grep -v pgrep || echo "no wmediumd (not needed for normal flights)"
echo "radios /sys/class/ieee80211: $(ls /sys/class/ieee80211 | wc -l)"
df -h / | tail -1
$PY -I -c "import json,time;d=json.load(open('$OFF'));print('clock offset drift s', round(((time.time_ns()-time.monotonic_ns())-d['epoch_minus_monotonic_ns'])/1e9,3))"
echo "uav_ids.parm md5 $(md5sum simulation/ardupilot/uav_ids.parm | cut -c1-8)"
for k in $INST; do
  inst_vars $k
  echo; echo "########## instance $k ($P, $(basename $M))"
  echo "### health"
  for i in $IFS_; do echo "$i $(iw dev $i info 2>/dev/null | grep -oE 'type [a-z]*|channel [0-9]+' | tr '\n' ' ')"; done
  docker exec ${P}_orchestrator md5sum /opt/orchestrator/orchestrator.py /opt/orchestrator/telemetry_logger.py
  docker inspect -f '{{.Config.Cmd}}' ${P}_ardupilot | grep -q uav_ids.parm && echo "ardupilot with uav_ids.parm" || echo "! ardupilot WITHOUT uav_ids.parm"
  echo "### rtf_low (<0.75 since)"; [ -f $SL ] && $PY -I $S/rtf_low.py $SL "$SINCE" 0.75
  echo "### watchdog log since"; [ -f $WL ] && $PY -I -c "
for l in open('$WL', errors='replace'):
    if l[:1].isdigit() and l[:16] >= '$SINCE': print(l.rstrip())" | tail -60
  echo "### progress"; $PY -I $S/attack_progress.py $R $M
  echo "model time: $($PY -I $S/eta_model.py $R $M "$SINCE")"
  echo "### orchestrator log since"
  L=$O/orch_$TAG.log
  docker logs --since $SINCE_UTC ${P}_orchestrator > $L 2>&1
  echo "starts $(grep -c '=== Starting mission' $L), completed $(grep -c '=== mission_.* completed' $L), failed $(grep -c '=== mission_.* failed' $L)"
  grep -E "ERROR|WARNING|=== mission_.* failed|Paused|attempt|Read-back|Crash|Timeout\." $L | grep -v "Progress" | cut -c1-200
  echo "ground phase min/max: $(grep -o 'Ground phase sim/wall [0-9.]*' $L | awk '{print $4}' | sort -n | sed -n '1p;$p' | tr '\n' ' ')"
  echo "start offset max: $(grep -o 'Start offset from spawn point [0-9.]*' $L | awk '{print $6}' | sort -n | tail -1)"
  # GCS failsafe must not trigger in normal flights; PrecLand text at SITL start and
  # the RTL waypoint at the mission end are not failsafe events
  fs=$(grep -iE "failsafe|mode.*RTL|-> RTL" $L | grep -vE "PrecLand|Mission: [0-9]+ RTL|RTL_ALT")
  [ -n "$fs" ] && { echo "! GCS failsafe / RTL outside the mission plan:"; echo "$fs" | cut -c1-200 | head -20; } || echo "no GCS failsafe"
  echo "SmartRTL buffer full: $(grep -c 'SmartRTL deactivated' $L)"
  echo "### normal check (GCS + on-board telemetry)"; $PY -I $S/normal_check.py $R $(basename $M) $O/${TAG}_normal.json "$SINCE"
  echo "### in-flight RTF (10-s windows > 3 m, sim/wall < 0.9)"; $PY -I $S/flight_rtf.py $R $M "$SINCE" 2>&1 | tail -30
  echo "### data check (sensor values, all completed missions)"
  $PY -I $S/data_check.py $R $M $O/${TAG}_data.json > $O/${TAG}_data.txt 2>&1; head -40 $O/${TAG}_data.txt
done
echo; echo "### degraded/crashed new (all instances)"
find dataset/_degraded_runs dataset/_crashed_runs -maxdepth 1 -mindepth 1 -newermt "$SINCE" | sort
if [ -f $HOME/par_monitor.log ]; then
  echo "### parallel monitor (FLAG/ALERT since, cycles)"
  $PY -I -c "
n = 0
for l in open('$HOME/par_monitor.log', errors='replace'):
    if l[:16] < '$SINCE': continue
    if ' OK cycle' in l: n += 1
    else: print(l.rstrip()[:220])
print('OK cycles since', n)"
  tail -1 $HOME/par_monitor.log
fi
[ "$H" = hp ] && echo "chain rf_jamming: $(pgrep -fc "^bash .*chain_rf_after_v3.sh") process(es), $(tail -1 $HOME/deploy_20261108/chain_rf.log 2>/dev/null)"
if [ "$H" = tr ]; then
  echo "### tuning"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
  pgrep -af "tune.py" | grep -v pgrep | grep -c "worker-id" | sed 's/^/tune processes (incl. loader workers): /'
  # studies of the running workers plus those written in the last 24 h (a
  # finished study stays in the report for a day); no fixed study names
  ST=$(cd training && { pgrep -af "tune.py" | grep -o -- "--study-name [^ ]*" | cut -d' ' -f2
       find optuna_studies -maxdepth 1 -name '*.db' -mmin -1440 -printf '%f\n' | sed 's/\.db$//'; } | sort -u | tr '\n' ' ')
  echo "studies: $ST"
  [ -n "$ST" ] && (cd training && $PY -I $S/tune_status.py optuna_studies $ST)
  LOGS=$(find training -maxdepth 1 -name 'tune_*.log' -mmin -1440 | sort)
  [ -n "$LOGS" ] && grep -hiE "Traceback|Error|nan|inf |diverg|failed" $LOGS | grep -v UserWarning | tail -20
  [ -n "$LOGS" ] && ls -la --time-style=+%F_%T $LOGS | awk '{print $6, $7}'
  for c in $HOME/deploy_20261108/rssm_ext/chain_rssm_ext.log; do [ -f $c ] && { echo "chain rssm_v2r_ext:"; tail -3 $c; }; done
fi
echo "### DONE"
} > $O/run_$H.log 2>&1
