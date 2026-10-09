#!/bin/bash
# =============================================================================
# sim_watchdog.sh — keeps an unattended dataset run going.
#
# WORKAROUND, not a root-cause fix: it limits downtime from failures whose
# causes are still open (gazebo RTF degradation over time; DroneBridge exiting
# under rf_jamming, which can kill MAVProxy's main loop). See the attack
# dataset design notes, sections 8, 10 and 11.5.
#
# Checks every CHECK_S seconds:
#   1. Stop flag (.stop_orchestrator, set by the orchestrator after 5
#      consecutive failures or when no vehicle heartbeat comes back after a
#      SITL reboot)                                         -> full restart
#   2. A stack container not running for > DOWN_GRACE_S   -> full restart
#      (socat is excluded: it restarts often on its own and is not on the
#      MAVLink path of the orchestrator)
#   3. MAVProxy main loop died ("Exception in thread main_loop") -> full restart
#   4. Orchestrator printed nothing for > STALL_S          -> full restart
#   5. gazebo RTF below RTF_THRESH in the last RTF_SAMPLES sampler entries
#      (single low samples occur during SITL reboots and are ignored)
#      -> restart gazebo + ardupilot only, and only between missions: the
#         watchdog sets the pause flag, the orchestrator finishes the
#         current mission and holds ("Paused"), the watchdog restarts
#         gazebo + ardupilot and removes the flag, the orchestrator reboots
#         the SITL and continues. No time limit on the wait (missions run
#         up to 75 min); a hung mission is caught by check 4. A pause flag
#         set by the orchestrator (degraded ground phase before takeoff) is
#         handled the same way.
#   6. RTF_SAMPLES consecutive sampler entries without an answer from
#      gazebo (real_time_factor=NA) while it runs -> full restart
# Full restart: stop orchestrator, restart gazebo, ardupilot, DroneBridge and
# MAVProxy, reset failed missions to pending, start orchestrator again. The
# interrupted mission is still "pending" in the manifest and is flown again.
# Gives up (orchestrator stopped, notification) if the radio interfaces are
# not in monitor mode (needs sudo, see start_sim.sh) or after more than
# MAX_RESTARTS_PER_H full restarts within an hour.
#
# Usage (detached, survives logout):
#   MANIFEST=missions_manifest_attacks.csv setsid nohup simulation/sim_watchdog.sh \
#       >/dev/null 2>&1 < /dev/null &
# =============================================================================

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$REPO_ROOT/dataset/${MANIFEST:-missions_manifest.csv}"
FLAG="$REPO_ROOT/dataset/.stop_orchestrator"
PAUSE="$REPO_ROOT/dataset/.pause_orchestrator"
SAMPLOG=${SAMPLOG:-~/sim_degradation.log}
WDLOG=${WDLOG:-~/sim_watchdog.log}
NTFY_TOPIC=${NTFY_TOPIC:-}

CHECK_S=30
RTF_THRESH=0.75
RTF_SAMPLES=3        # sampler writes every ~2 min
RTF_COOLDOWN_S=420   # > RTF_SAMPLES sampler periods: judge only fresh samples
DOWN_GRACE_S=120
STALL_S=1800         # > longest silent phase: RTL from ~2.9 km at 3 m/s (~970 s), attack timeout 600 s
MAX_RESTARTS_PER_H=4

SIM=(uav_ids_gazebo uav_ids_ardupilot)
LINK=(uav_ids_db_uav uav_ids_db_gcs uav_ids_mavproxy)
ORCH=uav_ids_orchestrator

log(){ echo "$(date "+%F %T") $*" >> "$WDLOG"; }
notify(){
  [ -n "$NTFY_TOPIC" ] && curl -s -m 5 -d "[UAV-IDS watchdog] $*" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null
}
running(){ [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = "true" ]; }

interfaces_ok(){
  local i
  for i in wlan0 wlan1 wlan2; do
    iw dev "$i" info 2>/dev/null | grep -q "type monitor" || return 1
  done
}

reset_failed(){
  python3 - "$MANIFEST" <<'PY' >> "$WDLOG" 2>&1
import csv, collections, sys
p = sys.argv[1]
rows = list(csv.DictReader(open(p)))
n = 0
for r in rows:
    if r["status"] == "failed":
        r["status"] = "pending"; n += 1
w = csv.DictWriter(open(p, "w", newline=""), fieldnames=rows[0].keys())
w.writeheader(); w.writerows(rows)
print("  reset", n, dict(collections.Counter(r["status"] for r in rows)))
PY
}

give_up(){
  log "GIVING UP: $* -> orchestrator stopped, manual intervention needed"
  notify "giving up: $*"
  docker stop "$ORCH" >> "$WDLOG" 2>&1
  exit 1
}

restarts=()
full_restart(){
  local why="$1" now
  now=$(date +%s)
  local recent=()
  for t in "${restarts[@]}"; do [ $((now - t)) -lt 3600 ] && recent+=("$t"); done
  restarts=("${recent[@]}" "$now")
  [ ${#restarts[@]} -gt $MAX_RESTARTS_PER_H ] && give_up "$why (more than $MAX_RESTARTS_PER_H full restarts within 1 h)"
  interfaces_ok || give_up "$why; radio interfaces not in monitor mode"

  log "$why -> full restart"
  docker stop "$ORCH" >> "$WDLOG" 2>&1
  docker restart "${SIM[@]}" >> "$WDLOG" 2>&1
  sleep 10
  docker restart "${LINK[@]}" >> "$WDLOG" 2>&1
  sleep 20
  reset_failed
  rm -f "$FLAG" "$PAUSE"
  pause_since=""
  docker start "$ORCH" >> "$WDLOG" 2>&1
  log "full restart done"
  last_rtf_restart=$(date +%s)
  mavproxy_checked=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  declare -gA down_since=()
}

declare -A down_since=()
# judge only samples taken after the watchdog started (old log entries may
# be from before a restart or fix)
last_rtf_restart=$(date +%s)
pause_since=""
rm -f "$PAUSE"
mavproxy_checked=$(date -u +%Y-%m-%dT%H:%M:%SZ)
log "watchdog start (manifest=$MANIFEST, rtf<$RTF_THRESH, stall>${STALL_S}s)"

while true; do
  sleep "$CHECK_S"
  now=$(date +%s)

  # 1. stop flag
  if [ -f "$FLAG" ]; then full_restart "STOP-FLAG"; continue; fi

  # 2. containers down
  reason=""
  for c in "${SIM[@]}" "${LINK[@]}" "$ORCH"; do
    if running "$c"; then
      unset "down_since[$c]"
    else
      : "${down_since[$c]:=$now}"
      [ $((now - down_since[$c])) -ge $DOWN_GRACE_S ] && reason="$c not running for ${DOWN_GRACE_S}s"
    fi
  done
  if [ -n "$reason" ]; then full_restart "$reason"; continue; fi

  # 3. MAVProxy main loop died (process and container stay up)
  since=$mavproxy_checked
  mavproxy_checked=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if docker logs --since "$since" uav_ids_mavproxy 2>&1 | grep -q "Exception in thread main_loop"; then
    full_restart "MAVProxy main loop died"; continue
  fi

  # 4. orchestrator stalled
  if running "$ORCH" && [ -z "$(docker logs --since "${STALL_S}s" "$ORCH" 2>&1 | head -1)" ]; then
    full_restart "orchestrator silent for ${STALL_S}s"; continue
  fi

  # 5. RTF degradation (proactive, cheap restart between missions). The
  #    orchestrator sets the pause flag itself when it measures a degraded
  #    ground phase; handle that like a low-RTF pause.
  if [ -z "$pause_since" ] && [ -f "$PAUSE" ]; then
    pause_since=$(date -u -d "@$(stat -c %Y "$PAUSE")" +%Y-%m-%dT%H:%M:%SZ)
    log "pause flag set by orchestrator -> restart gazebo+ardupilot when paused"
  fi
  if [ -n "$pause_since" ]; then
    if docker logs --since "$pause_since" "$ORCH" 2>&1 | grep -q "Paused (watchdog flag)"; then
      log "orchestrator paused -> restart gazebo+ardupilot"
      docker restart "${SIM[@]}" >> "$WDLOG" 2>&1
      sleep 10
      rm -f "$PAUSE"
      pause_since=""
      last_rtf_restart=$(date +%s)
      log "restart done, pause flag removed"
    fi
  elif [ $((now - last_rtf_restart)) -ge $RTF_COOLDOWN_S ]; then
    samples=$(grep -oE "real_time_factor[:= ]+([0-9.]+|NA)" "$SAMPLOG" 2>/dev/null \
              | tail -"$RTF_SAMPLES" | grep -oE "([0-9.]+|NA)$" | tr "\n" " ")
    read -r n low na <<< "$(echo "$samples" | awk -v t="$RTF_THRESH" '{
        for (i = 1; i <= NF; i++) { if ($i == "NA") na++; else if ($i < t) low++ }
        print NF, low + 0, na + 0 }')"
    # 6. gazebo not answering
    if [ "$n" -eq "$RTF_SAMPLES" ] && [ "$na" -eq "$RTF_SAMPLES" ] && running uav_ids_gazebo; then
      full_restart "gazebo stats not answering ($RTF_SAMPLES samples NA)"; continue
    fi
    if [ "$n" -eq "$RTF_SAMPLES" ] && [ "$low" -eq "$RTF_SAMPLES" ]; then
      log "RTF [ $samples] < $RTF_THRESH -> pause flag, restart after current mission"
      touch "$PAUSE"
      pause_since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    fi
  fi
done
