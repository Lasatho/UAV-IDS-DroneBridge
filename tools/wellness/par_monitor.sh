#!/bin/bash
# Permanent monitor of the parallel simulation instances on one host (read-only).
# Every CHECK_S: separation check of newly finished missions (check_separation.py),
# manifest integrity (every v3 mission at most once pending/completed over all
# instance manifests, moved rows stay moved), health of every instance (containers,
# watchdog, radios on their channel, SITL defaults with uav_ids.parm, RTF).
# Writes ~/par_monitor.log: "OK ..." per cycle, "FLAG ..." / "ALERT ..." on findings.
# usage: par_monitor.sh <n_instances>   (instances 0..n-1, instance 0 = docker-compose.yml)
N=$1; R="$(cd "$(dirname "$0")/../.." && pwd)"; D=$R/dataset; PY=$R/venv/bin/python
LOG=$HOME/par_monitor.log; STATE=$HOME/par_monitor.checked; CHECK_S=600
log(){ echo "$(date "+%F %T") $*" >> $LOG; }
log "monitor start: $N instances, repo $R"
while true; do
  issues=0
  # 1. separation of newly finished missions (labels written in the last 60 min, not yet OK)
  out=$($PY -I $R/tools/wellness/check_separation.py $D --since-min 60 --state $STATE 2>&1)
  n_ok=$(echo "$out" | grep -c " OK |")
  while IFS= read -r l; do
    [ -n "$l" ] || continue
    grep -qF "$l" $LOG.flags 2>/dev/null && continue   # report a flag once
    echo "$l" >> $LOG.flags; log "FLAG $l"; issues=$((issues + 1))
  done <<< "$(echo "$out" | grep -v " OK |")"
  # 2. manifest integrity
  mi=$($PY -I - $D <<'PY' 2>&1
import csv, collections, glob, sys
d = sys.argv[1]
c, bad = collections.Counter(), []
for f in sorted(glob.glob(f"{d}/missions_manifest_v3_tr*.csv")):
    for r in csv.DictReader(open(f)):
        if r["status"] in ("pending", "completed", "failed"):
            c[r["mission_id"]] += 1
        elif not r["status"].startswith("moved_tr"):
            bad.append(f"{r['mission_id']} status {r['status']} in {f.rsplit('/', 1)[1]}")
dup = [m for m, n in c.items() if n > 1]
print(f"manifests {len(c)} active, dup {len(dup)} {dup[:5]}, odd {bad[:5]}")
sys.exit(1 if dup or bad else 0)
PY
)
  [ $? -eq 0 ] || { log "ALERT manifest integrity: $mi"; issues=$((issues + 1)); }
  # 3. health per instance
  for k in $(seq 0 $((N - 1))); do
    if [ $k -eq 0 ]; then P=uav_ids; IFS_=(wlan0 wlan1 wlan2); CH=6; SL=$HOME/sim_degradation.log
    else P=uav_ids_$k; . $R/simulation/instances/$k.env; IFS_=($IF_DRONE $IF_GCS $IF_ATTACK); CH=$CHANNEL; SL=$HOME/sim_degradation_$k.log; fi
    for c in gazebo ardupilot db_uav db_gcs socat mavproxy orchestrator; do
      [ "$(docker inspect -f '{{.State.Running}}' ${P}_$c 2>/dev/null)" = "true" ] || { log "ALERT instance $k: ${P}_$c not running"; issues=$((issues + 1)); }
    done
    docker inspect -f '{{.Config.Cmd}}' ${P}_ardupilot 2>/dev/null | grep -q uav_ids.parm || { log "ALERT instance $k: ardupilot without uav_ids.parm"; issues=$((issues + 1)); }
    for i in "${IFS_[@]}"; do
      info=$(iw dev $i info 2>/dev/null)
      echo "$info" | grep -q "type monitor" && echo "$info" | grep -q "channel $CH " || { log "ALERT instance $k: $i not monitor on channel $CH"; issues=$((issues + 1)); }
    done
    low=$(tail -3 $SL 2>/dev/null | grep -oE "real_time_factor[:= ]+[0-9.]+" | grep -oE "[0-9.]+$" | awk '$1 < 0.75' | wc -l)
    [ "$low" -ge 3 ] && { log "ALERT instance $k: RTF < 0.75 in the last 3 samples"; issues=$((issues + 1)); }
  done
  # 4. live radio check: 20 s capture on every radio of every instance; a frame on a
  #    foreign channel stops the radio proxies of the leaking instance k >= 1
  #    (protective action, instance 0 is never touched).
  specs=""; declare -A chan_of=()
  for k in $(seq 0 $((N - 1))); do
    if [ $k -eq 0 ]; then ifs="wlan0 wlan1 wlan2"; f=2437
    else . $R/simulation/instances/$k.env; ifs="$IF_DRONE $IF_GCS $IF_ATTACK"; f=$FREQ_MHZ; fi
    chan_of[$f]=$k
    for i in $ifs; do specs="$specs $i=$f"; done
  done
  C=$HOME/par_monitor_cap; rm -rf $C; mkdir -p $C
  docker run --rm --privileged --network host -v $C:/c simulation-orchestrator sh -c     "for s in $specs; do i=\${s%=*}; (timeout 20 tcpdump -i \$i -w /c/\$i.pcap 2>/dev/null &); done; sleep 22; chmod a+r /c/*" > /dev/null 2>&1
  lr=$($PY -I $R/tools/wellness/live_radio_check.py $C $specs 2>&1)
  if echo "$lr" | grep -q "^FOREIGN\|^MISSING"; then
    log "ALERT live radio: $(echo "$lr" | grep "^FOREIGN\|^MISSING" | tr '\n' ';')"; issues=$((issues + 1))
    for fq in $(echo "$lr" | grep "^FOREIGN" | grep -oE "[0-9]{4}MHz" | cut -c1-4 | sort -u); do
      k=${chan_of[$fq]:-}
      if [ -n "$k" ] && [ "$k" -ge 1 ]; then
        docker stop uav_ids_${k}_db_gcs uav_ids_${k}_db_uav > /dev/null 2>&1
        log "ALERT protective stop: radio proxies of instance $k ($fq MHz) stopped"
      fi
    done
  fi
  nwd=$(pgrep -fc "^/bin/bash simulation/sim_watchdog.sh")
  [ "$nwd" -eq "$N" ] || { log "ALERT $nwd watchdogs for $N instances"; issues=$((issues + 1)); }
  log "OK cycle: $n_ok missions newly checked OK, $mi, issues $issues, load $(cut -d' ' -f1-3 /proc/loadavg)"
  sleep $CHECK_S
done
