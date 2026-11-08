#!/bin/bash
# Start simulation instance k (k >= 1) next to the running instance 0: radios on the
# instance channel, Gazebo model with the instance FDM port, compose project uav_ids_<k>,
# FS_GCS_ENABLE check, watchdog and RTF sampler of the instance.
# usage: start_instance.sh <k> <manifest> [<log>]
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"; K=$1; M=$2; L=${3:-$HOME/deploy_20261108/par/start_instance_$K.log}
log(){ echo "$(date "+%F %T") $*" | tee -a $L; }
set -a; . $R/simulation/instances/$K.env; set +a
export REPO_ROOT=$R MANIFEST=$M ORCH_PY=$HOME/deploy_20261108/par/orchestrator_instance.py
cd $R/simulation
[ "$K" -ge 1 ] || { log "k must be >= 1"; exit 1; }
[ -f $R/dataset/$M ] || { log "manifest missing"; exit 1; }
[ -f $ORCH_PY ] || { log "orchestrator copy missing"; exit 1; }
log "instance $K: manifest $M, channel $CHANNEL ($FREQ_MHZ MHz), orchestrator $(md5sum < $ORCH_PY | cut -c1-8)"
bash $R/simulation/setup_instance_radios.sh $K >> $L 2>&1 || { log "radio setup failed"; exit 1; }
tail -3 $L
SDF=instances/iris_with_gimbal_fdm$FDM_PORT.sdf
docker run --rm simulation-gazebo cat /opt/ardupilot_gazebo/models/iris_with_gimbal/model.sdf \
  | sed "s#<fdm_port_in>9002</fdm_port_in>#<fdm_port_in>$FDM_PORT</fdm_port_in>#" > $SDF
grep -q "<fdm_port_in>$FDM_PORT</fdm_port_in>" $SDF || { log "model FDM port not set"; exit 1; }
rm -f $R/dataset/.stop_orchestrator_$K $R/dataset/.pause_orchestrator_$K
C="docker compose -f docker-compose.instance.yml --env-file instances/$K.env"
$C up -d --no-build --force-recreate gazebo ardupilot >> $L 2>&1; sleep 10
$C up -d --no-build --force-recreate --no-deps dronebridge_uav dronebridge_gcs socat mavproxy >> $L 2>&1; sleep 20
$C up -d --no-build --force-recreate --no-deps orchestrator >> $L 2>&1; sleep 15
docker exec uav_ids_${K}_orchestrator md5sum /opt/orchestrator/orchestrator.py >> $L 2>&1
docker exec uav_ids_${K}_orchestrator python3 -c "
from pymavlink import mavutil
c = mavutil.mavlink_connection('tcp:127.0.0.1:$SITL_SERIAL1'); c.wait_heartbeat(timeout=30)
c.mav.param_request_read_send(c.target_system, c.target_component, b'FS_GCS_ENABLE', -1)
m = c.recv_match(type='PARAM_VALUE', blocking=True, timeout=10)
print('PARAM', m.param_id if m else None, m.param_value if m else None)
" >> $L 2>&1
tail -1 $L | grep -q "PARAM FS_GCS_ENABLE 1.0" && log "failsafe OK" || log "WARNING failsafe not confirmed"
cd $R
pkill -f "instance_sampler.sh $K\$"; setsid nohup bash simulation/instance_sampler.sh $K > /dev/null 2>&1 < /dev/null &
UAV_IDS_PREFIX=uav_ids_$K FLAG_SUFFIX=_$K RADIO_IFACES="$IF_DRONE $IF_GCS $IF_ATTACK" MANIFEST=$M \
  SAMPLOG=$HOME/sim_degradation_$K.log WDLOG=$HOME/sim_watchdog_$K.log \
  setsid nohup simulation/sim_watchdog.sh > /dev/null 2>&1 < /dev/null &
sleep 2; log "watchdog $(pgrep -f "^/bin/bash simulation/sim_watchdog.sh" | tr "\n" " ")"
echo "$(date "+%F %T") manual: instance $K started with $M" >> $HOME/sim_watchdog_$K.log
log "done"
