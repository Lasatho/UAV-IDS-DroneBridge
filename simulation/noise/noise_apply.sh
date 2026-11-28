#!/bin/bash
# Noise model onto one paused SITL instance (Lasse 2026-11-26 "GPS-Werte so umsetzen", Vault sensorrauschen):
# Gazebo IMU noise (model_noise.sdf), gps_noise.lua into /home/ardupilot/scripts, parameters (noise_ctl.py set),
# restart gazebo+ardupilot in the watchdog's order, verify at rest. Both are docker cp into the container
# filesystem: they survive docker restart (watchdog) and are lost only when the container is recreated.
# usage: noise_apply.sh PREFIX INSTANCE IMAGE   (e.g. uav_ids_2 2 simulation-orchestrator)
set -u
P=$1; K=$2; IMG=$3; W=$(cd "$(dirname "$0")" && pwd); L=$W/apply.log; PORT=$((5762 + 10 * K))
log(){ echo "$(date "+%F %T") [$P] $*" | tee -a $L; }
ctl(){ docker run --rm --network host -v $W:/w $IMG python3 /w/noise_ctl.py $PORT "$@"; }
log "start, SITL port $PORT"
docker logs --tail 3 ${P}_orchestrator 2>&1 | grep -qiE "pause|idle" || { log "STOP: orchestrator not paused/idle"; exit 1; }
M=/opt/ardupilot_gazebo/models/iris_with_standoffs/model.sdf
[ "$(docker exec ${P}_gazebo md5sum $M | cut -c1-8)" = 4c7c59a1 ] || log "note: gazebo model was not the original"
docker cp $W/model_noise.sdf ${P}_gazebo:$M || { log "STOP: cp model"; exit 1; }
log "gazebo model md5 $(docker exec ${P}_gazebo md5sum $M | cut -c1-8)"
docker exec -u root ${P}_ardupilot mkdir -p /home/ardupilot/scripts
docker cp $W/gps_noise.lua ${P}_ardupilot:/home/ardupilot/scripts/gps_noise.lua || { log "STOP: cp lua"; exit 1; }
docker exec -u root ${P}_ardupilot chown -R ardupilot:ardupilot /home/ardupilot/scripts
log "lua md5 $(docker exec ${P}_ardupilot md5sum /home/ardupilot/scripts/gps_noise.lua | cut -c1-8)"
ctl set >> $L 2>&1; tail -13 $L | grep -q "PARAMS_BAD 0" || { log "STOP: params not set"; exit 1; }
log "restart gazebo + ardupilot"; docker restart ${P}_gazebo ${P}_ardupilot >> $L 2>&1
sleep 75; ctl verify 90 >> $L 2>&1; tail -3 $L
tail -2 $L | grep -q "VERIFY OK" && log DONE_OK || log DONE_FAILED
