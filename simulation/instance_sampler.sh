#!/bin/bash
# RTF sampler of simulation instance k (like ~/sim_degradation_monitor.sh for instance 0),
# feeds the watchdog of the instance. usage: instance_sampler.sh <k>
K=$1; P=uav_ids_${K}; LOG=~/sim_degradation_${K}.log; WORLD=iris_runway
while true; do
  ts=$(date "+%Y-%m-%d %H:%M:%S")
  st=$(timeout -s KILL 15 docker stats --no-stream --format "{{.Name}} {{.CPUPerc}} {{.MemUsage}}" ${P}_gazebo ${P}_ardupilot 2>/dev/null | tr "\n" " ")
  load=$(uptime | sed "s/.*load average: //")
  rtf=$(timeout -s KILL 6 docker exec ${P}_orchestrator gz topic -e -t /world/$WORLD/stats -n 1 2>/dev/null | grep -i real_time_factor | head -1 | tr -d " ")
  [ -z "$rtf" ] && rtf="real_time_factor=NA"
  echo "$ts | $st load=$load | $rtf" >> "$LOG"
  sleep 120
done
