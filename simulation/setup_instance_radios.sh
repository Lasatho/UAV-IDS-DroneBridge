#!/bin/bash
# Radios of simulation instance k: three persistent mac80211_hwsim radios, monitor
# interfaces on the instance channel (like start_sim.sh for wlan0-2). Idempotent.
# No sudo on the host: runs in a privileged container with host network.
# usage: setup_instance_radios.sh <k>
set -eu
R="$(cd "$(dirname "$0")/.." && pwd)"; set -a; . "$R/simulation/instances/$1.env"; set +a
docker run --rm --privileged --network host -v "$R/simulation/instances:/i:ro" \
  -e PHY_DRONE -e PHY_GCS -e PHY_ATTACK -e IF_DRONE -e IF_GCS -e IF_ATTACK -e CHANNEL alpine sh -c '
apk add -q iw python3 >/dev/null 2>&1 || { echo "apk failed"; exit 1; }
for phy in $PHY_DRONE $PHY_GCS $PHY_ATTACK; do
  [ -e /sys/class/ieee80211/$phy ] || python3 /i/hwsim_new_radio.py $phy
done
add(){ phy=$1; dev=$2
  [ -e /sys/class/net/$dev ] || iw phy $phy interface add $dev type monitor
  ip link set $dev down; iw dev $dev set type monitor; ip link set $dev mtu 2304; ip link set $dev up
  iw dev $dev set channel $CHANNEL
  echo "$dev $(iw dev $dev info | grep -oE "type [a-z]+|channel [0-9]+ \([0-9]+ MHz\)" | tr "\n" " ")"
}
add $PHY_DRONE $IF_DRONE; add $PHY_GCS $IF_GCS; add $PHY_ATTACK $IF_ATTACK'
