#!/bin/bash
# Start wmediumd once the DroneBridge link is up.
#
# wmediumd must not register with hwsim before the DroneBridge containers are
# running: if it does, their startup invalidates its state and it dies with
# netlink EINVAL ("nl: cmd 2 ... Invalid argument"). This waits for a vehicle
# heartbeat first, then launches wmediumd for the rest of the session.
#
# Needs root (hwsim netlink). Launched by start_sim.sh when USE_WMEDIUMD=1, or
# run manually with sudo after the stack is up.
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="${SCRIPT_DIR}/wmediumd.cfg"
IMAGE="simulation-orchestrator:latest"

echo "[wmediumd] waiting for DroneBridge heartbeat before starting..."
for i in $(seq 1 60); do
    if docker run --rm --network host --entrypoint python3 "$IMAGE" -c '
from pymavlink import mavutil
import sys, time
c = mavutil.mavlink_connection("udpin:127.0.0.1:14553")
t = time.time()
while time.time() - t < 4:
    m = c.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
    if m and m.get_srcSystem() == 1:
        sys.exit(0)
sys.exit(1)
' 2>/dev/null; then
        echo "[wmediumd] heartbeat seen — starting wmediumd"
        exec wmediumd -c "$CFG"
    fi
    sleep 5
done

echo "[wmediumd] no heartbeat after wait — not starting" >&2
exit 1
