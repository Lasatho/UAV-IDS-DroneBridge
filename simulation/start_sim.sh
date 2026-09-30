#!/bin/bash

# ─── Cleanup ──────────────────────────────────────────────────────────────────
echo "[*] Cleaning up..."
docker compose down 2>/dev/null || true
sudo pkill -f db_proxy 2>/dev/null || true
sudo pkill -f arducopter 2>/dev/null || true
sudo pkill -f "gz sim" 2>/dev/null || true
sudo pkill -f mavproxy 2>/dev/null || true
sudo pkill -f socat 2>/dev/null || true
sudo pkill -f wmediumd 2>/dev/null || true

# ─── Kernel: mac80211_hwsim ───────────────────────────────────────────────────
echo "[*] Loading mac80211_hwsim..."
sudo modprobe -r mac80211_hwsim 2>/dev/null || true
sleep 1
sudo modprobe mac80211_hwsim radios=3 support_p2p_device=0
sleep 1

# ─── Medium model: wmediumd (opt-in) ──────────────────────────────────────────
# Adds an interference model to hwsim so RF jamming actually degrades the link
# (plain hwsim has none). Benign link stays clean (high SNR in the config).
# Opt-in via USE_WMEDIUMD=1 — it is still being validated against the
# monitor-mode DroneBridge link, so it is OFF by default to keep the normal
# sim reliable. Build once with wmediumd/setup_wmediumd.sh.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ "${USE_WMEDIUMD:-0}" = "1" ]; then
    if command -v wmediumd >/dev/null 2>&1; then
        echo "[*] Starting wmediumd (USE_WMEDIUMD=1)..."
        sudo wmediumd -c "${SCRIPT_DIR}/wmediumd/wmediumd.cfg" >/tmp/wmediumd.log 2>&1 &
        sleep 2
        if pgrep -f wmediumd >/dev/null; then
            echo "    wmediumd running (log: /tmp/wmediumd.log)"
        else
            echo "    WARNING: wmediumd failed to start — see /tmp/wmediumd.log"
        fi
    else
        echo "[!] USE_WMEDIUMD=1 but wmediumd not installed."
        echo "    Build it once: simulation/wmediumd/setup_wmediumd.sh"
    fi
fi

echo "[*] Configuring interfaces..."
for iface in wlan0 wlan1 wlan2; do
    sudo ip link set $iface down
    sudo iw dev $iface set type monitor
    sudo ip link set $iface mtu 2304
    sudo ip link set $iface up
    sudo iw dev $iface set channel 6 || true
done
sudo ip link set hwsim0 up  

echo "[*] Verifying interfaces..."
for iface in wlan0 wlan1 wlan2; do
    echo "--- $iface ---"
    iw dev $iface info | grep -E "type|channel"
done

# ─── Docker ───────────────────────────────────────────────────────────────────
echo "[*] Starting simulation stack..."
export REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$(dirname "$0")"
docker compose build orchestrator
docker compose up
