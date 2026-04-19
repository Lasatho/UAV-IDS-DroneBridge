#!/bin/bash
set -e

# ─── Kernel: mac80211_hwsim ───────────────────────────────────────────────────
echo "[*] Loading mac80211_hwsim..."
sudo modprobe -r mac80211_hwsim 2>/dev/null || true
sudo modprobe mac80211_hwsim radios=3 support_p2p_device=0

echo "[*] Configuring interfaces..."
for iface in wlan0 wlan1 wlan2; do
    sudo ip link set $iface down
    sudo iw dev $iface set type monitor
    sudo ip link set $iface up
    sudo iw dev $iface set channel 6
done
sudo ip link set hwsim0 up

echo "[*] Interfaces ready:"
iw dev | grep -E "Interface|type|channel"

# ─── Docker ───────────────────────────────────────────────────────────────────
echo "[*] Starting simulation stack..."
cd "$(dirname "$0")"
docker compose up