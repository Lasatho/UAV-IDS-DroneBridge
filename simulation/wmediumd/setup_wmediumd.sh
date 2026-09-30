#!/bin/bash
# One-time build + install of wmediumd (not packaged for Ubuntu).
# Run once on the simulation host. Requires sudo (apt + install to /usr/local/bin).
set -e

WMEDIUMD_REF="master"
BUILD_DIR="${HOME}/wmediumd_build"

echo "[*] Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y git gcc make pkg-config \
    libnl-3-dev libnl-genl-3-dev libconfig-dev

echo "[*] Cloning wmediumd..."
rm -rf "${BUILD_DIR}"
git clone https://github.com/bcopeland/wmediumd "${BUILD_DIR}"
cd "${BUILD_DIR}"
git checkout "${WMEDIUMD_REF}"

echo "[*] Building..."
make

echo "[*] Installing to /usr/local/bin/wmediumd..."
sudo install -m 0755 wmediumd/wmediumd /usr/local/bin/wmediumd

echo "[*] Done. wmediumd installed:"
/usr/local/bin/wmediumd -h 2>&1 | head -3 || true
