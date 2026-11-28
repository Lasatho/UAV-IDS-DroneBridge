"""Lua spoofing check on one SITL (v4 switch): baseline GPS, command file "2.0 0.0 10.0" (2 m/s north, 10 m) for
12 s, then removed. Pass: north shift 7-13 m while on (GPS noise sigma 0.5 m), back within 3 m afterwards.
usage: lua_spoof_check.py PORT CMD_FILE"""
import os, statistics as st, sys, time
from pymavlink import mavutil
port, cmdf = sys.argv[1], sys.argv[2]
c = mavutil.mavlink_connection(f"tcp:127.0.0.1:{port}", source_system=251); c.wait_heartbeat(timeout=60)
c.mav.command_long_send(c.target_system, c.target_component, 511, 0, 24, 200000, 0, 0, 0, 0, 0)
def rec(sec):
    v = []; t0 = time.time()
    while time.time() - t0 < sec:
        m = c.recv_match(type="GPS_RAW_INT", blocking=True, timeout=1)
        if m and m.fix_type >= 3: v.append(m.lat * 1e-7 * 111320)
    return st.mean(v) if v else None
if os.path.exists(cmdf): os.remove(cmdf)
base = rec(6)
with open(cmdf + ".tmp", "w") as f: f.write("2.0 0.0 10.0\n")
os.replace(cmdf + ".tmp", cmdf); rec(9); on = rec(3)
os.remove(cmdf); rec(8); off = rec(4)
sh, back = (on - base) if base and on else None, (off - base) if base and off else None
print(f"shift north on {sh} m (expect ~10), after removal {back} m (expect ~0)")
print("LUA_SPOOF OK" if sh is not None and 7 <= sh <= 13 and back is not None and abs(back) < 3 else "LUA_SPOOF FAILED")
