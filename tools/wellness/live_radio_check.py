#!/usr/bin/env python3
"""Frames per radiotap channel frequency in short captures of every instance radio.
usage: live_radio_check.py <capture_dir> <iface>=<own_freq_mhz> ...
Prints one line per interface; 'FOREIGN <iface> <freq>:<n> ...' if frames of another channel
were received (own transmitted frames carry no channel field and are ignored)."""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data"))
import pcap_to_csv as P  # noqa: E402

d = Path(sys.argv[1])
bad = 0
for spec in sys.argv[2:]:
    iface, own = spec.split("=")
    f = d / f"{iface}.pcap"
    if not f.exists():
        print(f"MISSING {iface}"); bad += 1; continue
    c = collections.Counter(P.radiotap_freq_mhz(b) for _, b, _ in P.parse_pcap_frames(f))
    foreign = {k: v for k, v in c.items() if k is not None and k != int(own)}
    tag = "FOREIGN " + " ".join(f"{k}MHz:{v}" for k, v in sorted(foreign.items())) + " " if foreign else "ok "
    print(tag + f"| {iface} own {own}: {dict(c)}")
    bad += bool(foreign)
sys.exit(1 if bad else 0)
