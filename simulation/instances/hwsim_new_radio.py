"""Create persistent mac80211_hwsim radios (no interface, NO_VIF) via generic netlink.
They stay until the module is reloaded or the host reboots. usage: hwsim_new_radio.py <phy-name>..."""
import socket
import struct
import sys

NETLINK_GENERIC, GENL_ID_CTRL, CTRL_CMD_GETFAMILY, CTRL_ATTR_FAMILY_ID, CTRL_ATTR_FAMILY_NAME = 16, 0x10, 3, 1, 2
HWSIM_CMD_NEW_RADIO, A_RADIO_NAME, A_NO_VIF = 4, 17, 18
NLM_F_REQUEST, NLM_F_ACK = 1, 4


def attr(t, data=b""):
    n = 4 + len(data)
    return struct.pack("HH", n, t) + data + b"\0" * ((4 - n % 4) % 4)


def msg(fam, cmd, attrs, seq):
    body = struct.pack("BBH", cmd, 1, 0) + attrs
    return struct.pack("IHHII", 16 + len(body), fam, NLM_F_REQUEST | NLM_F_ACK, seq, 0) + body


s = socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, NETLINK_GENERIC)
s.bind((0, 0))
s.send(msg(GENL_ID_CTRL, CTRL_CMD_GETFAMILY, attr(CTRL_ATTR_FAMILY_NAME, b"MAC80211_HWSIM\0"), 1))
d = s.recv(65536)
p, fam = 20, None
while p < len(d):
    n, t = struct.unpack_from("HH", d, p)
    if t == CTRL_ATTR_FAMILY_ID:
        fam = struct.unpack_from("H", d, p + 4)[0]
    p += (n + 3) & ~3
if fam is None:
    sys.exit("MAC80211_HWSIM family not found")
for i, name in enumerate(sys.argv[1:]):
    s.send(msg(fam, HWSIM_CMD_NEW_RADIO, attr(A_NO_VIF) + attr(A_RADIO_NAME, name.encode() + b"\0"), 10 + i))
    r = s.recv(65536)
    err = struct.unpack_from("i", r, 16)[0]
    print(name, "result", err, flush=True)
    if err < 0:
        sys.exit(f"{name}: error {err}")
