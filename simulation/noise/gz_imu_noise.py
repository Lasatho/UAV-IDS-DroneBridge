"""Insert (or replace) Gaussian <imu> noise into the imu_sensor of iris_with_standoffs/model.sdf (Lasse 2026-11-25,
Vault sensorrauschen). Gazebo standard IMU noise model, no bias. usage: gz_imu_noise.py IN OUT ACC_STD GYR_STD"""
import re, sys
src, dst, acc, gyr = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
s = open(src).read()
def axes(sd): return "".join(f"<{a}><noise type=\"gaussian\"><mean>0</mean><stddev>{sd}</stddev></noise></{a}>" for a in "xyz")
imu = f"\n        <imu><angular_velocity>{axes(gyr)}</angular_velocity><linear_acceleration>{axes(acc)}</linear_acceleration></imu>"
m = re.search(r'(<sensor name="imu_sensor" type="imu">)(.*?)(\s*</sensor>)', s, re.S)
body = re.sub(r"\s*<imu>.*?</imu>", "", m.group(2), flags=re.S)
s = s[:m.start()] + m.group(1) + body + imu + m.group(3) + s[m.end():]
open(dst, "w").write(s)
