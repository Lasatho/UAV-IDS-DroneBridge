# Sensor noise model (dataset v4)

Noise model of the v4 flights (EKF3, since 2026-11-26). The files are the versions that run on the simulation hosts.

| File | Purpose |
|---|---|
| `gps_noise.lua` | SITL script: GPS position noise as first-order Gauss-Markov process per axis (σ 0.5 m horizontal, 1.0 m vertical, τ 30 s) through `SIM_GPS1_GLTCH_X/Y/Z`, plus the GNSS spoofing drift (command file `spooftrig/cmd`, written by `data/attacks/gnss_spoofing.py`; the comment there still names the earlier separate script `gnss_spoof.lua`, which this one replaces) |
| `noise_ctl.py` | sets and verifies the SITL parameters (IMU noise mask, `SIM_GYR1_RND`, `SIM_BARO_RND`, GPS velocity error, roll gains 0.475 × default) |
| `gz_imu_noise.py` | adds Gaussian IMU noise to the Gazebo model: `gz_imu_noise.py <original model.sdf> model_noise.sdf 0.1 0.024` turns `iris_with_standoffs/model.sdf` (md5 `4c7c59a1…`) into the deployed model (md5 `4fa293b6…`) |
| `noise_apply.sh` | applies the model to one paused instance (`docker cp` of `model_noise.sdf` and `gps_noise.lua`, parameters, restart, check at rest). Expects `model_noise.sdf` next to it. Both copies live in the container filesystem; recreating the container drops them |
| `lua_spoof_check.py` | check of the Lua spoofing path on one SITL (10 m north shift on, back after removal) |

The plausibility check of the recorded flights is `tools/noise_qc/noise_qc.py` (rest spread, spikes before the attack, roll limit cycle, pairing with the original mission). Since 2026-11-28 its GPS innovation uses the GPS sample time (`time_usec`), and the thresholds are min(previous, 2 × p99) over 137 missions.
