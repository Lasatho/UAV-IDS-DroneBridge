"""
wind_model.py — time-varying wind for one mission.

Generates the mean wind vector the orchestrator publishes to Gazebo's
WindEffects system during a flight. The wind profile from
generate_missions.py (speed_min, speed_max, turbulence) sets:

  - a slowly varying base speed: Ornstein-Uhlenbeck process around the base
    mean with time constant tau_s and stationary std sigma_frac * range,
  - gusts (medium/high turbulence): Poisson arrivals, each a 1-cosine
    discrete gust shape (as in MIL-F-8785C) with random amplitude and
    duration, added on top of the base speed,
  - direction: OU process around the nominal direction (wind blows *from*
    that azimuth).

Speed is clipped to [speed_min, speed_max]. Deterministic for a given seed.
"""

import math
import random

# turbulence level -> process parameters
#   tau_s:       base speed OU time constant [s]
#   sigma_frac:  base speed stationary std as fraction of (max - min)
#   base_frac:   base mean position within [min, max] (0 = min, 1 = max)
#   gust_rate:   mean gust arrivals per second
#   gust_amp:    gust amplitude range as fraction of (max - min)
#   gust_dur:    gust duration range [s]
#   dir_sigma:   direction OU stationary std [deg]
#   dir_tau:     direction OU time constant [s]
TURBULENCE = {
    "none":   dict(tau_s=30.0, sigma_frac=0.15, base_frac=0.5, gust_rate=0.0,
                   gust_amp=(0.0, 0.0), gust_dur=(0.0, 0.0), dir_sigma=5.0, dir_tau=60.0),
    "low":    dict(tau_s=20.0, sigma_frac=0.20, base_frac=0.5, gust_rate=0.0,
                   gust_amp=(0.0, 0.0), gust_dur=(0.0, 0.0), dir_sigma=10.0, dir_tau=40.0),
    "medium": dict(tau_s=10.0, sigma_frac=0.20, base_frac=0.5, gust_rate=1 / 60,
                   gust_amp=(0.3, 0.6), gust_dur=(3.0, 8.0), dir_sigma=10.0, dir_tau=30.0),
    "high":   dict(tau_s=5.0, sigma_frac=0.20, base_frac=0.35, gust_rate=1 / 15,
                   gust_amp=(0.4, 0.9), gust_dur=(2.0, 6.0), dir_sigma=20.0, dir_tau=20.0),
}


class WindModel:
    def __init__(self, wind_params: dict, direction_deg: float, seed: int):
        self.vmin = float(wind_params.get("speed_min", 0.0))
        self.vmax = float(wind_params.get("speed_max", 0.0))
        self.turbulence = wind_params.get("turbulence", "none")
        self.p = TURBULENCE[self.turbulence]
        self.direction_deg = float(direction_deg)
        self.rng = random.Random(seed)
        rng_range = self.vmax - self.vmin
        self.base_mean = self.vmin + self.p["base_frac"] * rng_range
        self.base_sigma = self.p["sigma_frac"] * rng_range
        self.base = self.base_mean
        self.dir_dev = 0.0
        self.gusts = []      # (t_start, duration, amplitude)
        self.t = 0.0

    def _ou(self, x, mean, sigma, tau, dt):
        # exact discretisation of the OU process
        a = math.exp(-dt / tau)
        return mean + a * (x - mean) + sigma * math.sqrt(1 - a * a) * self.rng.gauss(0, 1)

    def step(self, dt: float):
        """Advance by dt seconds. Returns (speed, direction_deg, gust_speed)."""
        p = self.p
        self.t += dt
        self.base = self._ou(self.base, self.base_mean, self.base_sigma, p["tau_s"], dt)
        self.dir_dev = self._ou(self.dir_dev, 0.0, p["dir_sigma"], p["dir_tau"], dt)

        if p["gust_rate"] > 0 and self.rng.random() < p["gust_rate"] * dt:
            rng_range = self.vmax - self.vmin
            self.gusts.append((self.t,
                               self.rng.uniform(*p["gust_dur"]),
                               self.rng.uniform(*p["gust_amp"]) * rng_range))
        gust = 0.0
        alive = []
        for t0, dur, amp in self.gusts:
            tau = self.t - t0
            if tau <= dur:
                gust += 0.5 * amp * (1 - math.cos(2 * math.pi * tau / dur))
                alive.append((t0, dur, amp))
        self.gusts = alive

        speed = min(self.vmax, max(self.vmin, self.base + gust))
        return speed, (self.direction_deg + self.dir_dev) % 360.0, gust

    @staticmethod
    def to_enu(speed: float, direction_deg: float):
        """Wind blowing *from* direction_deg (0=N, 90=E) -> ENU air velocity."""
        theta = math.radians(direction_deg)
        return -speed * math.sin(theta), -speed * math.cos(theta)

    def describe(self) -> dict:
        return {"model": "ou+1cos_gusts", "turbulence": self.turbulence,
                "speed_min": self.vmin, "speed_max": self.vmax,
                "base_mean": self.base_mean, "base_sigma": self.base_sigma,
                "direction_deg": self.direction_deg, **self.p}
