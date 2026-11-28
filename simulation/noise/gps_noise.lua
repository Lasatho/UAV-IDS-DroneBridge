-- GPS position noise as first-order Gauss-Markov process per axis (north, east, altitude), plus the GNSS spoofing drift of
-- gnss_spoof.lua (additive, horizontal only), through SIM_GPS1_GLTCH_X/Y/Z (Lasse 2026-11-26, Vault sensorrauschen).
-- param:set does not save and sends no PARAM_VALUE, nothing appears on any MAVLink channel.
-- Noise: x[k+1] = a x[k] + SIGMA sqrt(1 - a^2) w, a = exp(-dt/TAU), stationary start, seeded from /dev/urandom so
-- every boot (the orchestrator reboots before each mission) gets its own realisation.
-- Spoofing: command file spooftrig/cmd "<rate m/s> <direction rad> <target offset m>" as in gnss_spoof.lua.
local PERIOD_MS = 200
local SIGMA = 0.5      -- m per horizontal axis (PX4 logs u-blox M8/M9/M10 at rest, median 0.4-0.6 m)
local SIGMA_Z = 1.0    -- m altitude (PX4 logs at rest, median 1.33 m, UAV-FD 0.24-0.51 m; Lasse 2026-11-26)
local TAU = 30.0       -- s (sensitivity levels 30 / 300 s)
local M_PER_DEG = 111320.0
local A = math.exp(-(PERIOD_MS / 1000.0) / TAU)
local B = SIGMA * math.sqrt(1.0 - A * A)
local BZ = SIGMA_Z * math.sqrt(1.0 - A * A)

local function seed()
  local f = io.open("/dev/urandom", "rb")
  local s = 0
  if f then
    local b = f:read(4)
    f:close()
    if b then for i = 1, #b do s = s * 256 + string.byte(b, i) end end
  end
  if s == 0 then s = millis():toint() end
  math.randomseed(s)
end

local function gauss()
  local u1 = math.random()
  if u1 < 1e-12 then u1 = 1e-12 end
  return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * math.random())
end

seed()
local nn, ne, nz = SIGMA * gauss(), SIGMA * gauss(), SIGMA_Z * gauss()
local off_n, off_e = 0.0, 0.0

local function read_cmd()
  local f = io.open("spooftrig/cmd", "r")
  if not f then return nil end
  local s = f:read("*l")
  f:close()
  if not s then return nil end
  local r, a, t = string.match(s, "^%s*([%d%.]+)%s+([%-%d%.]+)%s+([%d%.]+)")
  if not r then return nil end
  return tonumber(r), tonumber(a), tonumber(t)
end

local function coslat()
  local home = ahrs:get_home()
  local lat = home and home:lat() * 1e-7 or -35.3632
  return math.cos(math.rad(lat))
end

function update()
  nn = A * nn + B * gauss()
  ne = A * ne + B * gauss()
  nz = A * nz + BZ * gauss()
  local rate, ang, target = read_cmd()
  if rate then
    local d = math.sqrt(off_n * off_n + off_e * off_e)
    if d < target then
      local step = math.min(rate * PERIOD_MS / 1000.0, target - d)
      off_n = off_n + step * math.cos(ang)
      off_e = off_e + step * math.sin(ang)
    end
  else
    off_n, off_e = 0.0, 0.0
  end
  param:set('SIM_GPS1_GLTCH_X', (nn + off_n) / M_PER_DEG)
  param:set('SIM_GPS1_GLTCH_Y', (ne + off_e) / (M_PER_DEG * coslat()))
  param:set('SIM_GPS1_GLTCH_Z', nz)
  return update, PERIOD_MS
end

return update, PERIOD_MS
