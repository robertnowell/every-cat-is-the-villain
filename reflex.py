"""The fast policy. Pure: (Threat, FreeSpace, Mode, ultrasonic, last seen Threat) -> Command. Shape this one.

The camera only looks forward, so a visible cat is always somewhere ahead. Fleeing is therefore:
  cat visible        -> spin away from it (reverse while spinning if it is close), until it leaves the frame
  cat just left view -> it is now behind us: drive forward hard, toward the more open side, for MEMORY_S
  nothing seen       -> scan: turn slowly to look around
"""
from __future__ import annotations
from shapes import Threat, FreeSpace, Command, Mode, now

CRUISE, FLEE_MAX, SCAN, SPIN = 110, 255, 40, 200
TURN_S = 0.35           # keep spinning this long after the cat leaves the frame, so it ends up behind us
                        # (sim: SPIN 200 turns ~5.7 rad/s, so 0.35 s is ~115 deg; CALIBRATE on the real car Tuesday)
MEMORY_S = 2.5          # then run until this long after the cat was last seen
NEAR_CM, NEAR_S = 25, 3.0   # something this close on the ultrasonic, with a cat seen this recently, IS the cat
                            # (under ~30 cm the camera sees only legs and YOLO loses it)
CLOSE = 0.45            # proximity (box height / frame height) that counts as close

COMMIT_S = 0.8          # once a spin direction is chosen, keep it this long (depth estimates jitter frame to frame)
_latch = {"dir": 0, "t": -1e9}   # the one piece of state in the policy: hysteresis on the turn direction

# ---- clear-air seeking: spin once, remember how open each moment of the turn looked, face the best, drive ----
LOOK_SPEED = 120        # wheel speed while spinning to look
SPIN_RATE = 3.45        # rad/s at LOOK_SPEED. Sim value; CALIBRATE on the real car Tuesday (time one full turn)
GO_MAX_S = 3.0          # longest single drive toward open space before looking again
GO_STOP_CM = 50         # stop driving when the ultrasonic reads less than this
_ca = {"phase": "spin", "t0": None, "samples": [], "turn_s": 0.0, "until": 0.0}

TRAPPED_S, DODGE_SPIN_S, DODGE_GO_S = 1.0, 0.45, 0.8
_blk = {"since": None, "dodge_until": 0.0, "dir": 1}

# ---- wandering: the no-cat behaviour. Drive; if a wall is close, turn a random way for a random moment ----
import random as _random
WANDER_SPEED, WANDER_WALL_CM = 70, 35      # patrol at half the old 130 (70 is the slowest that moves it), so the escape looks fast
WANDER_TURN = 100                          # patrol turns, half the escape spin
_wander = {"turn_until": 0.0, "dir": 1}
NO_CAT = "wander"        # "wander" (default) or "clear_air" (spin-and-look)

def wander(free: FreeSpace, ultra_cm, t: float) -> Command:
    if t < _wander["turn_until"]:
        d = _wander["dir"]; return Command(t, WANDER_TURN * d, -WANDER_TURN * d, "wandering: turning %s" % ("right" if d > 0 else "left"))
    if (ultra_cm is not None and ultra_cm < WANDER_WALL_CM) or free.center < 0.3:
        _wander.update(dir=_random.choice((-1, 1)), turn_until=t + _random.uniform(0.3, 1.0))
        d = _wander["dir"]; return Command(t, SPIN * d, -SPIN * d, "wall ahead: turning %s" % ("right" if d > 0 else "left"))
    return Command(t, WANDER_SPEED, WANDER_SPEED, "wandering")

# ---- unstick: back up, then turn a random way ----
UNSTICK_BACK_S = 0.5
_unstick = {"until": 0.0, "turn_from": 0.0, "dir": 1}

def start_unstick(t: float):
    _unstick.update(dir=_random.choice((-1, 1)), turn_from=t + UNSTICK_BACK_S, until=t + UNSTICK_BACK_S + _random.uniform(0.4, 0.9))

def clear_air_reset():
    _ca.update(phase="spin", t0=None, samples=[], turn_s=0.0, until=0.0)

def clear_air(free: FreeSpace, ultra_cm, t: float) -> Command:
    full = 2 * 3.14159265 / SPIN_RATE
    if _ca["phase"] == "spin":
        if _ca["t0"] is None: _ca["t0"] = t; _ca["samples"] = []
        _ca["samples"].append((t - _ca["t0"], free.center))
        if t - _ca["t0"] < full:
            return Command(t, LOOK_SPEED, -LOOK_SPEED, "looking around (%d%%)" % int(100 * (t - _ca["t0"]) / full))
        smp = _ca["samples"]                                   # smooth, pick the most open moment of the turn
        best, best_v = 0.0, -1.0
        for i, (dt, _) in enumerate(smp):
            win = [v for (d2, v) in smp if abs(d2 - dt) < 0.15]
            v = sum(win) / len(win)
            if v > best_v: best, best_v = dt, v
        _ca.update(phase="turn", turn_s=best, until=t + best, best_v=best_v)
    if _ca["phase"] == "turn":
        if t < _ca["until"]:
            return Command(t, LOOK_SPEED, -LOOK_SPEED, "turning to clearest air (%.0f deg)" % (57.3 * SPIN_RATE * _ca["turn_s"]))
        _ca.update(phase="go", until=t + GO_MAX_S)
    if _ca["phase"] == "go":
        blocked = (ultra_cm is not None and ultra_cm < GO_STOP_CM) or free.center < 0.3
        if t < _ca["until"] and not blocked:
            return Command(t, CRUISE, CRUISE, "heading for clear air")
        clear_air_reset()
        return Command(t, 0, 0, "arrived: look again")
    clear_air_reset()
    return Command(t, 0, 0, "look again")

def _open_side(free: FreeSpace) -> int:
    t = now()
    if t - _latch["t"] < COMMIT_S and _latch["dir"]:
        return _latch["dir"]
    d = +1 if free.right >= free.left else -1
    _latch.update(dir=d, t=t)
    return d

def reflex(threat: Threat, free: FreeSpace, mode: Mode, ultra_cm: float | None, last: Threat | None = None, stalled: bool = False) -> Command:
    t = now()
    if stalled: start_unstick(t)
    if t < _unstick["until"]:                                     # stuck: back up, then turn a random way
        if t < _unstick["turn_from"]: return Command(t, -FLEE_MAX, -FLEE_MAX, "stuck: backing up")
        d = _unstick["dir"]; return Command(t, SPIN * d, -SPIN * d, "stuck: turning %s" % ("right" if d > 0 else "left"))
    if threat.present or (last is not None and t - last.t < MEMORY_S):
        clear_air_reset()                                        # any cat activity restarts the look-around afterwards
    # 0a. cat too close to see: the ultrasonic is hitting it; back straight away
    if not threat.present and ultra_cm is not None and ultra_cm < NEAR_CM and last is not None and t - last.t < NEAR_S:
        return Command(t, -FLEE_MAX, -FLEE_MAX, "cat too close to see (%.0f cm): back away" % ultra_cm)
    # 0b. something under 15 cm ahead: back off, swinging toward the open side. Reversing opens the gap from
    #     whatever it is; if it is a cat that followed us (too close for the camera), the gap lets the camera find it.
    #     If backing off has not worked within TRAPPED_S (we are backed into something), dodge: spin to the open
    #     side and drive out past it.
    if _blk["dodge_until"] > t:
        if t < _blk["dodge_until"] - DODGE_GO_S:
            d = _blk["dir"]; return Command(t, SPIN * d, -SPIN * d, "trapped: spin %s" % ("right" if d > 0 else "left"))
        return Command(t, FLEE_MAX, FLEE_MAX, "trapped: break out")
    if ultra_cm is not None and ultra_cm < 15:
        if _blk["since"] is None: _blk["since"] = t
        if t - _blk["since"] > TRAPPED_S:
            _blk.update(since=None, dir=_open_side(free), dodge_until=t + DODGE_SPIN_S + DODGE_GO_S)
            return Command(t, 0, 0, "trapped: dodge")
        s = _open_side(free)
        l, r = (-FLEE_MAX, -int(FLEE_MAX * 0.4)) if s > 0 else (-int(FLEE_MAX * 0.4), -FLEE_MAX)
        return Command(t, l, r, "blocked %.0f cm: back off toward %s" % (ultra_cm, "right" if s > 0 else "left"))
    _blk["since"] = None
    # 1. supervisor says cornered: spin toward the open side
    if mode.mode == "escape_corner" and mode.live():
        s = _open_side(free)
        return Command(t, SPIN * s, -SPIN * s, "escape_corner: spin %s" % ("right" if s > 0 else "left"))
    # 2. cat in view: turn away from it
    if threat.present:
        away = -1 if threat.bearing >= 0 else +1                       # cat right -> spin left
        if threat.proximity > CLOSE:
            fast, slow = -FLEE_MAX, -int(FLEE_MAX * 0.2)               # reverse, swinging the nose away
            l, r = (slow, fast) if away > 0 else (fast, slow)
            return Command(t, l, r, "cat close %+.0f deg (%.2f): reverse and turn %s" % (threat.bearing, threat.proximity, "right" if away > 0 else "left"))
        return Command(t, SPIN * away, -SPIN * away, "cat %+.0f deg (%.2f): spin %s" % (threat.bearing, threat.proximity, "right" if away > 0 else "left"))
    # 3a. cat just left the frame: keep turning the same way until it is behind us
    if last is not None and t - last.t < TURN_S:
        away = -1 if last.bearing >= 0 else +1
        return Command(t, SPIN * away, -SPIN * away, "cat left the frame: keep turning %s" % ("right" if away > 0 else "left"))
    # 3b. it is behind us now: run
    if last is not None and t - last.t < MEMORY_S:
        s = _open_side(free)
        if ultra_cm is not None and ultra_cm < 45:                     # wall coming: bend hard toward open space now
            l, r = (FLEE_MAX, 0) if s > 0 else (0, FLEE_MAX)
            return Command(t, l, r, "cat behind, wall %.0f cm: veer %s" % (ultra_cm, "right" if s > 0 else "left"))
        if free.center >= max(free.left, free.right) - 0.1:
            return Command(t, FLEE_MAX, FLEE_MAX, "cat behind: run straight")
        inner = int(FLEE_MAX * 0.55)
        l, r = (FLEE_MAX, inner) if s > 0 else (inner, FLEE_MAX)
        return Command(t, l, r, "cat behind: run, bearing %s" % ("right" if s > 0 else "left"))
    # 4. nothing seen
    if mode.mode == "idle" and mode.live() and mode.ttl_s < 1e8:
        return Command(t, 0, 0, "idle (supervisor)")
    if NO_CAT == "clear_air":
        return clear_air(free, ultra_cm, t)
    return wander(free, ultra_cm, t)
