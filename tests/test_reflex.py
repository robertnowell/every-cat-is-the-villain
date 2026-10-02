import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shapes import Threat, FreeSpace, Mode, now
import reflex as R
from reflex import reflex

def setup_function(_): R._latch.update(dir=0, t=-1e9); R.clear_air_reset(); R._blk.update(since=None, dodge_until=0.0); R._wander.update(turn_until=0.0); R._unstick.update(until=0.0)

def T(present=True, bearing=0.0, prox=0.3, t=None): return Threat(now() if t is None else t, present, 1, bearing, prox, 0.8)
def F(l=0.5, c=0.5, r=0.5): return FreeSpace(now(), l, c, r)
IDLE = Mode(now(), "idle", 1e9)

def test_no_cat_wanders_forward():
    c = reflex(T(present=False), F(), IDLE, 200); assert c.l == c.r > 0 and "wandering" in c.why

def test_wall_ahead_turns_a_random_way_then_resumes():
    import shapes
    clock = [300.0]; old = shapes._clock; shapes._clock = lambda: clock[0]
    try:
        c = reflex(T(present=False, t=clock[0]), F(), IDLE, 25); assert c.l == -c.r and "turning" in c.why
        clock[0] += 1.1
        c = reflex(T(present=False, t=clock[0]), F(), IDLE, 200); assert c.l == c.r > 0
    finally:
        shapes._clock = old

def test_wall_turn_direction_is_random():
    import shapes
    dirs = set(); clock = [400.0]; old = shapes._clock; shapes._clock = lambda: clock[0]
    try:
        for _ in range(30):
            R._wander.update(turn_until=0.0); dirs.add(reflex(T(present=False, t=clock[0]), F(), IDLE, 25).l > 0); clock[0] += 5
    finally:
        shapes._clock = old
    assert dirs == {True, False}

def test_clear_air_turns_to_the_most_open_moment_then_drives():
    import shapes
    R.NO_CAT = "clear_air"
    clock = [100.0]; old = shapes._clock; shapes._clock = lambda: clock[0]
    try:
        full = 2 * 3.14159265 / R.SPIN_RATE
        t = 0.0
        while t <= full + 0.05:                                    # openness peaks one third of the way round
            c = reflex(T(present=False, t=clock[0]), F(c=0.9 if abs(t - full / 3) < 0.2 else 0.2), IDLE, 200)
            clock[0] += 0.05; t += 0.05
        assert "turning to clearest air" in c.why or "heading" in c.why
        while "turning" in c.why:
            c = reflex(T(present=False, t=clock[0]), F(c=0.9), IDLE, 200); clock[0] += 0.05
        assert c.l == c.r and c.l > 0 and "clear air" in c.why
        spun = clock[0] - 100.0 - full
        assert abs(spun - full / 3) < 0.25, spun
    finally:
        shapes._clock = old; R.NO_CAT = "wander"

def test_cat_right_spins_left():
    c = reflex(T(bearing=+20), F(), IDLE, None); assert c.l < 0 < c.r

def test_cat_left_spins_right():
    c = reflex(T(bearing=-20), F(), IDLE, None); assert c.r < 0 < c.l

def test_never_drives_toward_a_visible_cat():
    for b in (-30, -10, 0, 10, 30):
        for p in (0.1, 0.3, 0.6):
            c = reflex(T(bearing=b, prox=p), F(), IDLE, None)
            assert not (c.l > 0 and c.r > 0), (b, p, c)

def test_cat_close_reverses_and_turns():
    c = reflex(T(bearing=5, prox=0.6), F(), IDLE, None); assert c.l < 0 and c.r < 0 and c.l != c.r

def test_just_lost_cat_keeps_turning_away():
    c = reflex(T(present=False), F(), IDLE, None, T(bearing=20, t=now() - 0.1)); assert c.l < 0 < c.r

def test_then_runs_forward():
    c = reflex(T(present=False), F(), IDLE, None, T(bearing=20, t=now() - 1.0)); assert c.l > 0 and c.r > 0

def test_memory_expires():
    c = reflex(T(present=False), F(), IDLE, 200, T(bearing=20, t=now() - 5)); assert "wander" in c.why

def test_blocked_overrides_everything():
    c = reflex(T(bearing=3, prox=0.6), F(c=0.2), IDLE, 10); assert c.l < 0 and c.r < 0 and "blocked" in c.why

def test_escape_corner_spins_to_open_side():
    c = reflex(T(), F(l=0.8, r=0.3), Mode(now(), "escape_corner", 5), None); assert c.l == -c.r and c.l < 0

def test_supervisor_idle_stops():
    c = reflex(T(present=False), F(), Mode(now(), "idle", 5), None); assert (c.l, c.r) == (0, 0)

def test_spin_direction_does_not_flip_on_jitter():
    a = reflex(T(present=False), F(l=0.6, c=0.1, r=0.5), IDLE, 10)
    b = reflex(T(present=False), F(l=0.5, c=0.1, r=0.6), IDLE, 10)
    assert (a.l > 0) == (b.l > 0), (a, b)

def test_cat_too_close_to_see_backs_away():
    c = reflex(T(present=False), F(), IDLE, 12, T(bearing=5, prox=0.8, t=now() - 1.0)); assert c.l < 0 and c.r < 0 and "too close" in c.why

def test_near_obstacle_without_recent_cat_backs_off():
    c = reflex(T(present=False), F(), IDLE, 12, T(bearing=5, t=now() - 10)); assert c.l < 0 and c.r < 0 and "blocked" in c.why

def test_trapped_dodges_after_backing_fails():
    import shapes
    clock = [200.0]; old = shapes._clock; shapes._clock = lambda: clock[0]
    try:
        whys = []
        for _ in range(40):
            whys.append(reflex(T(present=False, t=clock[0]), F(l=0.3, r=0.8), IDLE, 8).why); clock[0] += 0.05
        assert whys[0].startswith("blocked") and any("trapped: spin" in w for w in whys) and any("break out" in w for w in whys)
    finally:
        shapes._clock = old

def test_stalled_backs_up_then_turns():
    import shapes
    clock = [500.0]; old = shapes._clock; shapes._clock = lambda: clock[0]
    try:
        c = reflex(T(present=False, t=clock[0]), F(), IDLE, 200, None, stalled=True); assert c.l < 0 and c.r < 0 and "stuck" in c.why
        clock[0] += 0.6
        c = reflex(T(present=False, t=clock[0]), F(), IDLE, 200); assert c.l == -c.r and "stuck: turning" in c.why
        clock[0] += 1.0
        c = reflex(T(present=False, t=clock[0]), F(), IDLE, 200); assert "wandering" in c.why
    finally:
        shapes._clock = old

def test_stall_detector_fires_only_when_driving_and_still():
    import numpy as np
    from perceive import StallDetector
    d = StallDetector(); f = np.zeros((480, 640, 3), np.uint8)
    assert not any(d.update(f, False, t / 10) for t in range(30))           # parked and still: fine
    assert any(d.update(f, True, 10 + t / 10) for t in range(30))           # driving and still: stuck
