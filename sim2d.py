"""Closed-loop world: a room, a cat that chases, a robot that drives on reflex() commands.
The camera image is ray-cast from the robot's pose each tick (walls + a cat photo sprite), so YOLO sees
what the robot would see, and the robot's moves change the next frame. Free space and the ultrasonic come
from the same ray cast. Outputs metrics and a top-down + camera video."""
from __future__ import annotations
import math, random, time, argparse, json, os
import numpy as np, cv2
import shapes
from shapes import FreeSpace, Mode, now
from reflex import reflex

W, H, HFOV = 640, 480, math.radians(62)
ROOM = (4.0, 3.0)                                  # metres
BOXES = [(1.2, 0.0, 2.0, 0.5), (0.0, 2.2, 0.8, 3.0), (3.0, 1.6, 3.6, 2.2)]   # sofa, shelf, chair (x0,y0,x1,y1)
CAT_H = 0.25                                        # cat height, m
V_MAX, WHEELBASE = 0.55, 0.15                       # m/s at speed 255, m
CAUGHT = 0.30

def segments():
    x1, y1 = ROOM; segs = [((0, 0), (x1, 0)), ((x1, 0), (x1, y1)), ((x1, y1), (0, y1)), ((0, y1), (0, 0))]
    for a, b, c, d in BOXES: segs += [((a, b), (c, b)), ((c, b), (c, d)), ((c, d), (a, d)), ((a, d), (a, b))]
    return segs
SEGS = segments()

def ray_hit(px, py, ang):
    """Distance to the nearest wall, which segment it hit, and how far along that segment (metres)."""
    dx, dy = math.cos(ang), math.sin(ang); best, seg, along = 99.0, -1, 0.0
    for i, ((ax, ay), (bx, by)) in enumerate(SEGS):
        ex, ey = bx - ax, by - ay; den = dx * ey - dy * ex
        if abs(den) < 1e-9: continue
        t = ((ax - px) * ey - (ay - py) * ex) / den; u = ((ax - px) * dy - (ay - py) * dx) / den
        if t > 1e-6 and 0 <= u <= 1 and t < best: best, seg, along = t, i, u * math.hypot(ex, ey)
    return best, seg, along

def ray(px, py, ang):
    return ray_hit(px, py, ang)[0]

# ---- procedural Doom-ish textures (drawn here, not copied from any game) ----
def _tex_brick(n=64, seed=3):
    r = np.random.default_rng(seed); t = np.zeros((n, n, 3), np.uint8)
    for row in range(4):
        off = 0 if row % 2 == 0 else n // 4
        for col in range(-1, 3):
            base = np.array([40, 60, 120]) + r.integers(-18, 18, 3)          # BGR: brick red-brown
            x0 = col * n // 2 + off; y0 = row * n // 4
            t[y0:y0 + n // 4, max(0, x0):max(0, min(n, x0 + n // 2))] = base
    t = np.clip(t.astype(int) + r.integers(-14, 14, (n, n, 1)), 0, 255).astype(np.uint8)
    t[::n // 4, :] = (55, 60, 62); t[:, ::1][:, [0]] = (55, 60, 62)            # mortar lines
    for row in range(4):
        off = 0 if row % 2 == 0 else n // 4
        for col in range(0, 3):
            x = (col * n // 2 + off) % n; t[row * n // 4:(row + 1) * n // 4, x:x + 2] = (55, 60, 62)
    return t

def _tex_tech(n=64, seed=5):
    r = np.random.default_rng(seed); t = np.full((n, n, 3), (92, 96, 100), np.uint8)
    t = np.clip(t.astype(int) + r.integers(-10, 10, (n, n, 1)), 0, 255).astype(np.uint8)
    t[:, :3] = t[:, -3:] = (50, 52, 55); t[:3, :] = t[-3:, :] = (50, 52, 55)
    t[n // 2 - 3:n // 2 + 3, 6:-6] = (40, 150, 60)                            # green light strip
    for x, y in ((8, 8), (n - 9, 8), (8, n - 9), (n - 9, n - 9)): t[y - 2:y + 2, x - 2:x + 2] = (160, 165, 170)
    return t

def _tex_floor(n=64, seed=7):
    r = np.random.default_rng(seed); t = np.full((n, n, 3), (58, 66, 72), np.uint8)
    t = np.clip(t.astype(int) + r.integers(-12, 12, (n, n, 1)), 0, 255).astype(np.uint8)
    t[:2, :] = t[:, :2] = (34, 38, 42); return t

def _tex_ceil(n=64, seed=9):
    r = np.random.default_rng(seed); t = np.full((n, n, 3), (38, 40, 44), np.uint8)
    t = np.clip(t.astype(int) + r.integers(-8, 8, (n, n, 1)), 0, 255).astype(np.uint8)
    t[n // 2 - 8:n // 2 + 8, n // 2 - 8:n // 2 + 8] = (200, 225, 235)         # ceiling light
    return t

TEX_BRICK, TEX_TECH, TEX_FLOOR, TEX_CEIL = _tex_brick(), _tex_tech(), _tex_floor(), _tex_ceil()
N_ROOM_SEGS = 4           # the first four segments are the room walls (brick); the rest are furniture (tech panels)
WALL_H, CAM_H = 0.6, 0.08
DOOM = True

def ray_with_cat(px, py, ang, cat, r=0.1):
    """Ultrasonic: nearest of the walls and the cat's body (a 10 cm circle)."""
    d = ray(px, py, ang); dx, dy = math.cos(ang), math.sin(ang); cx, cy = cat[0] - px, cat[1] - py
    b = cx * dx + cy * dy; c = cx * cx + cy * cy - r * r; disc = b * b - c
    if b > 0 and disc >= 0: d = min(d, max(0.0, b - math.sqrt(disc)))
    return d

def inside_box(x, y, m=0.0):
    if not (m <= x <= ROOM[0] - m and m <= y <= ROOM[1] - m): return True
    return any(a - m <= x <= c + m and b - m <= y <= d + m for a, b, c, d in BOXES)

class World:
    def __init__(self, seed=0, cat_speed=0.35, sprite="walking.jpg"):
        rnd = random.Random(seed); self.rnd = rnd
        while True:
            self.r = [rnd.uniform(0.3, 3.7), rnd.uniform(0.3, 2.7), rnd.uniform(-math.pi, math.pi)]
            self.c = [rnd.uniform(0.3, 3.7), rnd.uniform(0.3, 2.7)]
            if not inside_box(*self.r[:2], 0.15) and not inside_box(*self.c, 0.15) and math.dist(self.r[:2], self.c) > 1.8: break
        self.cat_speed = cat_speed
        img = cv2.imread(sprite); h, w = img.shape[:2]; self.sprite = img[int(h*0.05):, int(w*0.12):int(w*0.88)]
        self.pause = 0

    def respawn_cat(self, min_dist=1.8):
        """Drop the cat somewhere new, away from the robot; the robot stays where it is."""
        for _ in range(500):
            c = [self.rnd.uniform(0.3, 3.7), self.rnd.uniform(0.3, 2.7)]
            if not inside_box(*c, 0.15) and math.dist(self.r[:2], c) > min_dist:
                self.c = c; self.pause = 0; return True
        return False

    def render(self):
        x, y, th = self.r
        f = (W / 2) / math.tan(HFOV / 2)
        cols = np.arange(W); rel = np.arctan((cols - W / 2) / f); ang = th + rel
        img = np.zeros((H, W, 3), np.uint8)
        # floor and ceiling casting (vectorised): each row below/above the horizon is a fixed distance away
        rows = np.arange(H // 2 + 1, H); dfl = f * CAM_H / (rows - H / 2)                    # floor rows
        rc = np.arange(0, H // 2); dce = f * (WALL_H - CAM_H) / (H / 2 - rc)                 # ceiling rows
        ca, sa, cr = np.cos(ang), np.sin(ang), np.cos(rel)
        for dist, rr, tex, sc in ((dfl, rows, TEX_FLOOR, 0.25), (dce, rc, TEX_CEIL, 0.5)):
            dd = dist[:, None] / cr[None, :]
            wx = x + dd * ca[None, :]; wy = y + dd * sa[None, :]
            tu = ((wx / sc) % 1 * 63).astype(int); tv = ((wy / sc) % 1 * 63).astype(int)
            shade = np.clip(1.25 - dd * 0.28, 0.25, 1.0)[..., None]
            img[rr] = (tex[tv, tu] * shade).astype(np.uint8)
        # walls
        zbuf = np.zeros(W)
        for col in range(W):
            d, seg, along = ray_hit(x, y, ang[col]); d *= cr[col]; zbuf[col] = d
            tex = TEX_BRICK if seg < N_ROOM_SEGS else TEX_TECH
            hh = f * WALL_H / max(d, 0.05)
            top = H / 2 - hh * ((WALL_H - CAM_H) / WALL_H); bot = H / 2 + hh * (CAM_H / WALL_H)
            y0, y1 = int(max(0, top)), int(min(H, bot))
            if y1 <= y0: continue
            tu = int((along / WALL_H) % 1 * 63)
            tv = ((np.arange(y0, y1) - top) / (bot - top) * 63).astype(int).clip(0, 63)
            shade = max(0.3, 1.1 - d * 0.22)
            img[y0:y1, col] = (tex[tv, tu] * shade).astype(np.uint8)
        # cat sprite (a real cat photo, so YOLO still sees a cat)
        cx, cy = self.c; dx, dy = cx - x, cy - y; dist = math.hypot(dx, dy)
        r_ = math.atan2(dy, dx) - th; r_ = (r_ + math.pi) % (2 * math.pi) - math.pi
        if abs(r_) < HFOV / 2 + 0.3 and dist > 0.05:
            depth = dist * math.cos(r_); ph = int(f * CAT_H / depth); pw = int(ph * self.sprite.shape[1] / self.sprite.shape[0])
            u = int(W / 2 + f * math.tan(r_)); vb = int(H / 2 + f * CAM_H / depth)
            if ph > 6:
                spr = cv2.resize(self.sprite, (max(1, pw), max(1, ph)))
                for i in range(spr.shape[1]):
                    col = u - pw // 2 + i
                    if 0 <= col < W and depth < zbuf[col]:
                        y0 = vb - ph; a0, b0 = max(0, y0), min(H, vb)
                        if b0 > a0: img[a0:b0, col] = spr[a0 - y0:b0 - y0, i]
        return img

    def free_space(self):
        x, y, th = self.r
        thirds = []
        for lo, hi in ((-HFOV / 2, -HFOV / 6), (-HFOV / 6, HFOV / 6), (HFOV / 6, HFOV / 2)):
            ds = [min(ray(x, y, th + a), 3.0) for a in np.linspace(lo, hi, 7)]
            thirds.append(float(np.mean(ds)) / 3.0)
        # image left = negative angle offset? col<W/2 -> atan negative -> th - ... ; left of image is th - HFOV/2
        return FreeSpace(now(), thirds[0], thirds[1], thirds[2], ray_with_cat(x, y, th, self.c) * 100)

    def step(self, l, r, dt):
        x, y, th = self.r
        vl, vr = V_MAX * l / 255, V_MAX * r / 255
        v, w = (vl + vr) / 2, (vr - vl) / WHEELBASE
        # image convention: positive bearing = cat right of centre; in world, right of heading is -angle, so flip w
        th2 = th - w * dt
        nx, ny = x + v * math.cos(th2) * dt, y + v * math.sin(th2) * dt
        if not inside_box(nx, ny, 0.08): x, y = nx, ny
        elif not inside_box(nx, y, 0.08): x = x + (nx - x) * 0.5           # slide along the wall, slowed by friction
        elif not inside_box(x, ny, 0.08): y = y + (ny - y) * 0.5
        self.r = [x, y, (th2 + math.pi) % (2 * math.pi) - math.pi]
        # cat: chase with noise, sometimes sit
        if self.pause > 0: self.pause -= dt; return
        if self.rnd.random() < 0.01: self.pause = self.rnd.uniform(0.5, 2.0)
        cx, cy = self.c; a0 = math.atan2(y - cy, x - cx) + self.rnd.gauss(0, 0.35)
        for da in (0, 0.6, -0.6, 1.2, -1.2, 1.9, -1.9):            # slide around furniture instead of sticking
            a = a0 + da; nx, ny = cx + self.cat_speed * math.cos(a) * dt, cy + self.cat_speed * math.sin(a) * dt
            if not inside_box(nx, ny, 0.1):
                if math.dist((nx, ny), (x, y)) >= 0.15: self.c = [nx, ny]   # the robot has a body; the cat stops at it
                break

    def topdown(self, s=120):
        img = np.full((int(ROOM[1] * s), int(ROOM[0] * s), 3), 245, np.uint8)
        for a, b, c, d in BOXES: cv2.rectangle(img, (int(a * s), int(b * s)), (int(c * s), int(d * s)), (170, 170, 170), -1)
        x, y, th = self.r
        led = getattr(self, "led", (0, 0, 0))
        cv2.circle(img, (int(x * s), int(y * s)), 9, (0, 0, 255) if led[0] > 128 else (200, 90, 30), -1)
        cv2.line(img, (int(x * s), int(y * s)), (int((x + 0.25 * math.cos(th)) * s), int((y + 0.25 * math.sin(th)) * s)), (200, 90, 30), 2)
        for sgn in (-1, 1):
            a = th + sgn * HFOV / 2; cv2.line(img, (int(x * s), int(y * s)), (int((x + 0.6 * math.cos(a)) * s), int((y + 0.6 * math.sin(a)) * s)), (230, 180, 150), 1)
        cv2.circle(img, (int(self.c[0] * s), int(self.c[1] * s)), 9, (60, 60, 220), -1)
        return img

def run(policy="reflex", seconds=60, seed=0, video=None, dt=0.1):
    from perceive import Perceiver
    per = Perceiver() if policy == "reflex" else None
    simt = [0.0]; shapes._clock = lambda: simt[0]              # policy runs on simulated time
    import reflex as _R; _R._random.seed(seed * 7919 + int(os.environ.get("REP", "0")))   # reproducible wandering
    w = World(seed); mode = Mode(now(), "idle", 1e9)
    dists, caught, seen, cmds = [], 0.0, 0, []
    last = None
    from perceive import StallDetector
    stall, last_cmd = StallDetector(), (0, 0)
    vw = None
    for k in range(int(seconds / dt)):
        frame = w.render(); fs = w.free_space()
        if policy == "reflex":
            th = per.threat(frame); seen += th.present
            if th.present: last = th
            stuck = stall.update(frame, last_cmd[0] > 0 and last_cmd[1] > 0, simt[0]) and os.environ.get("CATBOT_STALL", "1") == "1"
            cmd = reflex(th, fs, mode, fs.ultrasonic_cm, last, stuck); l, r = cmd.l, cmd.r; why = cmd.why; last_cmd = (l, r)
        elif policy == "wander":
            l, r, why = 110, 110, "wander"
            if fs.ultrasonic_cm < 25: l, r, why = 150, -150, "wander: wall"
        else:
            l, r, why = 0, 0, "still"
        w.step(l, r, dt); simt[0] += dt
        d = math.dist(w.r[:2], w.c); dists.append(d); caught += dt * (d < CAUGHT)
        if video:
            if vw is None: vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 10, (W + 480, H))
            td = cv2.resize(w.topdown(), (480, 360)); pane = np.full((H, 480, 3), 30, np.uint8); pane[:360] = td
            cv2.putText(pane, why[:44], (8, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(pane, "t=%.1fs  cat %.2f m  caught %.1fs" % (k * dt, d, caught), (8, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            vw.write(np.hstack([frame, pane]))
    if vw: vw.release()
    d = np.array(dists)
    return {"policy": policy, "seed": seed, "seconds": seconds, "caught_s": round(caught, 1), "caught_pct": round(100 * caught / seconds, 1),
            "mean_dist_m": round(float(d.mean()), 2), "min_dist_m": round(float(d.min()), 2), "cat_seen_pct": round(100 * seen / len(d), 1) if policy == "reflex" else None}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--policy", default="reflex"); ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--video")
    a = ap.parse_args(); print(json.dumps(run(a.policy, a.seconds, a.seed, a.video)))
