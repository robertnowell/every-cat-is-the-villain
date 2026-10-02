"""A stand-in for the Elegoo V4 camera car that speaks its firmware protocol exactly (from Elegoo's source):
MJPEG on :81/stream, JSON over TCP :100 with the 1 s {Heartbeat} that drops a client after 3 misses,
N=3 / N=4 / N=100 / N=21, motors hold their last command, and stop when the client leaves.
Behind it is the sim2d room, stepped in real time. car.py and main.py talk to it unchanged via --host 127.0.0.1."""
from __future__ import annotations
import socket, threading, time, json, math, argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
from sim2d import World, ray

class Fake:
    def __init__(self, seed=0, video_port=81, cmd_port=100, fw="v0"):
        self.fw = fw
        self.w = World(seed); self.lr = (0, 0); self.lock = threading.Lock()
        self.stats = {"frames_served": 0, "commands": 0, "heartbeats_echoed": 0, "drops": 0, "stops": 0, "ultrasonic": 0, "bad_json": 0, "lights": 0, "uart_bytes": 0, "max_backlog_ms": 0, "cmd_latency_ms_p95": 0}
        self.dists = []; self.t0 = time.time()
        self.video_port, self.cmd_port = video_port, cmd_port

    # physics + frames
    def run_world(self):
        last = time.time()
        while True:
            time.sleep(0.03); t = time.time(); dt = t - last; last = t
            with self.lock:
                self.w.step(*self.lr, dt); self.dists.append((t, math.dist(self.w.r[:2], self.w.c)))
                self.jpg = cv2.imencode(".jpg", self.w.render(), [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
                self.map_jpg = cv2.imencode(".jpg", cv2.resize(self.w.topdown(), (480, 360)), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

    def serve_video(self):
        fake = self
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                if self.path.startswith("/reset"):                   # new room / new chase, from the dashboard button
                    q = dict(kv.split("=") for kv in self.path.partition("?")[2].split("&") if "=" in kv)
                    with fake.lock:
                        if "seed" in q: fake.w = World(int(q["seed"])); fake.lr = (0, 0)      # whole new room
                        else: fake.w.respawn_cat()                                            # new cat, robot stays put
                        fake.dists = []
                    self.send_response(200); self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers(); self.wfile.write(b"ok"); return
                if self.path not in ("/stream", "/map"): self.send_response(404); self.end_headers(); return
                is_map = self.path == "/map"
                self.send_response(200); self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Type", "multipart/x-mixed-replace;boundary=123456789000000000000987654321"); self.end_headers()
                try:
                    while True:
                        jpg = getattr(fake, "map_jpg" if is_map else "jpg", None)
                        if jpg:
                            self.wfile.write(b"--123456789000000000000987654321\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg) + jpg + b"\r\n")
                            if not is_map: fake.stats["frames_served"] += 1
                        time.sleep(1 / 15)                     # ESP32-CAM class frame rate
                except (BrokenPipeError, ConnectionResetError): pass
        ThreadingHTTPServer(("127.0.0.1", self.video_port), H).serve_forever()

    UART_BYTES_S = 960      # 9600 baud ESP32 -> UNO, the real bottleneck

    def _uart(self, conn):
        """Frames reach the UNO only as fast as 9600 baud allows; the UNO acts on each when it has fully arrived."""
        import collections
        q = self.uart_q; lat = []
        while self.uart_alive:
            if not q: time.sleep(0.002); continue
            t_in, frame = q[0]
            time.sleep(len(frame) / self.UART_BYTES_S)          # wire time for this frame
            q.popleft(); self.stats["uart_bytes"] += len(frame)
            lat.append(1000 * (time.time() - t_in))
            self.stats["max_backlog_ms"] = max(self.stats["max_backlog_ms"], int(lat[-1]))
            if len(lat) >= 20: self.stats["cmd_latency_ms_p95"] = int(sorted(lat[-200:])[int(0.95 * len(lat[-200:])) - 1])
            self.apply(conn, frame)

    def apply(self, conn, frame):
        try: d = json.loads(frame)
        except ValueError: self.stats["bad_json"] += 1; return
        self.stats["commands"] += 1; n = d.get("N"); h = d.get("H", "")
        with self.lock:
            if n == 4: self.lr = (int(d["D2"]), int(d["D1"]))   # real car: D1 drives the right side
            elif n == 3:
                sp = int(d.get("D2", 0)); self.lr = {1: (-sp, sp), 2: (sp, -sp), 3: (sp, sp), 4: (-sp, -sp)}.get(int(d["D1"]), (0, 0))
            elif n == 100: self.lr = (0, 0); self.stats["stops"] += 1
            elif n == 8: self.stats["lights"] += 1; self.w.led = (int(d.get("D2", 0)), int(d.get("D3", 0)), int(d.get("D4", 0)))
            elif n == 21:
                from sim2d import ray_with_cat
                x, y, th = self.w.r; cm = int(min(ray_with_cat(x, y, th, self.w.c), 4.5) * 100); self.stats["ultrasonic"] += 1
                if self.fw == "v2" and math.dist((x, y), self.w.c) < 0.35 and ray_with_cat(x, y, th, self.w.c) < ray(x, y, th) - 0.01:
                    cm = 150                                       # v2.1.2: fur absorbs the ping -> timeout -> reports 150 "clear"
                cm = 150 if cm > 150 else cm                       # both firmwares clamp to 150
                try: conn.sendall(("{%s_%d}" % (h, cm)).encode())
                except OSError: pass

    def handle(self, conn):
        import collections
        buf, hb_ok, hb_miss, last_hb = "", True, 0, time.time()
        self.uart_q = collections.deque(); self.uart_alive = True
        threading.Thread(target=self._uart, args=(conn,), daemon=True).start()
        conn.settimeout(0.05)
        while True:
            if time.time() - last_hb > 1.0:                    # firmware: print {Heartbeat} each second, count misses
                last_hb = time.time()
                try: conn.sendall(b"{Heartbeat}")
                except OSError: break
                if hb_ok: hb_ok, hb_miss = False, 0
                else: hb_miss += 1
                if hb_miss > 3: self.stats["drops"] += 1; break
            try: data = conn.recv(256)
            except socket.timeout: continue
            except OSError: break
            if not data: break
            for ch in data.decode(errors="ignore"):
                if ch == "{": buf = ""
                if ch != " ": buf += ch
                if ch == "}" and buf.startswith("{"):
                    frame, buf = buf, ""
                    if frame == "{Heartbeat}": hb_ok = True; self.stats["heartbeats_echoed"] += 1; continue
                    self.uart_q.append((time.time(), frame))             # to the UNO, at 9600 baud
        self.uart_alive = False
        with self.lock: self.lr = (0, 0)                        # firmware sends {"N":100} when the client leaves
        conn.close()

    def serve_cmd(self):
        s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", self.cmd_port)); s.listen(1)
        while True:
            c, _ = s.accept(); self.handle(c)                 # one client at a time, like the firmware

    def start(self):
        for f in (self.run_world, self.serve_video, self.serve_cmd): threading.Thread(target=f, daemon=True).start()
        return self

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=-1, help="-1 = random start each launch"); ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--topdown", help="write a top-down video of the room here")
    ap.add_argument("--fw", default="v0", choices=["v0", "v2"], help="ultrasonic behaviour: v0 (2021, no echo=0) or v2 (2.1.2, no echo=150)")
    ap.add_argument("--video-port", type=int, default=81); ap.add_argument("--cmd-port", type=int, default=100)
    a = ap.parse_args()
    if a.seed < 0: a.seed = int(time.time() * 1000) % 100000
    print("room seed", a.seed, flush=True)
    f = Fake(a.seed, a.video_port, a.cmd_port, a.fw).start()
    if a.topdown:
        def rec():
            vw = cv2.VideoWriter(a.topdown, cv2.VideoWriter_fourcc(*"mp4v"), 10, (480, 360)); t_end = time.time() + a.seconds; stamps = []
            while time.time() < t_end:
                with f.lock: img = cv2.resize(f.w.topdown(), (480, 360))
                vw.write(img); stamps.append(time.time()); time.sleep(0.1)
            vw.release(); json.dump(stamps, open(a.topdown + ".json", "w"))
        recorder = threading.Thread(target=rec, daemon=True); recorder.start()
    time.sleep(a.seconds if a.seconds > 0 else 10 ** 9)
    if a.topdown: recorder.join(timeout=5)          # let the writer close the file
    d = [x for _, x in f.dists]
    print(json.dumps({**f.stats, "caught_pct": round(100 * sum(1 for x in d if x < 0.3) / max(1, len(d)), 1), "mean_dist_m": round(sum(d) / max(1, len(d)), 2)}))
