"""Elegoo Smart Robot Car V4 (camera edition) link, derived from Elegoo's firmware source
(ESP32_CameraServer_AP_20210107 + SmartRobotCarV4.0_V0_20210104).

Join the car's open Wi-Fi "ELEGOO-xxxxxxxx" first.
  video    http://192.168.4.1:81/stream        (MJPEG, 800x600 default, one viewer at a time)
  control  tcp 192.168.4.1:100, JSON frames delimited by { }, forwarded to the UNO at 9600 baud
  heartbeat: the car prints {Heartbeat} every 1 s and drops us after 3 unanswered; we echo it
  N=3 continuous move: D1 1 left(spin) 2 right(spin) 3 forward 4 back, D2 0..255
  N=4 wheel speeds: D1 left 0..255, D2 right 0..255 (forward only in the stock firmware)
  N=100 stop.  N=21 D1=2 ultrasonic -> {H_<cm>} (clamped to 150; 0 = no echo).  N=5 servo.  N=8 lights.

First-hookup hardening (1 Oct 2026, from the firmware audit):
  * The ESP32->UNO link is 9600 baud, about 960 bytes/s. Sending a drive command and an ultrasonic query
    every frame at 20 Hz is ~1,160 bytes/s, more than the link carries, so commands queue and the car
    lags further and further behind. All writes now go through one budgeted sender: drive commands only
    when they change, plus a 10 Hz keepalive; ultrasonic polled in the background at 5 Hz.
  * ultrasonic_cm() never blocks; it returns the latest reading under 0.6 s old, else None.
    0 cm means no echo or a loose wire, so it is reported as None, not as "touching".
  * frame_age() lets the loop stop the car when the video has frozen.
  * Every byte in and out is logged to demo_logs/car_wire.log for first-hookup debugging.
"""
from __future__ import annotations
import json, socket, threading, time, re, os
import cv2

DIRS = {"left": 1, "right": 2, "forward": 3, "back": 4}
BAUD_BYTES_S = 960            # 9600 baud, 10 bits per byte
LINK_BUDGET = 0.6             # use at most 60% of it, leaving room for the UNO's replies and heartbeats
DRIVE_KEEPALIVE_S = 0.1       # resend an unchanged drive command at most 10 times a second
ULTRA_PERIOD_S = 0.2          # ultrasonic polled 5 times a second
ULTRA_FRESH_S = 0.6
MIN_PWM = 60                  # below ~70 the TT motors stall on a hard floor (Elegoo tutorial: 25% duty "can't start";
                              # owner measurement: stall under 70). Non-zero speeds are lifted to at least this.

class ElegooCar:
    def __init__(self, host="192.168.4.1", video_port=81, cmd_port=100, watchdog_s=0.5, wire_log="demo_logs/car_wire.log"):
        self.host, self.video_port, self.cmd_port, self.watchdog_s = host, video_port, cmd_port, watchdog_s
        self._frame, self._frame_t = None, 0.0
        self._lock, self._send_lock = threading.Lock(), threading.Lock()
        self._last_cmd = time.time(); self._moving = False
        self._last_drive, self._last_drive_t = None, 0.0
        self._ultra, self._ultra_t = None, 0.0
        self._sent_bytes, self._sent_t0 = 0, time.time()
        self.stats = {"sent": 0, "skipped": 0, "replies": 0, "ultra_ok": 0, "ultra_zero": 0, "video_reopens": 0}
        os.makedirs(os.path.dirname(wire_log) or ".", exist_ok=True)
        self._wire = open(wire_log, "a")
        self._sock = socket.create_connection((host, cmd_port), timeout=5)
        self._sock.settimeout(0.2)
        self._alive = True
        for fn in (self._pump, self._video, self._watchdog, self._ultra_poll):
            threading.Thread(target=fn, daemon=True).start()

    def _log(self, d, data):
        try: self._wire.write("%.3f %s %s\n" % (time.time(), d, data)); self._wire.flush()
        except Exception: pass

    # ---- video: a reader thread keeps only the newest frame; reopen if the stream stalls ----
    def _video(self):
        url = f"http://{self.host}:{self.video_port}/stream"
        while self._alive:
            cap = cv2.VideoCapture(url)
            last_ok = time.time()
            while self._alive:
                ok, f = cap.read()
                if ok:
                    with self._lock: self._frame, self._frame_t = f, time.time()
                    last_ok = time.time()
                elif time.time() - last_ok > 2.0:
                    break
                else:
                    time.sleep(0.05)
            cap.release(); self.stats["video_reopens"] += 1; time.sleep(0.5)

    def frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def frame_age(self) -> float:
        return time.time() - self._frame_t if self._frame_t else 1e9

    # ---- control ----
    def _send(self, obj: dict, essential: bool = False) -> bool:
        """Budgeted write. Non-essential writes are skipped when the 9600-baud link is already near capacity."""
        data = json.dumps(obj, separators=(",", ":")).encode() + b"\n"   # newline is ignored by the bridge; one owner needed it
        with self._send_lock:
            now = time.time()
            if now - self._sent_t0 > 1.0: self._sent_bytes, self._sent_t0 = 0, now
            if not essential and self._sent_bytes + len(data) > BAUD_BYTES_S * LINK_BUDGET:
                self.stats["skipped"] += 1; return False
            self._sock.sendall(data); self._sent_bytes += len(data); self.stats["sent"] += 1
        self._log(">", data.decode())
        return True

    def _pump(self):
        buf = b""
        while self._alive:
            try: chunk = self._sock.recv(256)
            except socket.timeout: continue
            except OSError: break
            if not chunk: break
            buf += chunk
            while b"}" in buf:
                frame, _, buf = buf.partition(b"}")
                frame = frame[frame.rfind(b"{"):] + b"}" if b"{" in frame else frame + b"}"
                if frame == b"{Heartbeat}":
                    try: self._sock.sendall(b"{Heartbeat}")
                    except OSError: pass
                    continue
                s = frame.decode(errors="ignore"); self._log("<", s); self.stats["replies"] += 1
                m = re.match(r"\{u_(\d+)\}", s)
                if m:
                    cm = int(m.group(1))
                    if cm > 0: self._ultra, self._ultra_t = float(cm), time.time(); self.stats["ultra_ok"] += 1
                    else: self.stats["ultra_zero"] += 1          # no echo / loose wire: not "touching"

    def _ultra_poll(self):
        while self._alive:
            try: self._send({"H": "u", "N": 21, "D1": 2})
            except OSError: break
            time.sleep(ULTRA_PERIOD_S)

    def _watchdog(self):
        while self._alive:
            time.sleep(0.1)
            if self._moving and time.time() - self._last_cmd > self.watchdog_s:
                self.stop()

    def move(self, direction: str, speed: int):
        self._drive_frame({"H": "m", "N": 3, "D1": DIRS[direction], "D2": int(max(MIN_PWM, min(255, speed)))})

    def _drive_frame(self, obj):
        now = time.time(); key = (obj["N"], obj.get("D1"), obj.get("D2"))
        changed = key != self._last_drive
        if changed or now - self._last_drive_t > DRIVE_KEEPALIVE_S:
            if self._send(obj, essential=changed):
                self._last_drive, self._last_drive_t = key, now
        self._last_cmd, self._moving = now, True

    def drive(self, l: int, r: int):
        """Two wheel speeds. Same sign forward: N=4. Both back: N=3 back. Opposite signs: N=3 spin."""
        lift = lambda v: 0 if v == 0 else (max(MIN_PWM, abs(v)) * (1 if v > 0 else -1))
        l, r = lift(int(max(-255, min(255, l)))), lift(int(max(-255, min(255, r))))
        if l == 0 and r == 0:
            return self.stop()
        # N=4 is D1 = RIGHT side, D2 = left on the real car (measured 2 Oct: "D1" spun the right wheels); N=3 spins are not mirrored
        if l >= 0 and r >= 0: self._drive_frame({"H": "d", "N": 4, "D1": r, "D2": l})
        elif l <= 0 and r <= 0: self.move("back", max(-l, -r))
        elif l < 0 < r: self.move("left", max(-l, r))
        else: self.move("right", max(l, -r))

    def stop(self):
        if self._last_drive != ("stop",):
            self._send({"H": "s", "N": 100}, essential=True); self._last_drive, self._last_drive_t = ("stop",), time.time()
        self._moving = False

    def ultrasonic_cm(self, timeout=None):
        """Latest reading if fresh, else None. Never blocks."""
        return self._ultra if self._ultra is not None and time.time() - self._ultra_t < ULTRA_FRESH_S else None

    def lights(self, r: int, g: int, b: int):
        self._send({"H": "l", "N": 8, "D1": 0, "D2": int(r), "D3": int(g), "D4": int(b)}, essential=True)

    def flash_red(self, seconds: float = 1.5):
        self.lights(255, 0, 0)
        threading.Timer(seconds, lambda: self.lights(0, 0, 0)).start()

    def camera_servo(self, angle: int):
        self._send({"H": "c", "N": 5, "D1": 1, "D2": int(max(0, min(180, angle)))}, essential=True)

    def close(self):
        try: self.stop()
        except OSError: pass
        self._alive = False
        self._sock.close()
