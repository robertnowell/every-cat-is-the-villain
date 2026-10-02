"""First-hookup preflight for the real Elegoo car. Run it once, in order, before the demo loop.

    .venv/bin/python preflight.py            # interactive: it tells you what to do and asks what you saw
    .venv/bin/python preflight.py --yes      # non-interactive smoke run (used against fake_elegoo.py)

Writes demo_logs/preflight.json and prints the constants to change. Each step stops with a plain reason on failure.
"""
from __future__ import annotations
import argparse, json, os, re, socket, subprocess, sys, time
import cv2

R = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}
ap = argparse.ArgumentParser()
ap.add_argument("--host", default="192.168.4.1"); ap.add_argument("--video-port", type=int, default=81); ap.add_argument("--cmd-port", type=int, default=100)
ap.add_argument("--yes", action="store_true"); ap.add_argument("--skip-floor", action="store_true")
A = ap.parse_args()

def say(s): print("\n\033[1m" + s + "\033[0m", flush=True)
def ask(q, default="y"):
    if A.yes: print(q, "[auto:", default + "]"); return default
    return (input(q + " ") or default).strip().lower()
def ok(step, good, **kv):
    R["steps"][step] = {"ok": bool(good), **kv}
    print(("  PASS " if good else "  FAIL ") + step, kv if kv else "", flush=True)
    json.dump(R, open("demo_logs/preflight.json", "w"), indent=1)
    return good
def sh(cmd):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout
    except Exception as e: return str(e)
os.makedirs("demo_logs", exist_ok=True)

# 1. network
say("1/9 Network: Mac on the car's Wi-Fi, internet through the iPhone")
if A.host == "192.168.4.1":
    ssid = sh("networksetup -getairportnetwork en0") + sh("ipconfig getsummary en0 | grep ' SSID'")
    # macOS 15+ redacts the SSID without location permission, so a car that answers ping counts too
    on_car = "ELEGOO" in ssid or os.system("ping -c1 -t2 %s >/dev/null 2>&1" % A.host) == 0
    ok("wifi on ELEGOO network", on_car, ssid=ssid.strip()[:80])
    if not on_car: print("  -> System Settings > Wi-Fi > join ELEGOO-xxxxxxxx. Close the Elegoo phone app: one client only.")
    rt = sh("route -n get 192.168.4.1"); iface = re.search(r"interface: (\S+)", rt)
    ok("192.168.4.1 routes over Wi-Fi", iface and iface.group(1) == "en0", interface=iface.group(1) if iface else None)
    rt2 = sh("route -n get 1.1.1.1"); iface2 = re.search(r"interface: (\S+)", rt2)
    net = sh("curl -s -o /dev/null -w '%{http_code}' --max-time 5 https://integrate.api.nvidia.com/v1/models")
    ok("internet for cloud calls", net.strip() in ("200", "401"), default_interface=iface2.group(1) if iface2 else None, http=net.strip())
    if net.strip() not in ("200", "401"): print("  -> plug in the iPhone, Personal Hotspot on, tap Trust; set service order: sudo networksetup -ordernetworkservices \"iPhone USB\" \"Wi-Fi\"")
else:
    ok("network (local test host, skipped)", True)

# 2. video
say("2/9 Video: stream opens, frame rate, orientation")
cap = cv2.VideoCapture(f"http://{A.host}:{A.video_port}/stream")
frames, t0, first = 0, time.time(), None
while time.time() - t0 < 5:
    good, f = cap.read()
    if good:
        frames += 1
        if frames == 6: first = f                      # the first few OV2640 frames are dark and green; skip them
cap.release()
fps = frames / 5
if first is not None: cv2.imwrite("demo_logs/preflight_frame.jpg", first)
ok("video stream", frames > 10, fps=round(fps, 1), size=None if first is None else list(first.shape[:2]))
if frames <= 10: print("  -> open http://%s:%d/stream in Safari. Blank: power-cycle the car; one viewer only, close other tabs/app." % (A.host, A.video_port)); sys.exit(1)
if fps < 8: print("  -> slow stream: GET http://%s/control?var=framesize&val=8 for 640x480 (VGA measured ~14 fps)" % A.host)
up = ask("  Look at demo_logs/preflight_frame.jpg: floor at the bottom, not mirrored? [y/n]")
ok("image orientation", up.startswith("y"))
if not up.startswith("y"): print("  -> flip in perceive or set CAMERA_FLIP=1 (cv2.flip); bearings depend on it")

# 3. control link and heartbeat
say("3/9 Control link: TCP 100 and the 1 s heartbeat")
s = socket.create_connection((A.host, A.cmd_port), timeout=5); s.settimeout(3)
try: hb = s.recv(64)
except socket.timeout: hb = b""
ok("heartbeat received", b"{Heartbeat}" in hb, raw=hb.decode(errors="ignore")[:40])
if b"{Heartbeat}" in hb: s.sendall(b"{Heartbeat}")
# The ESP32 drops a client that misses ~3 heartbeats, and the prompts below wait on a human, so a background
# reader echoes every heartbeat and buffers everything else for drain().
import threading
_lock, _buf = threading.Lock(), bytearray()
def _reader():
    s.settimeout(None)
    while True:
        try: d = s.recv(256)
        except OSError: return
        if not d: return
        if b"{Heartbeat}" in d:
            with _lock: s.sendall(b"{Heartbeat}")
        with _lock: _buf.extend(d)
threading.Thread(target=_reader, daemon=True).start()
def send(obj):
    with _lock: s.sendall(json.dumps(obj, separators=(",", ":")).encode() + b"\n")
def drain(t=0.3):
    time.sleep(t)
    with _lock: out = bytes(_buf); _buf.clear()
    return out.decode(errors="ignore")

# 4. ultrasonic and firmware variant
say("4/9 Ultrasonic: replies, units, and what 'no echo' reads as")
vals = []
for i in range(10):
    send({"H": "u", "N": 21, "D1": 2}); r = drain(0.25)
    m = re.findall(r"\{u_(\d+)\}", r); vals += [int(x) for x in m]
ok("ultrasonic replies", len(vals) >= 6, readings=vals)
if not vals: print("  -> no {u_<cm>} replies: is the shield switch on 'Cam' (not 'Upload')? Log in demo_logs/car_wire.log"); 
if not A.yes:
    input("  Point the nose at open space (over 2 m away) or cover the sensor with a towel, then press Enter")
    far = []
    for i in range(6): send({"H": "u", "N": 21, "D1": 2}); far += [int(x) for x in re.findall(r"\{u_(\d+)\}", drain(0.25))]
    variant = "v0 (no echo = 0)" if far and max(far) == 0 else "v2.1.2 (no echo = 150)" if far and min(far) >= 145 else "unclear"
    ok("ultrasonic no-echo behaviour", True, readings=far, firmware_guess=variant)
    if "150" in variant: print("  -> fur/plush may read 150 'clear' at close range: rely on the camera for the cat, the sensor for walls")

# 5. wheels up
say("5/9 Wheels up: lift the car so the wheels spin freely")
ask("  Wheels off the ground? Press Enter")
tests = [("left wheels forward", {"H": "d", "N": 4, "D1": 150, "D2": 0}), ("right wheels forward", {"H": "d", "N": 4, "D1": 0, "D2": 150}),
         ("both forward", {"H": "d", "N": 4, "D1": 150, "D2": 150}), ("both backward", {"H": "m", "N": 3, "D1": 4, "D2": 150}),
         ("spin left (left back, right forward)", {"H": "m", "N": 3, "D1": 1, "D2": 150}), ("spin right", {"H": "m", "N": 3, "D1": 2, "D2": 150})]
bad = []
for name, cmd in tests:
    send(cmd); time.sleep(1.0); send({"H": "s", "N": 100}); drain(0.3)
    a = ask(f"  Did it do: {name}? [y/n]")
    if not a.startswith("y"): bad.append(name)
ok("motor directions", not bad, wrong=bad)
if bad: print("  -> a side spinning backwards: swap that side's motor plug on the shield (TB6612 vs DRV8835 board variants differ). Left/right swapped: swap the two plugs.")

# 6. lights
say("6/9 Lights: red flash (the panic signal)")
send({"H": "l", "N": 8, "D1": 0, "D2": 255, "D3": 0, "D4": 0}); time.sleep(1.0); send({"H": "l", "N": 8, "D1": 0, "D2": 0, "D3": 0, "D4": 0}); drain(0.2)
ok("lights red", ask("  Did the LEDs flash red? [y/n]").startswith("y"))

# 7. floor: spin rate, speed, deadband
if not A.skip_floor:
    say("7/9 On the floor: spin rate, speed and the slowest speed that moves it (clear 1 m around it)")
    ask("  Car on the floor, area clear? Press Enter")
    t0 = time.time(); send({"H": "m", "N": 3, "D1": 2, "D2": 120})
    if A.yes: time.sleep(1.8)
    else: input("  Spinning at 120. Press Enter the moment it completes ONE full turn")
    turn_s = time.time() - t0; send({"H": "s", "N": 100}); drain(0.2)
    rate = 6.2832 / turn_s
    ok("spin rate at 120", 0.5 < rate < 8, rad_s=round(rate, 2), seconds_per_turn=round(turn_s, 2))
    send({"H": "d", "N": 4, "D1": 130, "D2": 130}); time.sleep(2.0); send({"H": "s", "N": 100}); drain(0.2)
    d = ask("  Drove 2 s at 130. How far, in cm?", "60")
    try: speed = float(d) / 100 / 2
    except ValueError: speed = None
    ok("speed at 130", speed is not None, m_s=speed)
    moved = None
    for pwm in range(50, 131, 10):
        send({"H": "d", "N": 4, "D1": pwm, "D2": pwm}); time.sleep(0.6); send({"H": "s", "N": 100}); drain(0.15)
        if ask(f"  Speed {pwm}: did it move? [y/n]", "y" if pwm >= 70 else "n").startswith("y"): moved = pwm; break
    ok("deadband", moved is not None, first_moving_pwm=moved)
    R["suggested"] = {"reflex.SPIN_RATE (at 120)": round(rate, 2), "car.MIN_PWM": moved, "sim speed at 130 (m/s)": speed}

# 8. failsafe
say("8/9 Failsafe: closing the link must stop the car")
send({"H": "d", "N": 4, "D1": 100, "D2": 100}); time.sleep(0.5); s.close(); time.sleep(1.0)
ok("stops when the laptop disconnects", ask("  Did it stop within ~1 s of the disconnect? [y/n]").startswith("y"))

# 9. cat detection on the real stream
say("9/9 Cat detection on the real camera: hold the plush 30-100 cm in front, at floor level")
ask("  Plush in view? Press Enter")
from ultralytics import YOLO
import torch
m = YOLO("yolo11n.pt"); dev = "mps" if torch.backends.mps.is_available() else "cpu"
cap = cv2.VideoCapture(f"http://{A.host}:{A.video_port}/stream"); hits, n, best = 0, 0, 0.0; t0 = time.time()
while time.time() - t0 < 8:
    good, f = cap.read()
    if not good: continue
    n += 1; r = m(f, classes=[15], conf=0.25, verbose=False, device=dev)[0]
    if len(r.boxes): hits += 1; best = max(best, float(r.boxes.conf.max()))
cap.release()
ok("plush detected as cat", n and hits / n > 0.5, hit_rate=round(hits / max(n, 1), 2), best_conf=round(best, 2), frames=n)
if not n or hits / n <= 0.5: print("  -> try yolo11s.pt/yolo11m.pt (medium scored the plush 0.94), more light, or the phone-video fallback")

say("Done. Results: demo_logs/preflight.json")
fails = [k for k, v in R["steps"].items() if not v["ok"]]
print("  failed:", fails or "none"); print("  suggested constants:", R.get("suggested"))
