"""Why does the link drop when the motors start? Wheels UP. Flashes the lights, then runs each side slowly and briefly,
logging every byte from the car. If the car closes the link (or sends {"N":100}, its boot/disconnect marker) right
after a motor command, the motors are browning out the board: charge or reseat the batteries."""
import socket, threading, time, json, sys
host = sys.argv[1] if len(sys.argv) > 1 else "192.168.4.1"
s = socket.create_connection((host, 100), timeout=5)
log, alive, t0 = [], {"v": True}, time.time()
def rd():
    s.settimeout(None)
    while True:
        try: d = s.recv(256)
        except OSError as e: log.append((round(time.time()-t0, 2), "ERR " + str(e))); alive["v"] = False; return
        if not d: log.append((round(time.time()-t0, 2), "CLOSED by car")); alive["v"] = False; return
        log.append((round(time.time()-t0, 2), d.decode(errors="ignore")))
        if b"{Heartbeat}" in d: s.sendall(b"{Heartbeat}")
threading.Thread(target=rd, daemon=True).start()
def send(o):
    try: s.sendall(json.dumps(o, separators=(",", ":")).encode() + b"\n"); return True
    except OSError as e: log.append((round(time.time()-t0, 2), "SEND FAIL " + str(e))); return False
time.sleep(2.5); print("idle 2.5 s, link alive:", alive["v"])
steps = [("lights green", {"H": "l", "N": 8, "D1": 0, "D2": 0, "D3": 255, "D4": 0}),
         ("SPIN LEFT command (N=3 D1=1), speed 110", {"H": "m", "N": 3, "D1": 1, "D2": 110}),
         ("both forward, speed 150, 1.0 s", {"H": "d", "N": 4, "D1": 150, "D2": 150}),
         ("spin, speed 200, 1.0 s", {"H": "m", "N": 3, "D1": 2, "D2": 200})]
for name, cmd in steps:
    if "SPIN LEFT" in name: print("watch: which way does it spin, seen from above? (left = counter-clockwise)")
    ok = send(cmd); time.sleep(1.0 if "1.0 s" in name else 0.6); send({"H": "s", "N": 100}); time.sleep(2.0)
    print(f"{name:40s} sent={ok}  link alive after={alive['v']}")
    if not alive["v"]: break
send({"H": "l", "N": 8, "D1": 0, "D2": 0, "D3": 0, "D4": 0}); time.sleep(0.5)
print("\nwhat the car sent (seconds, bytes):")
for t, m in log: print(" ", t, repr(m)[:80])
