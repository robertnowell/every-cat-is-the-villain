"""Spin in place at a few low speeds to find the slowest that turns. WHEELS MOVE. Raw commands, no MIN_PWM lift."""
import socket, threading, time, json, sys
speeds = [int(x) for x in sys.argv[1:]] or [50, 60, 70]
s = socket.create_connection(("192.168.4.1", 100), timeout=5)
def rd():
    s.settimeout(None)
    while True:
        try: d = s.recv(256)
        except OSError: return
        if not d: return
        if b"{Heartbeat}" in d: s.sendall(b"{Heartbeat}")
threading.Thread(target=rd, daemon=True).start()
send = lambda o: s.sendall(json.dumps(o, separators=(",", ":")).encode() + b"\n")
time.sleep(1.5)
for sp in speeds:
    print(f"spinning at {sp} ...", flush=True)
    send({"H": "m", "N": 3, "D1": 2, "D2": sp}); time.sleep(float(__import__("os").environ.get("SPIN_S", 1.5))); send({"H": "s", "N": 100})
    print(f"  stopped. Did it turn at {sp}?", flush=True); time.sleep(2.0)
send({"H": "s", "N": 100}); time.sleep(0.3); print("done")
