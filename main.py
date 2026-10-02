"""catbot: run the loop. --sim video.mp4 replaces the car. --no-cloud skips the supervisor."""
import argparse, time, collections, threading, json, os
import cv2
from shapes import Mode, Threat, FreeSpace, Command, to_json, now
from perceive import Perceiver, StallDetector, CAT_LIKE
from reflex import reflex
from recorder import Recorder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", help="video file to use as the car")
    ap.add_argument("--host", default="192.168.4.1")
    ap.add_argument("--video-port", type=int, default=81)
    ap.add_argument("--cmd-port", type=int, default=100)
    ap.add_argument("--no-cloud", action="store_true")
    ap.add_argument("--cloud-steers", action="store_true", help="let the cloud planner's mode drive the robot (default: it only narrates; steering measured worse, 63-77%% caught vs 26-45%%)")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = run)")
    ap.add_argument("--every", type=float, default=3.0, help="supervisor period, seconds")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="print a line for every frame")
    ap.add_argument("--dash", action="store_true", help="serve the demo screen on :8000")
    ap.add_argument("--demo-video", help="also record the annotated camera view at 10 fps (needs --dash)")
    a = ap.parse_args()

    car = __import__("sim").SimCar(a.sim) if a.sim else __import__("car").ElegooCar(a.host, a.video_port, a.cmd_port)
    # real car: the medium model on contrast-boosted frames at conf 0.15 (small model found the plush in 1/70 frames, this 22/70)
    per = Perceiver() if a.sim else Perceiver(os.environ.get("CAT_WEIGHTS", "yolo11m.pt"), conf=0.15, clahe=True, classes=CAT_LIKE, debounce=3, window=5)
    rec = Recorder()
    sup = None if a.no_cloud else __import__("supervisor").Supervisor()
    mode, last_sup, recent = Mode(now(), "idle", 1e9), 0.0, collections.deque(maxlen=12)
    NARRATE = Mode(now(), "idle", 1e9)             # what the reflex sees when the cloud only narrates
    state = {"mode": mode, "paused": not a.sim and a.dash}   # the real car waits for Resume on the dashboard
    if a.dash:
        from dash import Dash, annotate
        mem = None
        if not a.no_cloud or os.environ.get("MEMORY_BACKEND") in ("relay", "vast"):
            if os.environ.get("MEMORY_BACKEND") == "relay":
                from vast_memory import RelayMemory; mem = RelayMemory()    # VAST via the event-VM relay
            elif os.environ.get("MEMORY_BACKEND") == "vast":
                from vast_memory import VastMemory; mem = VastMemory()      # the event's VAST + Cosmos pipeline
            else:
                from memory import Memory; mem = Memory()                   # local fallback: NVIDIA captions + embeddings
            def ingest_loop():
                if os.environ.get("MEMORY_BACKEND") == "relay": return                                   # caption clips as they close, so a question only searches
                while True:
                    try: mem.ingest_new()
                    except Exception as e: print("ingest skipped:", e)
                    time.sleep(8)
            threading.Thread(target=ingest_loop, daemon=True).start()
        Dash(state, mem).start(); print("dash on http://localhost:8000")
        if a.demo_video:
            import numpy as np
            stamps = []; state["demo_stamps"] = stamps
            def rec_demo():
                vw = None
                while True:
                    jpg = state.get("jpg")
                    if jpg:
                        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                        if vw is None: vw = cv2.VideoWriter(a.demo_video, cv2.VideoWriter_fourcc(*"mp4v"), 10, (img.shape[1], img.shape[0]))
                        vw.write(img); state["demo_writer"] = vw
                        stamps.append(time.time())
                        if len(stamps) % 20 == 0: json.dump(stamps, open(a.demo_video + ".json", "w"))
                    time.sleep(0.1)
            threading.Thread(target=rec_demo, daemon=True).start()
    n_frames, t_fps = 0, time.time()
    last = None
    busy = [False]
    stall, last_cmd = StallDetector(), (0, 0)
    prev_present = False
    PANIC_COOLDOWN, EPISODE_GAP = 6.0, 4.0
    # Panic maneuver: spin ~540 degrees (a full lap plus a half turn, ending facing away), then run straight.
    # Time-based; PANIC_SPIN_S is the time for 540 degrees at PANIC_SPIN_PWM on this floor (tune by eye).
    PANIC_SPIN_PWM, PANIC_SPIN_S = int(os.environ.get("PANIC_SPIN_PWM", 200)), float(os.environ.get("PANIC_SPIN_S", 1.5))
    PANIC_RUN_PWM, PANIC_RUN_S = int(os.environ.get("PANIC_RUN_PWM", 200)), float(os.environ.get("PANIC_RUN_S", 1.5))           # a new sighting after 4 s without one is a new cat appearance
    cosmos_only = bool(sup)                          # with the cloud on, Cosmos is the only cat detector
    last_seen_verdict, last_sight, last_panic_t = [0.0], [0.0], [0.0]
    os.makedirs("demo_logs", exist_ok=True); panic_log = open("demo_logs/panic.log", "a", buffering=1)
    def plog(msg):
        line = "%s %s" % (time.strftime("%H:%M:%S"), msg); panic_log.write(line + "\n"); print("PANICLOG", line)
    def sighting(src):
        t = time.time(); gap = t - last_sight[0]; last_sight[0] = t
        if gap < EPISODE_GAP: return                 # same appearance, already handled
        if t - last_panic_t[0] < PANIC_COOLDOWN:
            plog("new sighting (%s) but cooldown %.1fs left: no video" % (src, PANIC_COOLDOWN - (t - last_panic_t[0]))); return
        last_panic_t[0] = t; state["panic"] = t; state["panic_src"] = src
        state["maneuver"] = (t, t + PANIC_SPIN_S, t + PANIC_SPIN_S + PANIC_RUN_S)   # spin 540, then run
        plog("PANIC fired, video should play (%s)" % src)
        try: car.flash_red(1.5)
        except Exception as e: plog("flash failed: %s" % e)
        rec.event_raw('{"kind":"panic","t":%f}' % t)
    plog("agent started; cat detector: %s" % ("Cosmos only" if cosmos_only else "local YOLO (offline)"))
    t_end = time.time() + a.seconds if a.seconds else 1e18      # --seconds 0 runs until stopped
    try:
        while time.time() < t_end:
            f = car.frame()
            if f is None:
                time.sleep(0.02); continue
            if hasattr(car, "frame_age") and car.frame_age() > 0.5:      # video frozen: never drive blind
                car.stop(); time.sleep(0.05); continue
            ultra = car.ultrasonic_cm() if not a.sim else None
            fs = per.free_space(f, ultra)
            v = state.get("verdict")
            if cosmos_only:                               # ONE cat detector: Cosmos. Its latest verdict, if under 2.5 s old.
                fresh = v is not None and time.time() - v.t < 2.5
                if fresh and v.cat_intent.startswith("visible"):
                    side = v.cat_intent.split(",")[-1].strip()
                    th = Threat(now(), True, None, {"left": -20.0, "right": 20.0}.get(side, 0.0), 0.3, 1.0, None, kind="threat")
                else:
                    th = Threat(now(), False)
                if fresh and v.t != last_seen_verdict[0]:
                    last_seen_verdict[0] = v.t
                    plog("cosmos: %s | %s" % (v.cat_intent, (v.see or "")[:80]))
                    if th.present: sighting("cosmos")
            else:                                         # no cloud: the local detector is the only cat detector
                th = per.threat(f)
                if th.present: sighting("local detector (offline)")
            prev_present = th.present
            if th.present: last = th
            stuck = stall.update(f, last_cmd[0] > 0 and last_cmd[1] > 0, time.time())
            cmd = reflex(th, fs, mode if a.cloud_steers else NARRATE, ultra, last, stuck)
            man = state.get("maneuver")
            if man and time.time() < man[2]:
                if time.time() < man[1]:
                    cmd = Command(now(), PANIC_SPIN_PWM, -PANIC_SPIN_PWM, "PANIC: spinning 540")
                elif ultra is not None and ultra < 25:      # wall ahead: stop running, let the reflex steer
                    state["maneuver"] = None
                else:
                    cmd = Command(now(), PANIC_RUN_PWM, PANIC_RUN_PWM, "PANIC: running away")
            if state.get("paused"):                      # dashboard pause: wheels stop, perception and recording go on
                cmd = cmd.__class__(**{**cmd.__dict__, "l": 0, "r": 0, "why": "paused"}) if hasattr(cmd, "__dict__") else cmd
            last_cmd = (cmd.l, cmd.r)
            car.drive(cmd.l, cmd.r)
            rec.frame(f); rec.event(th); rec.event(fs); rec.event(cmd)
            state.update(threat=th, command=cmd)
            n_frames += 1
            if time.time() - t_fps >= 1.0:
                state["fps"] = n_frames; n_frames, t_fps = 0, time.time()
            if a.dash: state["jpg"] = annotate(f, th, cmd, mode)
            recent.append(to_json(th)); recent.append(to_json(cmd))
            if sup and time.time() - last_sup > a.every and not busy[0]:
                last_sup = time.time(); busy[0] = True
                snap, clip, ev = f.copy(), rec.last_clip(), "\n".join(recent)
                def work(snap=snap, clip=clip, ev=ev):
                    try:
                        v = sup.verdict(snap, clip); rec.event(v); state["verdict"] = v
                        m = sup.plan(ev, v); rec.event(m); state["mode"] = m
                        print("SUPERVISOR", to_json(v), "->", to_json(m))
                    except Exception as e:
                        print("supervisor skipped:", e)
                    finally:
                        busy[0] = False
                threading.Thread(target=work, daemon=True).start()
            mode = state.get("mode", mode)
            if a.verbose: print("%s | %s" % (to_json(th)[:90], cmd.why))
            if a.show:
                if th.bbox: cv2.rectangle(f, th.bbox[:2], th.bbox[2:], (0, 255, 0), 2)
                cv2.putText(f, cmd.why, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("catbot", f); cv2.waitKey(1)
            if a.sim and car.done(): break
    finally:
        car.close(); rec.close()
        if state.get("demo_writer"):
            state["demo_writer"].release(); json.dump(state["demo_stamps"], open(a.demo_video + ".json", "w"))

if __name__ == "__main__":
    main()
