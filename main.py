"""catbot: run the loop. --sim video.mp4 replaces the car. --no-cloud skips the supervisor."""
import argparse, time, collections, threading, json, os
import cv2
from shapes import Mode, Threat, FreeSpace, to_json, now
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
    per = Perceiver() if a.sim else Perceiver(os.environ.get("CAT_WEIGHTS", "yolo11m.pt"), conf=0.15, clahe=True, classes=CAT_LIKE)
    rec = Recorder()
    sup = None if a.no_cloud else __import__("supervisor").Supervisor()
    mode, last_sup, recent = Mode(now(), "idle", 1e9), 0.0, collections.deque(maxlen=12)
    NARRATE = Mode(now(), "idle", 1e9)             # what the reflex sees when the cloud only narrates
    state = {"mode": mode}
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
    prev_present, last_panic = False, 0.0
    PANIC_COOLDOWN = 5.0
    t_end = time.time() + a.seconds if a.seconds else 1e18      # --seconds 0 runs until stopped
    try:
        while time.time() < t_end:
            f = car.frame()
            if f is None:
                time.sleep(0.02); continue
            if hasattr(car, "frame_age") and car.frame_age() > 0.5:      # video frozen: never drive blind
                car.stop(); time.sleep(0.05); continue
            ultra = car.ultrasonic_cm() if not a.sim else None
            th, fs = per.threat(f), per.free_space(f, ultra)
            v = state.get("verdict")                     # Cosmos saw a cat in the last 2.5 s and YOLO did not: trust Cosmos
            if not th.present and v is not None and v.cat_intent.startswith("visible") and time.time() - v.t < 2.5:
                side = v.cat_intent.split(",")[-1].strip()
                th = Threat(now(), True, None, {"left": -20.0, "right": 20.0}.get(side, 0.0), 0.3, 0.5, None, kind="threat")
            if th.present and not prev_present and time.time() - last_panic > PANIC_COOLDOWN:
                last_panic = time.time(); state["panic"] = last_panic       # dashboard plays the panic video
                try: car.flash_red(1.5)                                      # car lights go red
                except Exception as e: print("flash failed:", e)
                rec.event_raw('{"kind":"panic","t":%f}' % last_panic)
            prev_present = th.present
            if th.present: last = th
            stuck = stall.update(f, last_cmd[0] > 0 and last_cmd[1] > 0, time.time())
            cmd = reflex(th, fs, mode if a.cloud_steers else NARRATE, ultra, last, stuck)
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
