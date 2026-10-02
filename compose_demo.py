"""Pair the camera recording and the top-down recording by wall-clock time, not by frame number.
Each recording writes <video>.json with one timestamp per frame. Output: side by side at 10 fps, H.264."""
import sys, json, bisect, subprocess, cv2, numpy as np

def frames(path):
    cap, out = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok: return out
        out.append(f)

def main(cam, top, out, fps=10):
    cf, tf = frames(cam), frames(top)
    cs, ts = json.load(open(cam + ".json"))[:len(cf)], json.load(open(top + ".json"))[:len(tf)]
    t0, t1 = max(cs[0], ts[0]), min(cs[-1], ts[-1])
    h = 360; raw = out + ".raw.mp4"
    cw = int(cf[0].shape[1] * h / cf[0].shape[0])
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (cw + 480, h))
    near = lambda st, t: min(max(bisect.bisect_left(st, t), 0), len(st) - 1)
    t = t0
    while t <= t1:
        a = cv2.resize(cf[near(cs, t)], (cw, h)); b = cv2.resize(tf[near(ts, t)], (480, h))
        vw.write(np.hstack([a, b])); t += 1 / fps
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30",
                    "-preset", "veryfast", "-movflags", "+faststart", out], check=True)
    print("synced %.1f s, max pairing gap %.0f ms" % (t1 - t0, 1000 * max(abs(cs[near(cs, x)] - ts[near(ts, x)]) for x in np.arange(t0, t1, 1 / fps))))

if __name__ == "__main__": main(*sys.argv[1:4])
