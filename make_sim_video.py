"""Synthesize a 12 s test video: a cat photo slides in from the right, grows, then a wall-ish dark frame; for sim runs and tests."""
import cv2, numpy as np
cat = cv2.imread("walking.jpg"); W, H = 640, 480
w = cv2.VideoWriter("sim_cat.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
floor = np.full((H, W, 3), 190, np.uint8); floor[:H//2] = 225
for i in range(120):
    f = floor.copy()
    if 20 <= i < 100:
        k = (i - 20) / 80
        ch = int(80 + 300 * k); cw = int(ch * cat.shape[1] / cat.shape[0])
        c = cv2.resize(cat, (cw, ch)); x = int(W - 60 - k * (W // 2)); y = H - ch - 10
        x0, y0 = max(x, 0), max(y, 0); c = c[:H - y0, :W - x0]
        f[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] = c
    cv2.putText(f, "t=%.1f" % (i / 10), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    w.write(f)
w.release(); print("sim_cat.mp4")
