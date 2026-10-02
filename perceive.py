"""YOLO cat detection + tracking -> Threat; monocular free space -> FreeSpace."""
from __future__ import annotations
import math, time, os
import numpy as np
from shapes import Threat, FreeSpace, now

CAT = 15  # COCO class id

class Perceiver:
    def __init__(self, weights="yolo11n.pt", device=None, conf=0.30, hfov_deg=62.0, debounce=2):
        from ultralytics import YOLO
        import torch
        self.model = YOLO(weights)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.conf, self.hfov, self.debounce = conf, hfov_deg, debounce
        self._hits = 0
        self._depth = None
        self._depth_tried = False

    def threat(self, frame) -> Threat:
        h, w = frame.shape[:2]
        r = self.model.track(frame, classes=[CAT], conf=self.conf, device=self.device,
                             persist=True, verbose=False, tracker="bytetrack.yaml")[0]
        if r.boxes is None or len(r.boxes) == 0:
            self._hits = 0
            return Threat(now(), False)
        # biggest cat wins
        boxes = r.boxes.xyxy.cpu().numpy(); confs = r.boxes.conf.cpu().numpy()
        ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [None] * len(boxes)
        i = int(np.argmax((boxes[:, 3] - boxes[:, 1])))
        x1, y1, x2, y2 = boxes[i]
        cx = (x1 + x2) / 2
        bearing = (cx - w / 2) / (w / 2) * (self.hfov / 2)
        proximity = float((y2 - y1) / h)
        self._hits = min(self._hits + 1, self.debounce)
        present = self._hits >= self.debounce
        return Threat(now(), present, None if ids[i] is None else int(ids[i]), float(bearing),
                      proximity, float(confs[i]), [int(v) for v in (x1, y1, x2, y2)])

    # ---- free space ----
    def _load_depth(self):
        self._depth_tried = True
        try:
            from transformers import pipeline
            self._depth = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
                                   device=self.device if self.device != "mps" else "mps")
        except Exception as e:
            print("depth model unavailable, FreeSpace falls back to ultrasonic only:", e)

    def free_space(self, frame, ultrasonic_cm=None) -> FreeSpace:
        if self._depth is None and not self._depth_tried and os.environ.get("CATBOT_DEPTH", "1") == "1":
            self._load_depth()
        if self._depth is None:
            return FreeSpace(now(), 0.5, 0.5, 0.5, ultrasonic_cm)
        from PIL import Image
        small = Image.fromarray(frame[:, :, ::-1]).resize((320, 240))
        d = np.asarray(self._depth(small)["depth"], dtype=np.float32)   # relative inverse depth: big = near
        d = (d - d.min()) / (d.max() - d.min() + 1e-6)
        far = 1.0 - d                                                    # 1 = far/open
        lower = far[far.shape[0] // 2:]                                  # the floor half is what we drive on
        w = lower.shape[1] // 3
        L, C, R = (float(lower[:, :w].mean()), float(lower[:, w:2 * w].mean()), float(lower[:, 2 * w:].mean()))
        return FreeSpace(now(), L, C, R, ultrasonic_cm)


class StallDetector:
    """Driving but the camera image is not changing = stuck (wheels against a chair leg, a box corner).
    Needs no extra sensor, so it works the same on the real car and in the simulator."""
    def __init__(self, still=2.0, hold_s=1.0):
        self.prev = None; self.since = None; self.still, self.hold_s = still, hold_s

    def update(self, frame, driving: bool, t: float) -> bool:
        import cv2
        g = cv2.cvtColor(cv2.resize(frame, (80, 60)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        moving = True if self.prev is None else float(np.abs(g - self.prev).mean()) > self.still
        self.prev = g
        if driving and not moving:
            if self.since is None: self.since = t
            if t - self.since > self.hold_s: self.since = None; return True
        else:
            self.since = None
        return False
