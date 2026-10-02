"""YOLO cat detection + tracking -> Threat; monocular free space -> FreeSpace."""
from __future__ import annotations
import math, time, os
import numpy as np
from shapes import Threat, FreeSpace, now

CAT = 15  # COCO class id
# The plush reads as "dog" (0.83) far more than "cat"; counting cat, dog and teddy bear took the floor test from 0/46 to 46/46 frames
CAT_LIKE = [15, 16, 77]
# ...but loosened that far a dark jacket filling the frame read as a cat. Measured on 543 no-cat frames from the real car plus
# the plush captures: these per-class floors and a max box area give 0 false alarms (was 22) and keep the floor plush at 46/46.
MIN_CONF = {15: 0.30, 16: 0.50, 77: 0.50}   # 0.40 let a hand through as "dog" 0.44; the plush reads dog 0.83
MAX_AREA = 0.80

class Perceiver:
    def __init__(self, weights="yolo11n.pt", device=None, conf=0.30, hfov_deg=62.0, debounce=2, clahe=False, window=5, classes=(CAT,)):
        from ultralytics import YOLO
        import torch, collections
        self.model = YOLO(weights)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.conf, self.hfov, self.debounce, self.classes = conf, hfov_deg, debounce, list(classes)
        self._hits = 0
        self._recent = collections.deque(maxlen=window)    # present = debounce hits within the last `window` frames
        self._last_box = None
        self._clahe = None
        if clahe:                                          # the car's camera is pale and low-contrast (measured 2 Oct)
            import cv2
            self._clahe = cv2.createCLAHE(2.5, (8, 8))
        self._depth = None
        self._depth_tried = False

    def _prep(self, frame):
        if self._clahe is None: return frame
        import cv2
        l, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
        return cv2.cvtColor(cv2.merge([self._clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

    def threat(self, frame) -> Threat:
        h, w = frame.shape[:2]
        r = self.model.track(self._prep(frame), classes=self.classes + ([0] if len(self.classes) > 1 else []), conf=self.conf, device=self.device,
                             persist=True, verbose=False, tracker="bytetrack.yaml")[0]
        if r.boxes is None or len(r.boxes) == 0:
            self._recent.append(False); self._hits = sum(self._recent)
            if self._hits >= self.debounce and self._last_box is not None:   # a flickering detection still counts
                return self._last_box._replace(t=now()) if hasattr(self._last_box, "_replace") else Threat(now(), True, *self._last_box[2:])
            return Threat(now(), False)
        # biggest cat wins
        boxes = r.boxes.xyxy.cpu().numpy(); confs = r.boxes.conf.cpu().numpy()
        ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [None] * len(boxes)
        if len(self.classes) > 1:                          # cat-like mode: per-class floors, drop frame-filling blobs
            cls = r.boxes.cls.cpu().numpy().astype(int)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / float(w * h)
            people = [boxes[k] for k in range(len(boxes)) if cls[k] == 0 and confs[k] >= 0.3]
            def on_person(b):                              # an arm or a jacket read as "cat" sits inside a person box
                for p in people:
                    ix = max(0, min(b[2], p[2]) - max(b[0], p[0])); iy = max(0, min(b[3], p[3]) - max(b[1], p[1]))
                    if ix * iy > 0.6 * (b[2] - b[0]) * (b[3] - b[1]): return True
                return False
            keep = [k for k in range(len(boxes)) if cls[k] != 0 and confs[k] >= MIN_CONF.get(cls[k], 1.0)
                    and area[k] < MAX_AREA and not on_person(boxes[k])]
            if not keep:
                self._recent.append(False); self._hits = sum(self._recent)
                return Threat(now(), False)
            boxes, confs, ids = boxes[keep], confs[keep], [ids[k] for k in keep]
        i = int(np.argmax((boxes[:, 3] - boxes[:, 1])))
        x1, y1, x2, y2 = boxes[i]
        cx = (x1 + x2) / 2
        bearing = (cx - w / 2) / (w / 2) * (self.hfov / 2)
        proximity = float((y2 - y1) / h)
        self._recent.append(True); self._hits = sum(self._recent)
        present = self._hits >= self.debounce
        t = Threat(now(), present, None if ids[i] is None else int(ids[i]), float(bearing),
                   proximity, float(confs[i]), [int(v) for v in (x1, y1, x2, y2)])
        self._last_box = (t.t, True, t.track, t.bearing, t.proximity, t.conf, t.bbox)
        return t

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
