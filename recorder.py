"""5 s clips + events.jsonl. Everything the memory will index comes from here."""
from __future__ import annotations
import os, time, cv2
from shapes import to_json

class Recorder:
    def __init__(self, root="clips", seg_s=5.0, fps=10):
        os.makedirs(root, exist_ok=True)
        self.root, self.seg_s, self.fps = root, seg_s, fps
        self.events = open(os.path.join(root, "events.jsonl"), "a")
        self._w = None; self._seg_start = 0; self._n = 0; self.current = None

    def frame(self, f):
        h, w = f.shape[:2]
        if self._w is None or time.time() - self._seg_start >= self.seg_s:
            if self._w is not None: self._w.release()
            self.current = os.path.join(self.root, "%06d.mp4" % self._n); self._n += 1
            self._w = cv2.VideoWriter(self.current, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            self._seg_start = time.time()
            self.event_raw('{"kind":"clip","t":%f,"path":"%s"}' % (self._seg_start, self.current))
        self._w.write(f)

    def event(self, obj): self.event_raw(to_json(obj))
    def event_raw(self, line: str):
        self.events.write(line + "\n"); self.events.flush()

    def last_clip(self): return self.current
    def close(self):
        if self._w is not None: self._w.release()
        self.events.close()
