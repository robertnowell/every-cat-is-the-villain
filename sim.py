"""A car made of a video file. Same interface as car.py so everything else runs without hardware."""
import cv2, time, threading

class SimCar:
    def __init__(self, video: str, fps: float = 10.0, loop: bool = True):
        self.cap = cv2.VideoCapture(video)
        self.fps, self.loop = fps, loop
        self.commands = []          # (t, l, r) log
        self._frame = None
        self._alive = True
        self._done = False
        threading.Thread(target=self._play, daemon=True).start()

    def _play(self):
        while self._alive:
            ok, f = self.cap.read()
            if not ok:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                self._done = True; break
            self._frame = f
            time.sleep(1.0 / self.fps)

    def frame(self):
        return None if self._frame is None else self._frame.copy()

    def drive(self, l, r): self.commands.append((time.time(), int(l), int(r)))
    def move(self, direction, speed): self.commands.append((time.time(), direction, int(speed)))
    def stop(self): self.commands.append((time.time(), 0, 0))
    def ultrasonic_cm(self, timeout=0.3): return 100.0
    def camera_servo(self, angle): pass
    def lights(self, r, g, b): pass
    def flash_red(self, seconds=1.5): self.commands.append((time.time(), "flash_red", seconds))
    def done(self): return self._done
    def close(self): self._alive = False
