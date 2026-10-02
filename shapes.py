"""The five JSON shapes every module speaks. One dataclass each; to_json/from_json are the contract."""
from __future__ import annotations
import json, time
from dataclasses import dataclass, asdict, field
from typing import Optional, List

_clock = time.time            # the simulator swaps this for simulated time

def now() -> float:
    return _clock()

@dataclass
class Threat:
    t: float
    present: bool
    track: Optional[int] = None
    bearing: float = 0.0        # degrees; negative = cat is to the left of centre
    proximity: float = 0.0      # bbox height / frame height, 0..1
    conf: float = 0.0
    bbox: Optional[List[int]] = None
    kind: str = "threat"

@dataclass
class FreeSpace:
    t: float
    left: float = 0.5           # mean relative depth per third, 1 = far/open, 0 = wall
    center: float = 0.5
    right: float = 0.5
    ultrasonic_cm: Optional[float] = None
    kind: str = "freespace"

@dataclass
class Command:
    t: float
    l: int                      # -255..255, negative = that side backwards
    r: int
    why: str = ""
    kind: str = "command"

@dataclass
class Verdict:
    t: float
    cornered: bool = False
    open_dir: str = "center"    # left | center | right
    cat_intent: str = "none"    # approaching | retreating | idle | none
    clip: Optional[str] = None
    model: str = ""
    kind: str = "verdict"

@dataclass
class Mode:
    t: float
    mode: str = "idle"          # flee | escape_corner | patrol | idle
    ttl_s: float = 3.0
    reason: str = ""
    trace: Optional[str] = None
    kind: str = "mode"

    def live(self) -> bool:
        return now() - self.t < self.ttl_s

def to_json(obj) -> str:
    return json.dumps(asdict(obj), separators=(",", ":"))

SHAPES = {c.kind: c for c in (Threat(0, False), FreeSpace(0), Command(0, 0, 0), Verdict(0), Mode(0))}
SHAPES = {k: type(v) for k, v in SHAPES.items()}

def from_json(line: str):
    d = json.loads(line)
    return SHAPES[d["kind"]](**d)
