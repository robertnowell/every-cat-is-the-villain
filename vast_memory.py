"""Memory backed by the event's VAST video pipeline (VAST DataEngine -> YOLO11 -> Cosmos Reason captions ->
Cosmos-Embed1 -> VastDB), per the VAST Builders Challenge starter repo (relaxedtomato/vast-builders-challenge).

Each 5 s clip the robot records is uploaded with a custom prompt about cats and escapes; questions go to the
pipeline's grounded agent. Same interface as memory.Memory: ingest_new(), ask(question) -> {answer, clips}.

Config comes from the team config the event hands out (/config/<team>.config on the event VM):
  INGRESS_URL, USERNAME, PASSWORD   (backend + JWT login)
Select it with MEMORY_BACKEND=vast.
"""
from __future__ import annotations
import os, glob, json, time, requests

CUSTOM_PROMPT = ("Footage from a camera 8 cm above the floor on a small robot that flees cats. Describe: is a cat visible "
                 "(where in the frame, how close, what it is doing); is the robot moving, turning or backing away; is the robot "
                 "boxed in by walls or furniture; did the robot get away from the cat in this clip. Be literal; say 'no cat' if none.")

class VastMemory:
    def __init__(self, root="clips"):
        self.root = root
        self.base = os.environ["INGRESS_URL"].rstrip("/")
        self.user, self.pw = os.environ["USERNAME"], os.environ["PASSWORD"]
        self.sent_path = os.path.join(root, "vast_uploaded.json")
        self.sent = json.load(open(self.sent_path)) if os.path.exists(self.sent_path) else {}
        self.rows = list(self.sent.values())
        self._token, self._token_t = None, 0

    def _auth(self):
        if not self._token or time.time() - self._token_t > 1800:
            r = requests.post(self.base + "/api/v1/auth/login", json={"username": self.user, "password": self.pw}, timeout=20)
            r.raise_for_status(); self._token, self._token_t = r.json()["access_token"], time.time()
        return {"Authorization": "Bearer " + self._token}

    def ingest_new(self) -> int:
        n = 0
        for p in sorted(glob.glob(os.path.join(self.root, "*.mp4"))):
            if p in self.sent or os.path.getsize(p) < 1000 or time.time() - os.path.getmtime(p) < 6: continue   # skip the clip still being written
            with open(p, "rb") as f:
                r = requests.post(self.base + "/api/v1/videos/upload", headers=self._auth(), timeout=120,
                                  files={"file": (os.path.basename(p), f, "video/mp4")},
                                  data={"is_public": "true", "tags": "catbot,robot", "scenario": "general", "camera_id": "catbot-cam",
                                        "capture_type": "robot", "location": "aws-builder-loft", "custom_prompt": CUSTOM_PROMPT})
            if r.ok and r.json().get("success"):
                self.sent[p] = {"clip": p, "t": os.path.getmtime(p), "object_key": r.json().get("object_key"), "caption": ""}
                n += 1
        json.dump(self.sent, open(self.sent_path, "w")); self.rows = list(self.sent.values())
        return n

    def ask(self, question: str, k: int = 5) -> dict:
        h = self._auth()
        a = requests.post(self.base + "/api/v1/agent/ask", headers=h, json={"question": question, "top_k": k}, timeout=90).json()
        s = requests.post(self.base + "/api/v1/search", headers=h, timeout=60,
                          json={"query": question, "top_k": k, "llm_top_n": 3, "min_similarity": 0.25, "metadata_filters": {"camera_id": "catbot-cam"}}).json()
        by_key = {v.get("object_key"): v for v in self.sent.values()}
        clips = []
        for res in (s.get("results") or [])[:3]:
            src = res.get("source") or res.get("original_video") or res.get("object_key") or ""
            mine = next((v for key, v in by_key.items() if key and (key in src or src in key)), None)
            clips.append({"clip": mine["clip"] if mine else src, "t": mine["t"] if mine else 0,
                          "score": round(float(res.get("similarity", res.get("score", 0)) or 0), 3),
                          "caption": (res.get("reasoning_content") or res.get("caption") or "")[:300]})
        return {"question": question, "answer": a.get("answer", ""), "evidence": a.get("evidence"), "clips": clips, "backend": "vast"}
