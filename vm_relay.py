"""Runs ON THE EVENT VM (the only machine that can reach the team's VAST pipeline).
Pulls new robot clips and pending questions from the laptop dashboard through its tunnel, uploads clips to the
VAST video pipeline (DataEngine -> YOLO11 -> Cosmos Reason -> Cosmos-Embed1 -> VastDB) with a cat-and-escape prompt,
answers questions with the pipeline's agent and search, and posts the answers back. Standard library only.

    python3 vm_relay.py https://<tunnel>.trycloudflare.com <relay-token>
"""
import json, sys, time, uuid, urllib.request, urllib.error

LAPTOP, TOKEN = sys.argv[1].rstrip("/"), sys.argv[2]
CFG = {}
import os
for line in open(os.environ.get("RELAY_CONFIG", "/config/team-42.config")):
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.strip().split("=", 1); CFG[k] = v.strip().strip('"')
BACKEND = CFG["INGRESS_URL"].rstrip("/")
PROMPT = ("Footage from a camera 8 cm above the floor on a small robot that flees cats. Describe: is a cat visible "
          "(where in the frame, how close, what it is doing); is the robot moving, turning or backing away; is the robot "
          "boxed in by walls or furniture; did the robot get away from the cat in this clip. Be literal; say 'no cat' if none.")

def http(method, url, body=None, headers=None, timeout=120):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r: return r.read()

def laptop(path, payload=None):
    sep = "&" if "?" in path else "?"
    if payload is None: return json.loads(http("GET", f"{LAPTOP}{path}{sep}token={TOKEN}"))
    return json.loads(http("POST", f"{LAPTOP}{path}{sep}token={TOKEN}", json.dumps(payload).encode(), {"Content-Type": "application/json"}))

_tok = {"v": None, "t": 0}
def auth():
    if not _tok["v"] or time.time() - _tok["t"] > 1500:
        r = json.loads(http("POST", BACKEND + "/api/v1/auth/login", json.dumps({"username": CFG["USERNAME"], "password": CFG["PASSWORD"]}).encode(), {"Content-Type": "application/json"}, 30))
        _tok.update(v=r["access_token"], t=time.time())
    return {"Authorization": "Bearer " + _tok["v"]}

def upload(name, data):
    b = uuid.uuid4().hex; parts = []
    fields = {"is_public": "true", "tags": "catbot,robot", "scenario": "general", "camera_id": "catbot-cam",
              "capture_type": "robot", "location": "aws-builder-loft", "custom_prompt": PROMPT}
    for k, v in fields.items(): parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\nContent-Type: video/mp4\r\n\r\n'.encode() + data + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    h = auth(); h["Content-Type"] = f"multipart/form-data; boundary={b}"
    return json.loads(http("POST", BACKEND + "/api/v1/videos/upload", b"".join(parts), h, 180))

def post(path, body):
    h = auth(); h["Content-Type"] = "application/json"
    try:
        return json.loads(http("POST", BACKEND + path, json.dumps(body).encode(), h, 150)), None
    except urllib.error.HTTPError as e:
        return None, f"{path} {e.code}: {e.read()[:200].decode(errors='replace')}"
    except Exception as e:
        return None, f"{path}: {str(e)[:200]}"

def ask(q):
    """Filtered search-and-answer first (only our robot's camera), then plain agent/ask, then search's own synthesis."""
    flt = {"query": q, "top_k": 10, "llm_top_n": 3, "min_similarity": 0.2, "metadata_filters": {"camera_id": "catbot-cam"}}
    errs, answer, chunks = [], "", []
    r, e = post("/api/v1/agent/search-and-answer", flt)
    if r: answer, chunks = r.get("answer", ""), (r.get("evidence") or {}).get("chunks") or []
    else: errs.append(e)
    if not answer:
        r, e = post("/api/v1/agent/ask", {"question": q, "top_k": 10})
        if r: answer = r.get("answer", "")
        else: errs.append(e)
    if not chunks or not answer:
        r, e = post("/api/v1/search", flt)
        if r:
            chunks = chunks or r.get("chunk_results") or r.get("results") or []
            answer = answer or (r.get("llm_synthesis") or {}).get("response", "")
        else: errs.append(e)
    if errs: print("ask errors:", errs, flush=True)
    return answer or ("VAST had no answer. " + " | ".join(errs)[:300]), chunks

print("relay up: laptop", LAPTOP, "-> VAST", BACKEND, flush=True)
KEYS = "relay_keys.json"
try: keys = json.load(open(KEYS))                       # object_key -> clip name on the laptop
except Exception: keys = {}
while True:
    try:
        p = laptop("/relay/pending")
        for name in p.get("clips", [])[:3]:
            data = http("GET", f"{LAPTOP}/relay/clip/{name}?token={TOKEN}")
            r = upload(name, data); ok = bool(r.get("success"))
            if ok: keys[r.get("object_key", "")] = name; json.dump(keys, open(KEYS, "w"))
            laptop("/relay/ack", {"clip": name, "ok": ok, "object_key": r.get("object_key"), "error": None if ok else str(r)[:200]})
            print("uploaded", name, ok, r.get("object_key"), flush=True)
        for q in p.get("questions", []):
            answer, chunks = ask(q["q"]); clips = []
            for c in chunks[:4]:
                src = json.dumps(c)
                mine = next((n for k, n in keys.items() if k and k in src), None)
                clips.append({"clip": mine, "caption": (c.get("reasoning_content") or c.get("caption") or "")[:300],
                              "score": c.get("similarity_score", c.get("similarity")), "source": c.get("original_video") or c.get("source"),
                              "start": c.get("best_match_start_sec")})
            laptop("/relay/answer", {"id": q["id"], "answer": answer, "clips": clips, "backend": "vast"})
            print("answered", q["q"][:60], "|", answer[:80], flush=True)
    except Exception as e:
        print("relay loop:", str(e)[:200], flush=True)
    time.sleep(3)
