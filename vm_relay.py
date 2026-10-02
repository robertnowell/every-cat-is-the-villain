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

def ask(q):
    h = auth(); h["Content-Type"] = "application/json"
    a = json.loads(http("POST", BACKEND + "/api/v1/agent/ask", json.dumps({"question": q, "top_k": 8}).encode(), h, 120))
    s = json.loads(http("POST", BACKEND + "/api/v1/search", json.dumps({"query": q, "top_k": 8, "llm_top_n": 3, "min_similarity": 0.2,
                         "metadata_filters": {"camera_id": "catbot-cam"}}).encode(), h, 120))
    return a, s

print("relay up: laptop", LAPTOP, "-> VAST", BACKEND, flush=True)
keys = {}                                             # object_key -> clip name on the laptop
while True:
    try:
        p = laptop("/relay/pending")
        for name in p.get("clips", [])[:3]:
            data = http("GET", f"{LAPTOP}/relay/clip/{name}?token={TOKEN}")
            r = upload(name, data); ok = bool(r.get("success"))
            if ok: keys[r.get("object_key", "")] = name
            laptop("/relay/ack", {"clip": name, "ok": ok, "object_key": r.get("object_key"), "error": None if ok else str(r)[:200]})
            print("uploaded", name, ok, r.get("object_key"), flush=True)
        for q in p.get("questions", []):
            try:
                a, s = ask(q["q"]); clips = []
                for res in (s.get("results") or [])[:4]:
                    src = json.dumps(res)
                    mine = next((n for k, n in keys.items() if k and k.split("/")[-1].split("_", 1)[-1] in src), None) or \
                           next((n for n in set(keys.values()) if n and n in src), None)
                    clips.append({"clip": mine, "caption": (res.get("reasoning_content") or res.get("caption") or res.get("description") or "")[:300],
                                  "score": res.get("similarity", res.get("score")), "source": res.get("source") or res.get("original_video")})
                laptop("/relay/answer", {"id": q["id"], "answer": a.get("answer", ""), "evidence": a.get("evidence"), "clips": clips, "backend": "vast"})
                print("answered", q["q"][:60], flush=True)
            except Exception as e:
                laptop("/relay/answer", {"id": q["id"], "answer": "VAST error: " + str(e)[:200], "clips": [], "backend": "vast"})
    except Exception as e:
        print("relay loop:", str(e)[:200], flush=True)
    time.sleep(3)
