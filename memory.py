"""The video agent's memory: every 5 s clip gets a caption (vision model), an embedding, and a row.
Ask it questions in English; it returns the matching clips and an LLM answer grounded in them.

Matches VAST's reference pipeline shape (clip -> VLM description -> nv-embedqa embedder -> vector search -> LLM),
with the vector store as a local numpy table tonight and VastDB when the event issues one (see VastStore stub).
Cosmos replaces the caption model via COSMOS_MODEL / COSMOS_BASE when its endpoint exists."""
from __future__ import annotations
import os, json, base64, time, glob
import numpy as np
import cv2
from openai import OpenAI

NV_BASE = os.environ.get("COSMOS_BASE", "https://integrate.api.nvidia.com/v1")
CAPTION_MODEL = os.environ.get("COSMOS_MODEL", "meta/llama-3.2-11b-vision-instruct")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nvidia/nemotron-3-embed-1b")
WB_BASE = "https://api.inference.wandb.ai/v1"
WB_MODEL = os.environ.get("PLANNER_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
WB_PROJECT = os.environ.get("WANDB_PROJECT", "robaroni-trykopi-ai/catbot")

CAPTION_PROMPT = ("This is a frame from the camera of a small wheeled robot whose job is to run away from a cat. "
                  "In two sentences describe what is happening: whether a cat is visible, how close it is, what it is doing, "
                  "whether the robot is in open floor or boxed in by walls or furniture.")

def _mid_frame(path):
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2)); ok, f = cap.read(); cap.release()
    return f if ok else None

def _events_for(clip_t: float, events: list, seg_s=5.0) -> dict:
    """Numbers from events.jsonl for the clip's window: what the robot itself measured."""
    win = [e for e in events if clip_t <= e["t"] < clip_t + seg_s]
    th = [e for e in win if e["kind"] == "threat" and e.get("present")]
    cmds = [e for e in win if e["kind"] == "command"]
    prox = [e["proximity"] for e in th]
    peak = max(prox, default=0.0)
    escaped = bool(prox) and peak >= 0.35 and prox[-1] < peak - 0.15      # the cat got close, then the robot opened the gap
    return {"cat_frames": len(th), "max_proximity": peak, "escaped": escaped,
            "reverses": sum(1 for c in cmds if c["l"] < 0 and c["r"] < 0),
            "spins": sum(1 for c in cmds if c["l"] * c["r"] < 0),
            "modes": sorted({e["mode"] for e in win if e["kind"] == "mode"}),
            "cornered": any(e.get("cornered") for e in win if e["kind"] == "verdict")}

class Memory:
    def __init__(self, root="clips", index_path=None):
        self.root = root
        self.index_path = index_path or os.path.join(root, "memory.json")
        self.nv = OpenAI(base_url=NV_BASE, api_key=os.environ.get("NVIDIA_API_KEY", "missing"))
        self.emb = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ.get("NVIDIA_API_KEY", "missing"))
        self.wb = OpenAI(base_url=WB_BASE, api_key=os.environ.get("WANDB_API_KEY", "missing"), default_headers={"OpenAI-Project": WB_PROJECT})
        self.rows = json.load(open(self.index_path)) if os.path.exists(self.index_path) else []

    # ---- ingest ----
    def caption(self, frame) -> str:
        ok, jpg = cv2.imencode(".jpg", cv2.resize(frame, (640, 480)), [cv2.IMWRITE_JPEG_QUALITY, 75])
        r = self.nv.chat.completions.create(model=CAPTION_MODEL, max_tokens=120, temperature=0.2, messages=[{"role": "user", "content": [
            {"type": "text", "text": CAPTION_PROMPT}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()}}]}])
        return (r.choices[0].message.content or "").strip()

    def embed(self, texts, input_type="passage"):
        r = self.emb.embeddings.create(model=EMBED_MODEL, input=texts, extra_body={"input_type": input_type, "truncate": "END"})
        return np.array([d.embedding for d in r.data], dtype=np.float32)

    def ingest_new(self) -> int:
        events = [json.loads(l) for l in open(os.path.join(self.root, "events.jsonl"))] if os.path.exists(os.path.join(self.root, "events.jsonl")) else []
        clip_t = {e["path"]: e["t"] for e in events if e["kind"] == "clip"}
        known = {r["clip"] for r in self.rows}
        new = [p for p in sorted(glob.glob(os.path.join(self.root, "*.mp4"))) if p not in known and os.path.getsize(p) > 0]
        for p in new:
            f = _mid_frame(p)
            if f is None: continue
            t = clip_t.get(p, os.path.getmtime(p))
            stats = _events_for(t, events)
            cap = self.caption(f)
            text = cap + " Robot log: cat visible in %d of ~50 frames; closest approach %d%% of frame height (%s); %s; reversed %d times, spun %d times; cornered=%s; modes=%s." % (
                stats["cat_frames"], round(stats["max_proximity"] * 100), "very close" if stats["max_proximity"] > 0.5 else "close" if stats["max_proximity"] > 0.3 else "far",
                "ESCAPE: the robot opened the gap" if stats["escaped"] else "no escape in this clip", stats["reverses"], stats["spins"], stats["cornered"], ",".join(stats["modes"]) or "none")
            vec = self.embed([text])[0]
            self.rows.append({"clip": p, "t": t, "caption": cap, "text": text, "stats": stats, "vec": vec.tolist()})
        json.dump(self.rows, open(self.index_path, "w"))
        return len(new)

    # ---- query ----
    def search(self, question: str, k=3):
        if not self.rows: return []
        q = self.embed([question], input_type="query")[0]
        M = np.array([r["vec"] for r in self.rows], dtype=np.float32)
        sims = M @ q / (np.linalg.norm(M, axis=1) * np.linalg.norm(q) + 1e-9)
        order = np.argsort(-sims)[:k]
        return [dict(self.rows[i], score=float(sims[i])) for i in order]

    def ask(self, question: str, k=3) -> dict:
        hits = self.search(question, k)
        ctx = "\n".join("[%d] clip %s at t=%s: %s" % (i + 1, os.path.basename(h["clip"]), time.strftime("%H:%M:%S", time.localtime(h["t"])), h["text"]) for i, h in enumerate(hits))
        r = self.wb.chat.completions.create(model=WB_MODEL, max_tokens=200, temperature=0, messages=[
            {"role": "system", "content": "You answer questions about a small robot's recorded life from its clip captions and logs. Each clip is 5 seconds. "
             "'Closest approach' is the cat's size as a percentage of the frame height (bigger = closer), not a distance. An 'escape' is a clip whose log says ESCAPE. "
             "'Cornered' is the vision model's judgement. Answer in two or three sentences, name the clip time, cite clips as [n]. If nothing matches, say so plainly."},
            {"role": "user", "content": "Clips:\n%s\n\nQuestion: %s" % (ctx, question)}])
        return {"question": question, "answer": (r.choices[0].message.content or "").strip(), "clips": [{"clip": h["clip"], "t": h["t"], "score": round(h["score"], 3), "caption": h["caption"]} for h in hits]}

if __name__ == "__main__":
    import sys
    m = Memory()
    n = m.ingest_new(); print("ingested", n, "new clips; index has", len(m.rows))
    for q in (sys.argv[1:] or ["when did the cat last corner me?", "show every escape", "when was the cat closest?"]):
        a = m.ask(q); print("\nQ:", q); print("A:", a["answer"]); [print("   ", os.path.basename(c["clip"]), c["score"], "|", c["caption"][:90]) for c in a["clips"]]
