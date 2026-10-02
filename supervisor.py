"""Slow loop, every few seconds, advisory only.
verdict(): a vision model looks at the latest frame (Cosmos on the event endpoint when issued; NVIDIA-hosted Llama 3.2 vision tonight).
plan():    a W&B Inference LLM turns the recent events + verdict into a Mode, traced with Weave when WEAVE=1.
Keys come from the environment (claude-secrets run --inject NVIDIA_API_KEY=NVIDIA_API_KEY WANDB_API_KEY=WANDB_API_KEY -- ...)."""
from __future__ import annotations
import os, json, base64, time, re
import cv2
from openai import OpenAI
from shapes import Verdict, Mode, now

NV_BASE = os.environ.get("COSMOS_BASE", "https://integrate.api.nvidia.com/v1")
NV_MODEL = os.environ.get("COSMOS_MODEL", "meta/llama-3.2-11b-vision-instruct")
WB_BASE = "https://api.inference.wandb.ai/v1"
WB_MODEL = os.environ.get("PLANNER_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
WB_PROJECT = os.environ.get("WANDB_PROJECT", "robaroni-trykopi-ai/catbot")

# Neutral on purpose: an earlier prompt said the robot "must run away from a cat", and the model answered
# "cornered, cat approaching" on 15 of 15 frames, 11 of them catless. This wording scored 10/12 on cat and
# 10/11 on cornered against hand-labelled frames (29 Sep); the old one scored 7/12 and 5/11.
VERDICT_PROMPT = ("This is one frame from a camera mounted 8 cm above the floor on a small wheeled robot. Describe only what is "
       "actually visible. First write one sentence saying what you see. Then answer: is a cat visible in this frame? "
       "If so, in which part of the frame (left, center or right)? Which part of the floor ahead is most open "
       "(left, center or right)? Is the robot boxed in, meaning walls or objects fill nearly the whole view close to the camera? "
       "Is a hot dog (a sausage in a bun, real or toy) visible? If so, where? "
       "Reply ONLY with JSON: {\"see\": string, \"cat_visible\": true or false, \"cat_side\": \"left|center|right|none\", "
       "\"open_dir\": \"left|center|right\", \"cornered\": true or false, "
       "\"hotdog_visible\": true or false, \"hotdog_side\": \"left|center|right|none\"}")
PLAN_SYSTEM = ("You are the mode planner for a small robot that flees a cat. Modes: flee (cat seen, keep reflexes running), "
               "escape_corner (spin out of a corner first), patrol (no cat, wander), idle (stay). "
               "Reply ONLY with JSON {\"mode\": str, \"ttl_s\": int, \"reason\": str}.")

def _json(s: str) -> dict:
    m = re.search(r"\{.*\}", s, re.S)
    return json.loads(m.group(0)) if m else {}

class Supervisor:
    def __init__(self):
        self.nv = OpenAI(base_url=NV_BASE, api_key=os.environ.get("COSMOS_API_KEY") or os.environ.get("NVIDIA_API_KEY", "missing"))
        self.wb = OpenAI(base_url=WB_BASE, api_key=os.environ.get("WANDB_API_KEY", "missing"),
                         default_headers={"OpenAI-Project": WB_PROJECT})
        self.plan = self._plan
        if os.environ.get("WEAVE") == "1":
            import weave
            weave.init(WB_PROJECT)
            self.plan = weave.op()(self._plan)

    def verdict(self, frame, clip=None) -> Verdict:
        ok, jpg = cv2.imencode(".jpg", cv2.resize(frame, (640, 480)), [cv2.IMWRITE_JPEG_QUALITY, 75])
        url = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
        t0 = time.time()
        r = self.nv.chat.completions.create(model=NV_MODEL, max_tokens=220, temperature=0.2, messages=[{"role": "user", "content": [
            {"type": "text", "text": VERDICT_PROMPT}, {"type": "image_url", "image_url": {"url": url}}]}])
        d = _json((r.choices[0].message.content or "").replace("```json", "").replace("```", ""))
        vis = d.get("cat_visible", False); vis = vis if isinstance(vis, bool) else str(vis).lower() == "true"
        v = Verdict(now(), bool(d.get("cornered", False)), str(d.get("open_dir", "center")),
                    ("visible, " + str(d.get("cat_side", "")).strip()) if vis else "none", clip, NV_MODEL)
        v.see = str(d.get("see", ""))[:200]
        hd = d.get("hotdog_visible", False); hd = hd if isinstance(hd, bool) else str(hd).lower() == "true"
        v.hotdog = ("visible, " + str(d.get("hotdog_side", "")).strip()) if hd else "none"
        v.latency_s = round(time.time() - t0, 2)
        return v

    def _plan(self, recent_events: str, verdict: Verdict) -> Mode:
        t0 = time.time()
        r = self.wb.chat.completions.create(model=WB_MODEL, max_tokens=120, temperature=0, messages=[
            {"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": "Recent events (newest last):\n%s\nLatest vision verdict: cornered=%s open_dir=%s cat_intent=%s" %
             (recent_events, verdict.cornered, verdict.open_dir, verdict.cat_intent)}])
        d = _json(r.choices[0].message.content or "")
        m = Mode(now(), str(d.get("mode", "idle")), float(d.get("ttl_s", 3)), str(d.get("reason", "")))
        m.latency_s = round(time.time() - t0, 2)
        return m
