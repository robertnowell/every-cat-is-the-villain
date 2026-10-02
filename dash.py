"""Demo screen: live annotated stream, the current Threat/Command/Mode/Verdict, a question box over the memory, clip playback.
Runs inside the main loop process (Dash(state).start()) and reads the shared `state` dict the loop updates."""
from __future__ import annotations
import threading, time, os, json
import cv2
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
import uvicorn

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>catbot</title>
<style>body{margin:0;background:#111;color:#eee;font:15px/1.4 -apple-system,Helvetica,sans-serif}
.g{display:grid;grid-template-columns:2fr 1fr;gap:16px;padding:16px}img{width:100%;border:1px solid #333;border-radius:6px}
.p{background:#1b1b1b;border:1px solid #333;border-radius:6px;padding:12px;margin-bottom:12px}.k{font:12px ui-monospace,monospace;letter-spacing:.08em;color:#8a8;text-transform:uppercase}
.big{font-size:28px;font-weight:700}.mode-flee{color:#f6a}.mode-escape_corner{color:#fa4}.mode-patrol{color:#8cf}.mode-idle{color:#999}
input{box-sizing:border-box;width:100%;padding:10px;font-size:16px;background:#111;color:#eee;border:1px solid #444;border-radius:6px}
.clip{display:flex;gap:10px;margin-top:10px;align-items:flex-start}video{width:220px;border-radius:4px}small{color:#999}</style></head><body>
<div id="start" style="position:fixed;inset:0;background:rgba(0,0,0,.82);display:flex;align-items:center;justify-content:center;z-index:9">
 <button onclick="startDemo()" style="font-size:28px;padding:22px 40px;border-radius:12px;border:0;background:#f6a;color:#111;font-weight:700;cursor:pointer">Start the demo (with sound)</button></div>
<div class="g"><div style="position:relative"><img src="/stream"><video id="panic" src="/panic.mp4" playsinline preload="auto" style="position:absolute;right:12px;bottom:104px;width:21%;aspect-ratio:9/16;object-fit:cover;background:#000;border:3px solid #f33;border-radius:8px;box-shadow:0 4px 18px rgba(0,0,0,.6);display:none;z-index:2"></video><video id="happy" src="/happy.mp4" playsinline preload="auto" style="position:absolute;left:12px;bottom:104px;width:30%;aspect-ratio:16/9;object-fit:cover;background:#000;border:3px solid #3c3;border-radius:8px;box-shadow:0 4px 18px rgba(0,0,0,.6);display:none;z-index:2"></video><div class="p" id="why"></div><button id="pause" onclick="togglePause()" title="Space bar also works" style="margin-top:8px;width:100%;padding:14px;border-radius:8px;border:0;background:#f5c542;color:#111;font-size:18px;font-weight:800;cursor:pointer">⏸ PAUSE ROBOT (space)</button></div>
<div><div class="p"><div class="k">cloud planner says</div><div class="big" id="mode">idle</div><div id="reason"></div><small id="lat"></small></div>
<div class="p" id="mapbox" style="display:none"><div class="k">the room, from above (simulator only) · blue robot, red cat</div><img id="map" style="margin-top:6px">
 <button onclick="releaseCat()" style="margin-top:8px;padding:8px 14px;border-radius:6px;border:0;background:#8cf;color:#111;font-weight:700;cursor:pointer">Release a new cat</button></div>
<div class="p"><div class="k">cat</div><div id="threat"></div></div>
<div class="p"><div class="k">vision verdict</div><div id="verdict"></div></div>
<div class="p"><div class="k">ask the memory</div><input id="q" placeholder="when did the cat last corner me?" onkeydown="if(event.key==='Enter')ask()"><div id="answer"></div><div id="clips"></div></div></div></div>
<script>
let lastPanic=null, lastHappy=null; const SIM=%SIM%;
function startDemo(){document.getElementById('start').style.display='none'; const v=document.getElementById('panic'); v.muted=true; v.play().then(()=>{v.pause(); v.currentTime=0; v.muted=false;}).catch(()=>{});}
async function togglePause(){ const r=await (await fetch('/pause',{method:'POST'})).json(); showPause(r.paused); }
function showPause(p){ const b=document.getElementById('pause'); b.textContent=p?'▶ RESUME ROBOT (space)':'⏸ PAUSE ROBOT (space)'; b.style.background=p?'#5c5':'#f5c542'; }
document.addEventListener('keydown',e=>{ if(e.code==='Space' && document.activeElement.tagName!=='INPUT'){ e.preventDefault(); togglePause(); } });
function releaseCat(){ if(SIM) fetch(SIM+'/reset'); }
if(SIM){document.getElementById('mapbox').style.display='block'; document.getElementById('map').src=SIM+'/map';}
function panic(){const v=document.getElementById('panic'); v.style.display='block'; v.currentTime=0; v.muted=false;
 v.play().then(()=>blog('video playing with sound')).catch(e=>{blog('play with sound BLOCKED ('+e.name+'), retrying muted'); v.muted=true; v.play().then(()=>blog('video playing muted')).catch(e2=>blog('muted play failed too: '+e2.name));}); v.onended=()=>{v.style.display='none';};}
function happyPlay(){const v=document.getElementById('happy'); v.style.display='block'; v.currentTime=0; v.muted=false;
 v.play().then(()=>blog('happy video playing')).catch(e=>{blog('happy play BLOCKED ('+e.name+')'); v.muted=true; v.play();}); v.onended=()=>{v.style.display='none';};}
function blog(m){ fetch('/log?m='+encodeURIComponent(m)); }
async function tick(){const s=await (await fetch('/state')).json();
 if(s.happy && s.happy!==lastHappy){ if(lastHappy!==null){ blog('browser got happy '+s.happy.toFixed(1)); happyPlay(); } lastHappy=s.happy; } else if(lastHappy===null) lastHappy=s.happy||0;
 if(s.panic && s.panic!==lastPanic){ if(lastPanic!==null){ blog('browser got panic '+s.panic.toFixed(1)); panic(); } else blog('page loaded; skipping old panic '+s.panic.toFixed(1)); lastPanic=s.panic; } else if(lastPanic===null) lastPanic=s.panic||0;
 document.getElementById('mode').textContent=s.mode.mode; document.getElementById('mode').className='big mode-'+s.mode.mode;
 document.getElementById('reason').textContent=s.mode.reason||''; document.getElementById('lat').textContent='verdict '+(s.verdict.latency_s||'-')+' s · planner '+(s.mode.latency_s||'-')+' s · loop '+(s.fps||0)+' fps';
 const t=s.threat; document.getElementById('threat').textContent=t.present?('bearing '+t.bearing.toFixed(0)+'° · proximity '+t.proximity.toFixed(2)+' · conf '+t.conf.toFixed(2)+' · track '+t.track):'no cat';
 const v=s.verdict; document.getElementById('verdict').textContent=v.model?((v.see?('“'+v.see+'” '):'')+'· cat '+v.cat_intent+' · open '+v.open_dir+(v.cornered?' · boxed in':'')+'  ('+v.model.split('/').pop()+')'):'none yet';
 document.getElementById('why').textContent=s.paused?'PAUSED: wheels stopped, still watching':(s.command.why||''); showPause(s.paused);}
setInterval(tick,500); tick();
const q0=new URLSearchParams(location.search).get('q'); if(q0){document.getElementById('q').value=q0; ask();}
async function ask(){const q=document.getElementById('q').value; document.getElementById('answer').textContent='thinking…';
 const a=await (await fetch('/ask?q='+encodeURIComponent(q))).json(); document.getElementById('answer').textContent=a.answer;
 document.getElementById('clips').innerHTML=a.clips.map(c=>'<div class="clip"><video src="/clip/'+c.clip.split('/').pop()+'#t=2.5" preload="auto" controls muted></video><div><b>'+new Date(c.t*1000).toLocaleTimeString()+'</b> · score '+c.score+'<br><small>'+c.caption+'</small></div></div>').join('');}
</script></body></html>"""

class Dash:
    def __init__(self, state: dict, memory=None, clips_root="clips", port=8000):
        self.state, self.memory, self.clips_root, self.port = state, memory, clips_root, port
        app = FastAPI()
        @app.get("/", response_class=HTMLResponse)
        def index(): return PAGE.replace("%SIM%", json.dumps(os.environ.get("CATBOT_SIM_URL", "")))
        @app.get("/state")
        def st():
            s = self.state
            def d(o): return {k: v for k, v in (o.__dict__ if hasattr(o, "__dict__") else {}).items()} if o else {}
            return JSONResponse({"threat": d(s.get("threat")), "command": d(s.get("command")), "mode": d(s.get("mode")),
                                 "verdict": d(s.get("verdict")), "fps": s.get("fps", 0), "panic": s.get("panic"), "happy": s.get("happy"), "paused": bool(s.get("paused"))})
        @app.get("/panic.mp4")
        def panic_video():
            here = os.path.dirname(os.path.abspath(__file__))
            local = os.path.join(here, "assets", "local", "panic_run_morty.mp4")   # untracked; see README
            default = local if os.path.exists(local) else os.path.join(here, "assets", "panic.mp4")
            return FileResponse(os.environ.get("PANIC_VIDEO", default), media_type="video/mp4")
        @app.get("/happy.mp4")
        def happy_video():
            here = os.path.dirname(os.path.abspath(__file__))
            return FileResponse(os.environ.get("HAPPY_VIDEO", os.path.join(here, "assets", "local", "happy_hotdog.mp4")), media_type="video/mp4")
        @app.get("/log")
        def blog(m: str = ""):
            os.makedirs("demo_logs", exist_ok=True)
            with open("demo_logs/panic.log", "a") as fh: fh.write("%s BROWSER %s\n" % (time.strftime("%H:%M:%S"), m[:200]))
            return JSONResponse({"ok": True})

        @app.post("/pause")
        def pause():
            self.state["paused"] = not self.state.get("paused"); return JSONResponse({"paused": self.state["paused"]})

        @app.get("/stream")
        def stream():
            def gen():
                while True:
                    jpg = self.state.get("jpg")
                    if jpg: yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg) + jpg + b"\r\n"
                    time.sleep(0.07)
            return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
        @app.get("/ask")
        def ask(q: str):
            if not self.memory: return JSONResponse({"answer": "memory not loaded", "clips": []})
            if not self.memory.rows: return JSONResponse({"answer": "Still describing the first clips; ask again in a moment.", "clips": []})
            return JSONResponse(self.memory.ask(q))
        # ---- relay for the event VM (token-guarded; the VM reaches us through a cloudflared tunnel) ----
        def _guard(token):
            if not os.environ.get("RELAY_TOKEN") or token != os.environ["RELAY_TOKEN"]: raise HTTPException(403)
        @app.get("/relay/pending")
        def relay_pending(token: str = ""):
            _guard(token); return JSONResponse(self.memory.pending() if hasattr(self.memory, "pending") else {"clips": [], "questions": []})
        @app.get("/relay/clip/{name}")
        def relay_clip(name: str, token: str = ""):
            _guard(token); return clip(name)
        @app.post("/relay/ack")
        async def relay_ack(request: Request, token: str = ""):
            _guard(token); self.memory.ack(await request.json()); return JSONResponse({"ok": True})
        @app.post("/relay/answer")
        async def relay_answer(request: Request, token: str = ""):
            _guard(token); self.memory.answer(await request.json()); return JSONResponse({"ok": True})
        @app.get("/relay/script")
        def relay_script(token: str = ""):
            _guard(token); return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vm_relay.py"), media_type="text/x-python")

        @app.get("/clip/{name}")
        def clip(name: str):
            src = os.path.join(self.clips_root, os.path.basename(name))
            out = os.path.join(self.clips_root, "h264", os.path.basename(name))
            if not os.path.exists(out):                      # OpenCV writes mp4v, which browsers will not play
                os.makedirs(os.path.dirname(out), exist_ok=True)
                import subprocess
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=False)
            return FileResponse(out if os.path.exists(out) else src, media_type="video/mp4")
        self.app = app

    def start(self):
        threading.Thread(target=lambda: uvicorn.run(self.app, host="0.0.0.0", port=self.port, log_level="warning"), daemon=True).start()
        return self

def annotate(frame, threat, command, mode):
    f = frame.copy()
    if threat and threat.present and threat.bbox:
        x1, y1, x2, y2 = threat.bbox
        cv2.rectangle(f, (x1, y1), (x2, y2), (80, 220, 80), 2)
        cv2.putText(f, "cat %.2f  %+.0f deg  prox %.2f" % (threat.conf, threat.bearing, threat.proximity), (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 80), 2)
    cv2.putText(f, (command.why if command else ""), (10, f.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(f, "planner: " + (mode.mode if mode else "idle"), (f.shape[1] - 230, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 180, 255), 2)
    ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return jpg.tobytes()
