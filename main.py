from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import base64, io
import numpy as np

app = FastAPI(title="VO Buddy API")

# CORS: keep "*" for testing; lock down to your domain later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"ok": True, "service": "vo-buddy-api", "endpoints": ["/analyze", "/docs", "/healthz"]}

@app.get("/healthz")
def health():
    return {"status": "ok"}

class AnalyzeReq(BaseModel):
    audio: Dict[str, str]              # {"base64": "..."} (can be empty for now)
    filename: Optional[str] = None
    specs: Optional[str] = ""
    script: Optional[str] = ""
    style_bank: Optional[bool] = False

# ----- Lightweight helpers (no native/system deps) -----
def duration_sec(n_samples: int, sr: int) -> float:
    return float(n_samples / max(sr, 1))

def rms(x, frame=1024, hop=512):
    frames = []
    N = len(x)
    for i in range(0, max(N - frame, 0), hop):
        frames.append(float(np.sqrt(np.mean(x[i:i+frame]**2) + 1e-12)))
    return np.array(frames)

def estimate_pauses(x, sr, rms_arr, thresh=0.02):
    hop_s = 512 / max(sr, 1)
    silent = rms_arr < thresh
    pauses = []
    i = 0
    L = len(silent)
    while i < L:
        if silent[i]:
            j = i
            while j < L and silent[j]:
                j += 1
            dur = (j - i) * hop_s
            if dur >= 0.2:
                pauses.append(round(dur, 3))
            i = j
        else:
            i += 1
    return pauses

def clipping_percent(x, clip=0.999):
    if len(x) == 0:
        return 0.0
    return float(np.mean(np.abs(x) >= clip) * 100.0)

def sibilance_index_fft(x, sr):
    if len(x) == 0:
        return 0.0
    n = 1 << 15
    spec = np.fft.rfft(x[:n], n=n)
    freqs = np.fft.rfftfreq(n, 1/max(sr, 1))
    def band(a,b):
        m = (freqs >= a) & (freqs < b)
        return float(np.sum(np.abs(spec[m])**2) + 1e-9)
    hi = band(5000, 8000)
    mid = band(1000, 4000)
    return float(hi / mid)

def heuristics_to_notes(metrics: dict, specs: str, script: str) -> List[str]:
    notes = []
    d = metrics.get("duration", 0)
    wpm = metrics.get("wpm")
    pauses = metrics.get("pauses", [])
    sibil = metrics.get("sibilance", 0.0)
    clip = metrics.get("clip_pct", 0.0)

    if wpm is not None:
        if wpm > 170: notes.append("Seemed a bit rushed—give it a breath.")
        elif wpm < 135: notes.append("Reads a touch slow—tighten the beats.")
    else:
        if d < 10:
            notes.append("Keep it steady; let key phrases breathe.")

    if any(p > 0.6 for p in pauses):
        notes.append("Trim a little air between phrases.")

    if sibil > 0.9:
        notes.append("Light de-ess around 5–7 kHz; keep consonants natural.")

    if clip > 0.1:
        notes.append("Watch peaks—back off input gain slightly.")

    if specs and ("avoid announcer" in specs.lower() or "not announcer" in specs.lower()):
        notes.append("Avoid the announcer lift—keep it talky and grounded.")

    if not script:
        notes.append("E
