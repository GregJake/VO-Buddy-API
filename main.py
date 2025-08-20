from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import base64, io, numpy as np

app = FastAPI()

# CORS: keep "*" for testing; later restrict to your domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeReq(BaseModel):
    audio: Dict[str, str]              # {"base64": "..."}
    filename: Optional[str] = None
    specs: Optional[str] = ""
    script: Optional[str] = ""
    style_bank: Optional[bool] = False

def _lazy_load_soundfile():
    try:
        import soundfile as sf  # type: ignore
        return sf
    except Exception:
        return None  # missing libsndfile or other env issue

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
    # rough band energy ratio: 5–8k vs 1–4k
    if len(x) == 0:
        return 0.0
    n = 1 << 15  # shorter FFT to save CPU on free tier
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
    wpm = metrics.get("wpm")  # not available yet (until transcription)
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
        notes.append("Emphasize one clear keyword per sentence.")

    if not notes:
        notes = [
            "Keep it conversational; let edges be a little imperfect.",
            "Let brand/benefit land; soften the final word."
        ]
    return notes[:4]

def alt_reads(specs: str) -> List[str]:
    s = (specs or "").lower()
    if "retail" in s or "upbeat" in s:
        return [
            "Punch urgency words; tighten gaps; ride a quicker beat.",
            "Keep articulation crisp; smile only on benefits."
        ]
    if "luxury" in s or "aspirational" in s:
        return [
            "Slow ~10–15%; let keywords land softly—zero sell.",
            "Lower pitch floor slightly; warm, restrained delivery."
        ]
    if "corporate" in s or "b2b" in s:
        return [
            "Even posture; articulate transitions; steady tone.",
            "Neutral warmth; emphasize clarity over energy."
        ]
    return [
        "Talk to one person—half-smile on the benefit; downstep the last three words.",
        "Add a micro-pause before the CTA verb and let it ring."
    ]

@app.post("/analyze")
def analyze(req: AnalyzeReq):
    # Decode audio with lazy import to avoid startup crashes
    sf = _lazy_load_soundfile()
    x = None
    sr = 16000

    if sf is not None:
        try:
            raw = base64.b64decode(req.audio["base64"])
            data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            if getattr(data, "ndim", 1) > 1:
                import numpy as _np
                data = _np.mean(data, axis=1).astype("float32")
            x = data
        except Exception:
            # Fall back gracefully if decoding fails
            x = None

    metrics = {}
    if x is not None and isinstance(x, np.ndarray):
        d = duration_sec(len(x), sr)
        r = rms(x)
        pauses = estimate_pauses(x, sr, r)
        metrics = {
            "duration": round(d, 2),
            "clip_pct": round(clipping_percent(x), 3),
            "sibilance": round(sibilance_index_fft(x, sr), 3),
            "pauses": pauses,
            # "wpm": None  # add after transcription
        }
    else:
        # Minimal fallback when we can't decode audio (still returns useful text)
        metrics = {
            "duration": 0.0,
            "clip_pct": 0.0,
            "sibilance": 0.0,
            "pauses": [],
        }

    notes = heuristics_to_notes(metrics, req.specs or "", req.script or "")
    alts = alt_reads(req.specs or "")

    return {"notes": notes, "altReads": alts}
