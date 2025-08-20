from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, List
import base64, io, numpy as np

import soundfile as sf  # audio decode
# Optional later:
# from faster_whisper import WhisperModel

app = FastAPI()

class AnalyzeReq(BaseModel):
    audio: Dict[str, str]              # {"base64": "..."}
    filename: Optional[str] = None
    specs: Optional[str] = ""
    script: Optional[str] = ""
    style_bank: Optional[bool] = False

def load_audio_from_b64(b64: str):
    raw = base64.b64decode(b64)
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if data.ndim > 1:  # mono-ize
        data = np.mean(data, axis=1)
    return data, sr

def duration_sec(x: np.ndarray, sr: int) -> float:
    return float(len(x) / sr)

def rms(x, frame=1024, hop=512):
    # simple RMS per frame
    frames = []
    for i in range(0, len(x)-frame, hop):
        frames.append(np.sqrt(np.mean(x[i:i+frame]**2) + 1e-12))
    return np.array(frames)

def estimate_pauses(x, sr, rms_arr, thresh=0.02):
    # naive: below threshold = silence
    hop_s = 512 / sr
    silent = rms_arr < thresh
    pauses = []
    i = 0
    while i < len(silent):
        if silent[i]:
            j = i
            while j < len(silent) and silent[j]:
                j += 1
            dur = (j - i) * hop_s
            if dur >= 0.2:  # >=200ms
                pauses.append(dur)
            i = j
        else:
            i += 1
    return pauses

def clipping_percent(x, clip=0.999):
    return float(np.mean(np.abs(x) >= clip) * 100.0)

def sibilance_index(x, sr):
    # very rough band energy ratio: 5–8k vs 1–4k
    spec = np.fft.rfft(x, n=1<<16)
    freqs = np.fft.rfftfreq(1<<16, 1/sr)
    def band(a,b):
        m = (freqs >= a) & (freqs < b)
        return np.sum(np.abs(spec[m])**2) + 1e-9
    hi = band(5000, 8000)
    mid = band(1000, 4000)
    return float(hi / mid)

def heuristics_to_notes(metrics: dict, specs: str, script: str) -> List[str]:
    notes = []
    d = metrics["duration"]
    wpm = metrics.get("wpm")  # may be None early on
    pauses = metrics.get("pauses", [])
    sibil = metrics["sibilance"]
    clip = metrics["clip_pct"]

    # pacing
    if wpm:
        if wpm > 170: notes.append("Seemed a bit rushed—give it a breath.")
        elif wpm < 135: notes.append("Reads a touch slow—tighten the beats.")
    else:
        # fallback without transcript
        if d < 10: notes.append("Keep it steady; let key phrases breathe.")
    # pauses
    if any(p > 0.6 for p in pauses):
        notes.append("Trim a little air between phrases.")
    # sibilance
    if sibil > 0.9:
        notes.append("Light de-ess around 5–7 kHz; keep consonants natural.")
    # clipping
    if clip > 0.1:
        notes.append("Watch peaks—back off input gain slightly.")
    # specs-driven
    if specs and ("avoid announcer" in specs.lower() or "not announcer" in specs.lower()):
        notes.append("Avoid the announcer lift—keep it talky and grounded.")
    if not script:
        notes.append("Emphasize one clear keyword per sentence.")

    # ensure at least 2 helpful lines
    if not notes:
        notes = ["Keep it conversational; let edges be a little imperfect.",
                 "Let brand/benefit land; soften the final word."]
    return notes[:4]

def alt_reads(specs: str) -> List[str]:
    s = (specs or "").lower()
    if "retail" in s or "upbeat" in s:
        return ["Punch urgency words; tighten gaps; ride a quicker beat.",
                "Keep articulation crisp; smile only on benefits."]
    if "luxury" in s or "aspirational" in s:
        return ["Slow ~10–15%; let keywords land softly—zero sell.",
                "Lower pitch floor slightly; warm, restrained delivery."]
    if "corporate" in s or "b2b" in s:
        return ["Even posture; articulate transitions; steady tone.",
                "Neutral warmth; emphasize clarity over energy."]
    return ["Talk to one person—half-smile on the benefit; downstep the last three words.",
            "Add a micro-pause before the CTA verb and let it ring."]

@app.post("/analyze")
def analyze(req: AnalyzeReq):
    x, sr = load_audio_from_b64(req.audio["base64"])
    d = duration_sec(x, sr)
    r = rms(x)
    pauses = estimate_pauses(x, sr, r)

    metrics = {
        "duration": round(d, 2),
        "clip_pct": round(clipping_percent(x), 3),
        "sibilance": round(sibilance_index(x, sr), 3),
        # "wpm": <set later when transcription is added>
        "pauses": pauses
    }

    # TODO (phase 2): transcription -> words, wpm
    # model = WhisperModel("base") ...
    # metrics["wpm"] = ...

    notes = heuristics_to_notes(metrics, req.specs or "", req.script or "")
    alts = alt_reads(req.specs or "")

    return {"notes": notes, "altReads": alts}
