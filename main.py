# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import base64, tempfile, os, math, re
from openai import OpenAI

app = FastAPI(title="VO Buddy API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your domain once stable
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class AnalyzeReq(BaseModel):
    audio: Dict[str, str]              # {"base64": "<...>"}
    filename: Optional[str] = None
    specs: Optional[str] = ""
    script: Optional[str] = ""
    style_bank: Optional[bool] = False

class AnalyzeResp(BaseModel):
    notes: List[str]
    altReads: List[str]
    meta: Dict[str, Optional[float]]   # for debugging/QA: duration, wpm, longest_pause

# ---------- Utils ----------
CTA_WORDS = {"call","visit","today","now","shop","learn","sign","download","subscribe","join","try","order","book"}

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def detect_cta(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in CTA_WORDS)

def compute_wpm(word_count: int, duration_s: float) -> Optional[float]:
    if duration_s <= 0 or word_count <= 0:
        return None
    return (word_count / duration_s) * 60.0

def segment_gaps(segments) -> List[float]:
    """Return list of pauses between segments in seconds."""
    gaps = []
    for i in range(1, len(segments)):
        prev_end = segments[i-1].get("end", 0)
        cur_start = segments[i].get("start", 0)
        gap = max(0.0, float(cur_start) - float(prev_end))
        if gap >= 0.15:  # ignore micro gaps
            gaps.append(gap)
    return gaps

def human_notes(metrics: dict, specs: str, script: str) -> List[str]:
    """Map metrics/specs/script to Greg-style notes (no hard numbers)."""
    notes: List[str] = []
    wpm = metrics.get("wpm")
    longest_pause = metrics.get("longest_pause", 0.0)
    has_cta = metrics.get("cta_present", False)
    duration = metrics.get("duration", 0.0)

    s = (specs or "").lower()
    sc = clean_text(script)

    # Pace
    if wpm is not None:
        if wpm > 170:
            notes.append("Seemed a bit rushed—give it a breath and let the thought land.")
        elif wpm < 130:
            notes.append("Reads a touch slow—tighten the beats and keep it moving.")
        else:
            notes.append("Nice, steady pace—feels natural and easy to follow.")
    else:
        if duration < 8:
            notes.append("Short take—try a full pass so the idea has room.")

    # Pauses
    if longest_pause and longest_pause > 0.7:
        notes.append("Trim a little air between phrases so momentum doesn’t stall.")
    elif longest_pause and longest_pause < 0.2:
        notes.append("Give it a tiny bit more air between ideas—just enough for a listener blink.")

    # Specs awareness
    if "avoid announcer" in s or "not announcer" in s:
        notes.append("Keep it talky—no announcer lift on the brand line.")
    if "retail" in s or "upbeat" in s:
        notes.append("Hit the urgency words and keep the gaps tight.")
    if "luxury" in s or "aspirational" in s:
        notes.append("Ease off the sell; let keywords land softly with a warm finish.")
    if "corporate" in s or "b2b" in s:
        notes.append("Prioritize clarity over hype; even tone, confident posture.")

    # Script awareness
    if not sc:
        notes.append("If you paste the script, I’ll time emphasis and the CTA more precisely.")
    elif has_cta:
        notes.append("Let the CTA breathe—micro-pause before the verb so it rings.")

    # De-dup and cap
    seen, out = set(), []
    for n in notes:
        if n not in seen:
            seen.add(n); out.append(n)
    return out[:4] if out else ["Clean read—keep it conversational and let the last word land."]

def alt_reads(specs: str) -> List[str]:
    s = (specs or "").lower()
    if "retail" in s or "upbeat" in s:
        return [
            "Ride a quicker beat; punch the urgency words; crisp articulation.",
            "Smile only on the benefit—keep the sell light and precise."
        ]
    if "luxury" in s or "aspirational" in s:
        return [
            "Slow the cadence a hair; lower the pitch floor; warm restraint.",
            "Let each keyword ring—zero sell on the last word."
        ]
    if "corporate" in s or "b2b" in s:
        return [
            "Neutral warmth; articulate transitions; steady pace.",
            "Lead with clarity—tiny breath before important terms."
        ]
    return [
        "Talk to one person—half-smile on the benefit; downstep the last three words.",
        "Add a micro-pause before the CTA verb and let it ring."
    ]

# ---------- OpenAI client ----------
def openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

# ---------- Routes ----------
@app.get("/")
def home():
    return {"ok": True, "service": "vo-buddy-api", "endpoints": ["/analyze", "/docs", "/healthz"]}

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalyzeResp)
def analyze(req: AnalyzeReq):
    """
    1) Decode base64 -> temp file
    2) Transcribe with whisper-1 (timestamps on)
    3) Compute duration, wpm, pause stats
    4) Map to Greg-style notes + alt reads
    """
    # --- 1) decode audio to temp file
    b64 = req.audio.get("base64", "")
    if not b64:
        # Graceful fallback: no audio posted
        metrics = {"duration": 0.0, "wpm": None, "longest_pause": 0.0, "cta_present": detect_cta(req.script or "")}
        return AnalyzeResp(notes=human_notes(metrics, req.specs or "", req.script or ""),
                           altReads=alt_reads(req.specs or ""),
                           meta=metrics)

    raw = base64.b64decode(b64)
    suffix = ".wav" if (req.filename or "").lower().endswith(".wav") else ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(raw)
        tmp_path = f.name

    try:
        # --- 2) transcription
        client = openai_client()
        with open(tmp_path, "rb") as af:
            tr = client.audio.transcriptions.create(
                model="whisper-1",
                file=af,
                response_format="verbose_json",   # returns segments with start/end
                temperature=0.0
            )

        # Fallback handling if fields missing
        segments = tr.segments or []
        text = tr.text or ""

        # --- 3) metrics from segments
        if segments:
            duration = float(segments[-1].get("end", 0.0))
        else:
            # If no segments, try duration from whisper metadata or fall back to 0
            duration = float(getattr(tr, "duration", 0.0) or 0.0)

        words = len(re.findall(r"\b[\w']+\b", text))
        wpm = compute_wpm(words, duration)
        gaps = segment_gaps(segments)
        longest_pause = max(gaps) if gaps else 0.0
        cta_present = detect_cta((req.script or "") + " " + text)

        metrics = {
            "duration": round(duration, 2) if duration else 0.0,
            "wpm": round(wpm, 1) if wpm else None,
            "longest_pause": round(longest_pause, 2) if longest_pause else 0.0,
            "cta_present": bool(cta_present),
        }

        # --- 4) map to human notes
        notes = human_notes(metrics, req.specs or "", req.script or "")
        alts = alt_reads(req.specs or "")
        return AnalyzeResp(notes=notes, altReads=alts, meta=metrics)

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
