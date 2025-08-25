# main.py
from fastapi import FastAPI, Depends, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
from openai import OpenAI
import base64, tempfile, os, re, secrets

# Optional: local duration check (prevents >90s before hitting OpenAI)
try:
    import soundfile as sf   # already in requirements.txt
except Exception:
    sf = None  # if missing, we’ll fall back to a post-transcription check

# ------------------------------------------------------------------------------
# App + CORS (tight, production-friendly)
# ------------------------------------------------------------------------------
app = FastAPI(title="VO Buddy API", version="0.2.6")

ALLOWED_ORIGINS = [
    "https://voboothbuddy.com",
    "https://www.voboothbuddy.com",
    # Wix preview/editor domains (kept so your editor & preview keep working)
    "https://editor.wix.com",
    "https://manage.wix.com",
    "https://preview.wix.com",
    # Your Render URL (Swagger/testing)
    "https://vo-buddy-api.onrender.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ------------------------------------------------------------------------------
# Basic auth dependency (protects both POST endpoints)
# - We accept either BASIC_AUTH_SECRET or API_KEY (your current Render Secret)
# ------------------------------------------------------------------------------
def _get_secret() -> str:
    return os.getenv("BASIC_AUTH_SECRET") or os.getenv("API_KEY") or ""

def require_basic_auth(request: Request):
    secret = _get_secret()
    if not secret:
        # If you prefer to run “open” temporarily, comment these 3 lines out.
        raise HTTPException(status_code=500, detail="Server auth secret not configured.")
    token = request.headers.get("Authorization", "")
    expected = f"Basic {secret}"
    if not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
class AnalyzeReq(BaseModel):
    audio: Dict[str, str]              # {"base64": "<...>"}  (unused by Wix; kept for Swagger/manual)
    filename: Optional[str] = None
    specs: Optional[str] = ""
    script: Optional[str] = ""
    style_bank: Optional[bool] = False
    deep: Optional[bool] = False       # “Deep Coaching” toggle (from Wix)

class AnalyzeResp(BaseModel):
    notes: List[str]
    altReads: List[str]
    meta: Dict[str, Optional[float]]   # duration, wpm, longest_pause, cta_present

# ------------------------------------------------------------------------------
# Utils for analysis text
# ------------------------------------------------------------------------------
CTA_WORDS = {"call","visit","today","now","shop","learn","sign","download","subscribe","join","try","order","book"}

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def detect_cta(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in CTA_WORDS)

def compute_wpm(word_count: int, duration_s: float) -> Optional[float]:
    if duration_s <= 0 or word_count <= 0:
        return None
    return (word_count / duration_s) * 60.0

def segment_gaps(segments: List[dict]) -> List[float]:
    """Return list of pauses (s) between segments."""
    gaps: List[float] = []
    for i in range(1, len(segments)):
        prev_end = float(segments[i-1].get("end", 0.0))
        cur_start = float(segments[i].get("start", 0.0))
        gap = max(0.0, cur_start - prev_end)
        if gap >= 0.15:
            gaps.append(gap)
    return gaps

def human_notes(metrics: dict, specs: str, script: str, deep: bool) -> List[str]:
    """Greg-style notes. If deep=True, add a couple of coaching frames."""
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

    if deep:
        # Light “deep coaching” adds
        notes.append("Decide: telling or selling? Only punch words when you’re selling; finesse in telling.")
        notes.append("Pick a scene partner. If you can’t see them, they can’t hear it—aim your thoughts at one person.")

    # Dedup + cap
    out, seen = [], set()
    for n in notes:
        if n not in seen:
            out.append(n); seen.add(n)
    return out[:5] if out else ["Clean read—keep it conversational and let the last word land."]

def alt_reads(specs: str, deep: bool) -> List[str]:
    s = (specs or "").lower()
    if "retail" in s or "upbeat" in s:
        alts = [
            "Ride a quicker beat; punch the urgency words; crisp articulation.",
            "Smile on the benefit—keep the sell light and precise."
        ]
    elif "luxury" in s or "aspirational" in s:
        alts = [
            "Slow the cadence a hair; lower the pitch floor; warm restraint.",
            "Let each keyword ring—zero sell on the last word."
        ]
    elif "corporate" in s or "b2b" in s:
        alts = [
            "Neutral warmth; articulate transitions; steady pace.",
            "Lead with clarity—tiny breath before important terms."
        ]
    else:
        alts = [
            "Talk to one person—half-smile on the benefit; downstep the last three words.",
            "Add a micro-pause before the CTA verb and let it ring."
        ]
    if deep:
        alts.append("Break into acts—change one knob each act (tempo, pitch, mic distance) so each beat feels fresh.")
    return alts

# ------------------------------------------------------------------------------
# OpenAI client
# ------------------------------------------------------------------------------
def openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.get("/")
def home():
    return {"ok": True, "service": "vo-buddy-api", "endpoints": ["/analyze-multipart", "/analyze", "/docs", "/healthz"]}

@app.get("/healthz")
def health():
    return {"status": "ok"}

# ---- Multipart (Wix uses this) ------------------------------------------------
@app.post("/analyze-multipart", dependencies=[Depends(require_basic_auth)])
async def analyze_multipart(
    audio_file: UploadFile = File(...),
    specs: str = Form(""),
    script: str = Form(""),
    style_bank: bool = Form(False),
    deep: bool = Form(False),
):
    """
    Accepts multipart upload from the browser, enforces 90s guard locally,
    then reuses the JSON analyzer underneath.
    """
    try:
        raw = await audio_file.read()

        # Write temp file
        suffix = ".wav" if (audio_file.filename or "").lower().endswith(".wav") else ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(raw)
            tmp_path = f.name

        # ---- 90s guard (best-effort, local decode) ----
        duration_local = 0.0
        if sf is not None:
            try:
                with sf.SoundFile(tmp_path) as snd:
                    duration_local = float(len(snd)) / float(snd.samplerate or 1)
            except Exception:
                duration_local = 0.0
        if duration_local and duration_local > 90.0:
            os.remove(tmp_path)
            raise HTTPException(status_code=400, detail="Audio is longer than 90 seconds. Please trim and try again.")

        # Convert to base64 for the JSON pipeline
        b64 = base64.b64encode(raw).decode("utf-8")
        req = AnalyzeReq(
            audio={"base64": b64},
            filename=audio_file.filename or "upload.wav",
            specs=specs,
            script=script,
            style_bank=style_bank,
            deep=deep,
        )
        # Clean up the temp file; JSON endpoint will decode again as needed
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return analyze(req)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"analyze-multipart failed: {e}")

# ---- JSON (Swagger/manual) ----------------------------------------------------
@app.post("/analyze", response_model=AnalyzeResp, dependencies=[Depends(require_basic_auth)])
def analyze(req: AnalyzeReq):
    """
    1) Decode base64 -> temp file
    2) Transcribe with whisper-1 (timestamps on)
    3) Compute duration, wpm, pause stats
    4) Map to Greg-style notes + alt reads
    """
    # --- 1) decode audio to temp file
    b64 = (req.audio or {}).get("base64", "")
    if not b64:
        # No audio? Give helpful, script/specs-based notes anyway.
        metrics = {
            "duration": 0.0,
            "wpm": None,
            "longest_pause": 0.0,
            "cta_present": detect_cta(req.script or ""),
        }
        return AnalyzeResp(
            notes=human_notes(metrics, req.specs or "", req.script or "", bool(req.deep)),
            altReads=alt_reads(req.specs or "", bool(req.deep)),
            meta=metrics,
        )

    raw = base64.b64decode(b64)
    suffix = ".wav" if (req.filename or "").lower().endswith(".wav") else ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(raw)
        tmp_path = f.name

    try:
        # --- 2) transcription (Whisper)
        client = openai_client()
        with open(tmp_path, "rb") as af:
            tr = client.audio.transcriptions.create(
                model="whisper-1",
                file=af,
                response_format="verbose_json",  # includes segments with start/end
                temperature=0.0,
            )

        # Normalize segments (dict list)
        segments: List[dict] = []
        for seg in (tr.segments or []):
            # Some SDKs return objects; we convert to plain dict
            start = getattr(seg, "start", None)
            end = getattr(seg, "end", None)
            text = getattr(seg, "text", None)
            if start is None and isinstance(seg, dict):
                start = seg.get("start")
                end = seg.get("end")
                text = seg.get("text")
            segments.append({"start": float(start or 0.0), "end": float(end or 0.0), "text": text or ""})

        text = getattr(tr, "text", None) or (tr.get("text") if isinstance(tr, dict) else "") or ""
        # Duration (pre-guard if we couldn’t use soundfile)
        if segments:
            duration = float(segments[-1]["end"])
        else:
            duration = float(getattr(tr, "duration", 0.0) or 0.0)

        # --- 90s guard (fallback if we skipped local soundfile)
        if duration and duration > 90.0:
            raise HTTPException(status_code=400, detail="Audio is longer than 90 seconds. Please trim and try again.")

        # --- 3) metrics
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
        notes = human_notes(metrics, req.specs or "", req.script or "", bool(req.deep))
        alts = alt_reads(req.specs or "", bool(req.deep))
        return AnalyzeResp(notes=notes, altReads=alts, meta=metrics)

    except HTTPException:
        raise
    except Exception as e:
        # Bubble up useful details to the client
        raise HTTPException(status_code=502, detail=f"analyze failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
