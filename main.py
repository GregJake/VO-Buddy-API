from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import base64, tempfile, os, re, math
from openai import OpenAI

# ------------------- Config -------------------
ALLOWED_ORIGINS = ["https://www.voboothbuddy.com"]  # tighten to your domain
MAX_DURATION = 90.0  # seconds

app = FastAPI(title="VO Buddy API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Auth -------------------
def verify_api_key(x_api_key: str = Header(...)):
    expected = os.getenv("API_KEY")
    if not expected or x_api_key != expected.strip():
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

# ------------------- Models -------------------
class AnalyzeReq(BaseModel):
    audio: Dict[str, str]              # {"base64": "<...>"}
    filename: Optional[str] = None
    specs: Optional[str] = ""
    script: Optional[str] = ""
    style_bank: Optional[bool] = False
    deep: Optional[bool] = False       # deep coaching toggle

class AnalyzeResp(BaseModel):
    notes: List[str]
    altReads: List[str]
    meta: Dict[str, Optional[float]]   # duration, wpm, longest_pause, cta_present

# ------------------- Utils -------------------
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
    gaps = []
    for i in range(1, len(segments)):
        prev_end = segments[i-1].get("end", 0)
        cur_start = segments[i].get("start", 0)
        gap = max(0.0, float(cur_start) - float(prev_end))
        if gap >= 0.15:
            gaps.append(gap)
    return gaps

def human_notes(metrics: dict, specs: str, script: str, deep: bool) -> List[str]:
    notes: List[str] = []
    wpm = metrics.get("wpm")
    longest_pause = metrics.get("longest_pause", 0.0)
    has_cta = metrics.get("cta_present", False)
    duration = metrics.get("duration", 0.0)
    s = (specs or "").lower()
    sc = clean_text(script)

    # Pace
    if wpm:
        if wpm > 170: notes.append("Seemed a bit rushed—give it a breath and let the thought land.")
        elif wpm < 130: notes.append("Reads a touch slow—tighten the beats and keep it moving.")
        else: notes.append("Nice, steady pace—feels natural and easy to follow.")
    else:
        if duration < 8: notes.append("Short take—try a full pass so the idea has room.")

    # Pauses
    if longest_pause > 0.7: notes.append("Trim a little air between phrases so momentum doesn’t stall.")
    elif 0 < longest_pause < 0.2: notes.append("Give it a tiny bit more air between ideas—just enough for a listener blink.")

    # Specs awareness
    if "avoid announcer" in s: notes.append("Keep it talky—no announcer lift on the brand line.")
    if "retail" in s: notes.append("Hit the urgency words and keep the gaps tight.")
    if "luxury" in s: notes.append("Ease off the sell; let keywords land softly with a warm finish.")
    if "corporate" in s: notes.append("Prioritize clarity over hype; even tone, confident posture.")

    # Script awareness
    if not sc:
        notes.append("If you paste the script, I’ll time emphasis and the CTA more precisely.")
    elif has_cta:
        notes.append("Let the CTA breathe—micro-pause before the verb so it rings.")

    # Deep coaching extras
    if deep:
        notes.append("Think about whether you’re *telling* or *selling*—finesse vs. punch.")
        notes.append("Decide if you’re the lead actor or support; let that color your choices.")
        notes.append("Find the subtext—the words behind the words—and layer that in.")
        notes.append("Break the script into acts; vary tempo, pitch, and energy across sections.")

    # De-dup and cap
    seen, out = set(), []
    for n in notes:
        if n not in seen: seen.add(n); out.append(n)
    return out[:6] if out else ["Clean read—keep it conversational and let the last word land."]

def alt_reads(specs: str) -> List[str]:
    s = (specs or "").lower()
    if "retail" in s: return ["Ride a quicker beat; punch urgency words.", "Smile only on the benefit—keep the sell light."]
    if "luxury" in s: return ["Slow the cadence; lower pitch floor.", "Let each keyword ring—zero sell on the last word."]
    if "corporate" in s: return ["Neutral warmth; steady pace.", "Lead with clarity—tiny breath before key terms."]
    return ["Talk to one person—half-smile on the benefit.", "Add a micro-pause before the CTA verb and let it ring."]

def openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key: raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

# ------------------- Routes -------------------
@app.get("/")
def home():
    return {"ok": True, "service": "vo-buddy-api", "endpoints": ["/analyze", "/docs", "/healthz"]}

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalyzeResp, dependencies=[Depends(verify_api_key)])
def analyze(req: AnalyzeReq):
    b64 = req.audio.get("base64", "")
    if not b64:
        metrics = {"duration": 0.0, "wpm": None, "longest_pause": 0.0, "cta_present": detect_cta(req.script or "")}
        return AnalyzeResp(notes=human_notes(metrics, req.specs, req.script, req.deep),
                           altReads=alt_reads(req.specs), meta=metrics)

    raw = base64.b64decode(b64)
    suffix = ".wav" if (req.filename or "").lower().endswith(".wav") else ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(raw); tmp_path = f.name

    try:
        client = openai_client()
        with open(tmp_path, "rb") as af:
            tr = client.audio.transcriptions.create(model="whisper-1", file=af,
                                                    response_format="verbose_json", temperature=0.0)
        segments = tr.segments or []
        text = tr.text or ""
        duration = float(segments[-1].end if segments else getattr(tr, "duration", 0.0))
        if duration > MAX_DURATION:
            raise HTTPException(status_code=400, detail=f"Audio exceeds {MAX_DURATION} seconds.")
        words = len(re.findall(r"\b[\w']+\b", text))
        wpm = compute_wpm(words, duration)
        gaps = segment_gaps([s.model_dump() for s in segments]) if segments else []
        longest_pause = max(gaps) if gaps else 0.0
        cta_present = detect_cta((req.script or "") + " " + text)
        metrics = {"duration": round(duration, 2), "wpm": round(wpm, 1) if wpm else None,
                   "longest_pause": round(longest_pause, 2), "cta_present": bool(cta_present)}
        return AnalyzeResp(notes=human_notes(metrics, req.specs, req.script, req.deep),
                           altReads=alt_reads(req.specs), meta=metrics)
    finally:
        try: os.remove(tmp_path)
        except: pass

@app.post("/analyze-multipart", dependencies=[Depends(verify_api_key)])
async def analyze_multipart(audio_file: UploadFile = File(...), specs: str = Form(""),
                            script: str = Form(""), style_bank: bool = Form(False), deep: bool = Form(False)):
    try:
        raw = await audio_file.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        req = AnalyzeReq(audio={"base64": b64}, filename=audio_file.filename or "upload.wav",
                         specs=specs, script=script, style_bank=style_bank, deep=deep)
        return analyze(req)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"analyze-multipart failed: {e}")
