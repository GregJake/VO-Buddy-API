# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from openai import OpenAI
import base64, tempfile, os, re

# --- Configure your allowed origins ---
ALLOWED_ORIGINS = [
    "https://voboothbuddy.com",
    "https://www.voboothbuddy.com",
    # add your Wix site preview origin if needed, e.g. "https://editor.wix.com"
]

API_KEY_ENV = "API_KEY"  # set this env var in Render

app = FastAPI(title="VO Buddy API", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class AnalyzeReq(BaseModel):
    audio: Dict[str, str]              # {"base64": "..."}
    filename: Optional[str] = None
    specs: Optional[str] = ""
    script: Optional[str] = ""
    style_bank: Optional[bool] = False
    deep: Optional[bool] = False

class AnalyzeResp(BaseModel):
    notes: List[str]
    altReads: List[str]
    meta: Dict[str, Optional[float]]   # duration, wpm, longest_pause, cta_present

# ---------- Utils ----------
CTA_WORDS = {
    "call","visit","today","now","shop","learn","sign","download","subscribe","join","try","order","book"
}

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def detect_cta(text: str) -> bool:
    return any(w in (text or "").lower() for w in CTA_WORDS)

def compute_wpm(word_count: int, duration_s: float) -> Optional[float]:
    if duration_s <= 0 or word_count <= 0:
        return None
    return (word_count / duration_s) * 60.0

def _seg_get(seg: Any, field: str, default: float = 0.0) -> float:
    if isinstance(seg, dict):
        return float(seg.get(field, default) or 0.0)
    return float(getattr(seg, field, default) or 0.0)

def segment_gaps(segments) -> List[float]:
    gaps: List[float] = []
    for i in range(1, len(segments)):
        prev_end = _seg_get(segments[i-1], "end", 0.0)
        cur_start = _seg_get(segments[i], "start", 0.0)
        gap = max(0.0, cur_start - prev_end)
        if gap >= 0.15:
            gaps.append(gap)
    return gaps

def human_notes(metrics: dict, specs: str, script: str) -> List[str]:
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
        notes.append("Let the CTA breathe—micro-pause before the CTA verb so it rings.")

    # De-dup & cap
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
            "Smile only on the benefit—keep the sell light and precise.",
        ]
    if "luxury" in s or "aspirational" in s:
        return [
            "Slow the cadence a hair; lower the pitch floor; warm restraint.",
            "Let each keyword ring—zero sell on the last word.",
        ]
    if "corporate" in s or "b2b" in s:
        return [
            "Neutral warmth; articulate transitions; steady pace.",
            "Lead with clarity—tiny breath before important terms.",
        ]
    return [
        "Talk to one person—half-smile on the benefit; downstep the last three words.",
        "Add a micro-pause before the CTA verb and let it ring.",
    ]

# ---------- Method-guided coaching (paraphrased) ----------
def split_acts_from_text(txt: str) -> List[str]:
    if not txt: return []
    t = clean_text(txt)
    parts = re.split(r"[\.!\?]|—|--", t)
    parts = [p.strip() for p in parts if p and len(p.strip()) > 2]
    if len(parts) <= 1: return parts
    if len(parts) > 3:  return [parts[0], parts[1], " ".join(parts[2:])]
    return parts

def guess_selling_or_telling(specs: str, script: str, transcript: str) -> str:
    t = " ".join([specs or "", script or "", transcript or ""]).lower()
    sell_words = ["sale","now","today","shop","deal","order","limited","save","get","call","visit","sign up","download","subscribe","join","book"]
    score = sum(1 for w in sell_words if w in t)
    return "selling" if score >= 2 else "telling"

def find_brand_and_benefit(text: str) -> Dict[str, str]:
    out = {"brand": "", "benefit": ""}
    if not text: return out
    caps = re.findall(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)?)\b", text)
    if caps: out["brand"] = caps[0][:40]
    m = re.search(r"(for|so you can|so that|helps)\s+([^\.!\?]{4,80})", text, flags=re.I)
    if m: out["benefit"] = m.group(2).strip().rstrip(",;:")
    return out

def method_feedback(specs: str, script: str, transcript: str, max_tips: int = 4) -> List[str]:
    tips: List[str] = []
    mode = guess_selling_or_telling(specs, script, transcript)
    acts = split_acts_from_text(script or transcript)
    brandbits = find_brand_and_benefit(script or transcript)

    tips.append("This leans sales—pop the value words, keep the melody simple, and land the CTA clean." if mode=="selling"
                else "This leans story—finesse the keywords, add a little musicality, and keep it human.")
    tips.append("You're the driver—set the tone early, then ease off so the brand feels effortless."
                if brandbits.get("brand") else
                "Play strong support—guide clearly, let the product/message sit center stage.")
    tips.append("Picture who you're talking to and what's happening—if you can see it, they'll hear it.")
    tips.append("Layer subtext—what are you implying that the script can't say? Let that peek through.")
    tips.append("Give each section a different color—shift pace/pitch slightly between beats so it evolves."
                if len(acts)>=2 else
                "Shape a small arc—start warm, lift on the benefit, settle confident at the end.")
    tips.append("Use simple tools: tiny tempo lift on benefits, soften consonants for warmth, closer mic for intimacy.")
    tips.append("Add a touch of physicality—gesture or posture change to make the words feel 3D.")
    tips.append(f"Choose one payoff line—let \"{brandbits['benefit']}\" ring, then exit clean."
                if brandbits.get("benefit") else
                "Choose one payoff line—let it ring, then leave space so it sticks.")
    tips.append("Trust your choices—paint with words and commit to the vibe you set.")

    seen, out = set(), []
    for t in tips:
        if t not in seen:
            seen.add(t); out.append(t)
        if len(out) >= max_tips: break
    return out

# ---------- Security ----------
def require_api_key(x_api_key: Optional[str]) -> None:
    expected = os.getenv(API_KEY_ENV)
    if not expected:
        # Safety: if API key not set server-side, block all requests
        raise HTTPException(status_code=503, detail="Server misconfigured: API key not set.")
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized: missing or invalid API key.")

# ---------- OpenAI client ----------
def openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

# ---------- Routes ----------
@app.get("/")
def home():
    return {"ok": True, "service": "vo-buddy-api", "endpoints": ["/analyze", "/analyze-multipart", "/docs", "/healthz"]}

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalyzeResp)
def analyze(req: AnalyzeReq, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)

    # 1) Decode base64 -> temp file
    b64 = (req.audio or {}).get("base64", "")
    if not b64:
        metrics = {"duration": 0.0, "wpm": None, "longest_pause": 0.0, "cta_present": detect_cta(req.script or "")}
        transcript_text = ""
        notes_basic = human_notes(metrics, req.specs or "", req.script or "")
        method_notes = method_feedback(req.specs or "", req.script or "", transcript_text, max_tips=(7 if req.deep else 4))
        blended = []
        for n in notes_basic + method_notes:
            if n not in blended: blended.append(n)
        notes_out = blended[: (8 if req.deep else 6)]
        return AnalyzeResp(notes=notes_out, altReads=alt_reads(req.specs or ""), meta=metrics)

    raw = base64.b64decode(b64)
    suffix = ".wav" if (req.filename or "").lower().endswith(".wav") else ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(raw); tmp_path = f.name

    try:
        # 2) Transcribe
        client = openai_client()
        with open(tmp_path, "rb") as af:
            try:
                tr = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=af,
                    response_format="verbose_json",
                    temperature=0.0,
                )
            except Exception as e:
                msg = str(e)
                if "insufficient_quota" in msg or "quota" in msg or "429" in msg:
                    raise HTTPException(status_code=402, detail="Transcription unavailable: OpenAI quota exhausted.")
                raise HTTPException(status_code=502, detail=f"Transcription failed: {msg}")

        segments = getattr(tr, "segments", None) or []
        text = getattr(tr, "text", "") or ""

        # 3) Metrics
        if segments:
            duration = _seg_get(segments[-1], "end", 0.0)
        else:
            duration = float(getattr(tr, "duration", 0.0) or 0.0)

        # --- 90-second guard ---
        if duration and duration > 90.0:
            raise HTTPException(status_code=400, detail="Clip is over 90 seconds—trim and re-upload.")

        words = len(re.findall(r"\b[\w']+\b", text))
        wpm = compute_wpm(words, duration) if duration else None
        gaps = segment_gaps(segments)
        longest_pause = max(gaps) if gaps else 0.0
        cta_present = detect_cta((req.script or "") + " " + text)

        metrics = {
            "duration": round(duration, 2) if duration else 0.0,
            "wpm": round(wpm, 1) if wpm else None,
            "longest_pause": round(longest_pause, 2) if longest_pause else 0.0,
            "cta_present": bool(cta_present),
        }

        # 4) Feedback
        notes_basic = human_notes(metrics, req.specs or "", req.script or "")
        method_notes = method_feedback(req.specs or "", req.script or "", text or "", max_tips=(7 if req.deep else 4))
        blended = []
        for n in notes_basic + method_notes:
            if n not in blended: blended.append(n)
        notes_out = blended[: (8 if req.deep else 6)]

        return AnalyzeResp(notes=notes_out, altReads=alt_reads(req.specs or ""), meta=metrics)
    finally:
        try: os.remove(tmp_path)
        except Exception: pass

@app.post("/analyze-multipart")
async def analyze_multipart(
    audio_file: UploadFile = File(...),
    specs: str = Form(""),
    script: str = Form(""),
    style_bank: bool = Form(False),
    deep: bool = Form(False),
    x_api_key: Optional[str] = Header(None),
):
    require_api_key(x_api_key)
    try:
        raw = await audio_file.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        req = AnalyzeReq(
            audio={"base64": b64},
            filename=audio_file.filename or "upload.wav",
            specs=specs, script=script,
            style_bank=style_bank,
            deep=deep,
        )
        return analyze(req, x_api_key=x_api_key)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"analyze-multipart failed: {e}")
