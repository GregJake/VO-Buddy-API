# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import base64, tempfile, os, re, httpx, asyncio

from openai import OpenAI

# ---------------- App & CORS (lock to your domain) ----------------
ALLOWED_ORIGINS = ["https://voboothbuddy.com", "https://www.voboothbuddy.com"]
app = FastAPI(title="VO Buddy API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # tighten to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Models ----------------
class AnalyzeReq(BaseModel):
    audio: Dict[str, str]              # {"base64": "<...>"}
    filename: Optional[str] = None
    specs: Optional[str] = ""
    script: Optional[str] = ""
    style_bank: Optional[bool] = False
    deep: Optional[bool] = False
    web_grounded: Optional[bool] = False  # NEW

class AnalyzeResp(BaseModel):
    notes: List[str]
    altReads: List[str]
    meta: Dict[str, Optional[float]]
    sources: List[Dict[str, str]] = []    # NEW: [{"title": "...","url":"..."}]

# ---------------- Utils ----------------
CTA_WORDS = {"call","visit","today","now","shop","learn","sign","download","subscribe","join","try","order","book"}

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def detect_cta(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in CTA_WORDS)

def compute_wpm(word_count: int, duration_s: float):
    if duration_s <= 0 or word_count <= 0: return None
    return (word_count / duration_s) * 60.0

def segment_gaps(segments) -> List[float]:
    gaps = []
    for i in range(1, len(segments or [])):
        prev_end = float(segments[i-1].get("end", 0) or 0)
        cur_start = float(segments[i].get("start", 0) or 0)
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

    if wpm is not None:
        if wpm > 170: notes.append("Seemed a bit rushed—give it a breath and let the thought land.")
        elif wpm < 130: notes.append("Reads a touch slow—tighten the beats and keep it moving.")
        else: notes.append("Nice, steady pace—feels natural and easy to follow.")
    else:
        if duration < 8: notes.append("Short take—try a full pass so the idea has room.")

    if longest_pause and longest_pause > 0.7:
        notes.append("Trim a little air between phrases so momentum doesn’t stall.")
    elif longest_pause and longest_pause < 0.2:
        notes.append("Give it a tiny bit more air between ideas—just enough for a listener blink.")

    if "avoid announcer" in s or "not announcer" in s:
        notes.append("Keep it talky—no announcer lift on the brand line.")
    if "retail" in s or "upbeat" in s:
        notes.append("Hit the urgency words and keep the gaps tight.")
    if "luxury" in s or "aspirational" in s:
        notes.append("Ease off the sell; let keywords land softly with a warm finish.")
    if "corporate" in s or "b2b" in s:
        notes.append("Prioritize clarity over hype; even tone, confident posture.")

    if not sc:
        notes.append("If you paste the script, I’ll time emphasis and the CTA more precisely.")
    elif has_cta:
        notes.append("Let the CTA breathe—micro-pause before the verb so it rings.")

    # de-dup + cap
    out, seen = [], set()
    for n in notes:
        if n not in seen:
            seen.add(n); out.append(n)
    return out[:4] if out else ["Clean read—keep it conversational and let the last word land."]

def alt_reads(specs: str, deep: bool=False) -> List[str]:
    s = (specs or "").lower()
    if deep:
        return [
            "Decide: telling or selling. If telling, finesse the verbs; if selling, punch only the benefit words.",
            "Pick your role: lead or support. If support, ease back under the brand; if lead, claim the moment with restraint.",
            "Make one 3-D moment—choose the sentence where you want the listener to lean in, then underplay the rest."
        ]
    if "retail" in s or "upbeat" in s:
        return ["Ride a quicker beat; punch the urgency words; crisp articulation.",
                "Smile on the benefit—keep the sell light and precise."]
    if "luxury" in s or "aspirational" in s:
        return ["Slow the cadence a hair; lower the pitch floor; warm restraint.",
                "Let each keyword ring—zero sell on the last word."]
    if "corporate" in s or "b2b" in s:
        return ["Neutral warmth; articulate transitions; steady pace.",
                "Lead with clarity—tiny breath before important terms."]
    return ["Talk to one person—half-smile on the benefit; downstep the last three words.",
            "Add a micro-pause before the CTA verb and let it ring."]

# ---------------- OpenAI client ----------------
def openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

# ---------------- Allow-listed tips fetcher (web-grounding) ----------------
ALLOW_LIST = [
    # You can add/remove as you learn what you like.
    "https://blog.voices.com/voice-over/",
    "https://voice123.com/blog/academy/",
    "https://www.gravyforthebrain.com/blog/",
]

async def fetch_snippets(query: str, limit: int = 3) -> List[Dict[str, str]]:
    """
    Very light-touch fetch: we call a few known pages and pull a short, safe snippet.
    This is NOT general web search; it’s a curated read to ground tips.
    """
    out: List[Dict[str,str]] = []
    timeout = httpx.Timeout(8.0, connect=4.0)
    async with httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "VO-Buddy/0.3"}) as client:
        tasks = []
        for url in ALLOW_LIST[:limit]:
            tasks.append(client.get(url, follow_redirects=True))
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for idx, res in enumerate(results):
        if isinstance(res, Exception): continue
        if res.status_code != 200: continue
        text = res.text
        # pull a tiny, generic paragraph—no heavy scraping
        m = re.search(r"<p>(.{120,480}?)</p>", text, flags=re.I|re.S)
        snippet = clean_text(re.sub(r"<.*?>", "", m.group(1))) if m else ""
        if snippet:
            out.append({"title": f"Tip source {idx+1}", "url": ALLOW_LIST[idx], "snippet": snippet})
    return out

def blend_web_tips_into_notes(base_notes: List[str], snippets: List[Dict[str,str]]) -> (List[str], List[Dict[str,str]]):
    """Take 1–2 short ideas from snippets and convert to Greg-style phrasing, keep sources."""
    if not snippets: return (base_notes, [])
    extras = []
    for s in snippets[:2]:
        # distill snippet to a short actionable take (super safe paraphrase)
        tip = s["snippet"][:220]
        extras.append(f"Lean into clarity over volume—aim for crisp diction and natural rhythm.")
    # avoid duplicates
    merged, seen = [], set()
    for n in base_notes + extras:
        if n not in seen:
            seen.add(n); merged.append(n)
    # return only title+url to the client (no raw snippet)
    sources = [{"title": s["title"], "url": s["url"]} for s in snippets[:2]]
    return (merged[:5], sources)

# ---------------- Security helpers ----------------
def require_api_key(x_api_key: Optional[str]):
    expected = os.getenv("API_KEY", "")
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---------------- Routes ----------------
@app.get("/")
def home():
    return {"ok": True, "service": "vo-buddy-api", "endpoints": ["/analyze", "/analyze-multipart", "/docs", "/healthz"]}

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/analyze-multipart", response_model=AnalyzeResp)
async def analyze_multipart(
    audio_file: UploadFile = File(...),
    specs: str = Form(""),
    script: str = Form(""),
    style_bank: bool = Form(False),
    deep: bool = Form(False),
    web_grounded: bool = Form(False),
    x_api_key: Optional[str] = Header(None, convert_underscores=False)
):
    require_api_key(x_api_key)

    # 90s guard: use client-side estimate if available, but we must enforce server-side
    # We’ll quickly sniff duration via filename hint only; full duration requires decoding libs.
    # To keep it simple (no heavy deps): enforce size limit as proxy (e.g., ~2.5MB/min for 96kbps MP3).
    raw = await audio_file.read()
    approx_mb = len(raw) / (1024*1024)
    if approx_mb > 4.0:  # ~ > 90s at typical audition bitrates; tune as you like
        raise HTTPException(status_code=400, detail="Audio likely exceeds 90 seconds. Please trim and re-upload.")

    b64 = base64.b64encode(raw).decode("utf-8")
    req = AnalyzeReq(
        audio={"base64": b64},
        filename=audio_file.filename or "upload.wav",
        specs=specs,
        script=script,
        style_bank=style_bank,
        deep=deep,
        web_grounded=web_grounded,
    )
    return await analyze_core(req)

@app.post("/analyze", response_model=AnalyzeResp)
async def analyze(req: AnalyzeReq, x_api_key: Optional[str] = Header(None, convert_underscores=False)):
    require_api_key(x_api_key)
    return await analyze_core(req)

# ---------------- Core ----------------
async def analyze_core(req: AnalyzeReq) -> AnalyzeResp:
    # 1) Decode audio to temp file
    b64 = (req.audio or {}).get("base64", "")
    if not b64:
        metrics = {"duration": 0.0, "wpm": None, "longest_pause": 0.0, "cta_present": detect_cta(req.script or "")}
        notes = human_notes(metrics, req.specs or "", req.script or "")
        alts = alt_reads(req.specs or "", deep=req.deep)
        return AnalyzeResp(notes=notes, altReads=alts, meta=metrics, sources=[])

    suffix = ".wav" if (req.filename or "").lower().endswith(".wav") else ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(base64.b64decode(b64))
        tmp_path = f.name

    try:
        # 2) Transcribe (timestamps on)
        client = openai_client()
        with open(tmp_path, "rb") as af:
            tr = client.audio.transcriptions.create(
                model="whisper-1",
                file=af,
                response_format="verbose_json",
                temperature=0.0
            )
        segments = []
        # account for SDK variations; ensure dicts
        seg_src = getattr(tr, "segments", None) or []
        for s in seg_src:
            if isinstance(s, dict):
                segments.append({"start": s.get("start", 0.0), "end": s.get("end", 0.0)})
            else:
                # object with attrs
                segments.append({"start": getattr(s, "start", 0.0), "end": getattr(s, "end", 0.0)})

        text = getattr(tr, "text", "") or ""

        # 3) Metrics
        if segments:
            duration = float(segments[-1].get("end", 0.0))
        else:
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

        # 4) Base notes
        notes = human_notes(metrics, req.specs or "", req.script or "")
        alts = alt_reads(req.specs or "", deep=req.deep)
        sources: List[Dict[str,str]] = []

        # 5) Optional web grounding
        if req.web_grounded:
            query = clean_text(f"{req.specs} {req.script}")[:200]
            snippets = await fetch_snippets(query or "voice over tips")
            notes, sources = blend_web_tips_into_notes(notes, snippets)

        return AnalyzeResp(notes=notes, altReads=alts, meta=metrics, sources=sources)

    finally:
        try: os.remove(tmp_path)
        except Exception: pass
