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
    if "luxury" in s: notes.append("Ease off the sell; let keywords land
