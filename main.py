from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List

# ---------- App ----------
app = FastAPI(title="VO Buddy API", version="0.1.0")

# CORS (wide open for testing; later restrict to your domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # e.g., ["https://your-wix-domain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class AnalyzeReq(BaseModel):
    audio: Dict[str, str]              # {"base64": "<...>"}  // we ignore it for now
    filename: Optional[str] = None
    specs: Optional[str] = ""
    script: Optional[str] = ""
    style_bank: Optional[bool] = False

class AnalyzeResp(BaseModel):
    notes: List[str]
    altReads: List[str]

# ---------- Friendly routes ----------
@app.get("/")
def home():
    return {"ok": True, "service": "vo-buddy-api", "endpoints": ["/analyze", "/docs", "/healthz"]}

@app.get("/healthz")
def health():
    return {"status": "ok"}

# ---------- Analyze (JSON) ----------
@app.post("/analyze", response_model=AnalyzeResp)
def analyze(req: AnalyzeReq):
    """
    MVP: return practical notes + alt reads using only specs/script.
    (We ignore the audio for now so the server is rock-solid.)
    """

    s = (req.specs or "").lower()
    script = (req.script or "").strip()

    # Base notes everyone finds helpful
    notes = [
        "Keep it conversational; let the edges be a little imperfect.",
        "Let the brand/benefit land—soften the final word.",
    ]

    # Specs-driven tweaks (simple heuristics)
    if "avoid announcer" in s or "not announcer" in s:
        notes.append("Avoid the announcer lift—keep it talky and grounded.")
    if "retail" in s or "upbeat" in s:
        notes.append("Punch urgency words; keep the gaps tight.")
    if "luxury" in s or "aspirational" in s:
        notes.append("Ease off the sell; slow slightly and land keywords softly.")
    if "corporate" in s or "b2b" in s:
        notes.append("Prioritize clarity over energy; steady tone between beats.")

    # Script-aware hint
    if not script:
        notes.append("If you paste the script, I’ll tailor emphasis timing and CTA advice.")

    # Keep it tight (max 4)
    notes = notes[:4]

    # Alt read suggestions (2–3)
    if "retail" in s or "upbeat" in s:
        alts = [
            "Ride a quicker beat; hit urgency words; crisp articulation.",
            "Smile only on the benefit; keep sales pressure light.",
        ]
    elif "luxury" in s or "aspirational" in s:
        alts = [
            "Slow ~10–15%; lower pitch floor; relaxed, warm authority.",
            "Let each keyword ring; zero sell on the last word.",
        ]
    elif "corporate" in s or "b2b" in s:
        alts = [
            "Neutral warmth; articulate transitions; steady pace.",
            "Focus on clarity; tiny breath before important terms.",
        ]
    else:
        alts = [
            "Talk to one person—half-smile on the benefit; downstep the last three words.",
            "Add a micro-pause before the CTA verb and let it ring.",
        ]

    return AnalyzeResp(notes=notes, altReads=alts)
