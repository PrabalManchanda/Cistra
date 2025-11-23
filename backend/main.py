import os
import json
from typing import Optional, Dict, Any

import joblib
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
import numpy as np
import pandas as pd

# -------------------------------
# ENV + APP SETUP
# -------------------------------
load_dotenv()
app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing in .env")

gemini = genai.Client(api_key=GEMINI_API_KEY)

# Session memory (per session_id)
session_memory = {}

# -------------------------------
# LOAD ML MODELS
# -------------------------------
MODEL1_PATH = "../model/model2-rf-weighted/rf_weighted_model.pkl"
SCALER1_PATH = "../model/model2-rf-weighted/robust_scaler_weighted.pkl"

ml_model = joblib.load(MODEL1_PATH)
scaler = joblib.load(SCALER1_PATH)


# -------------------------------
# Request + Response Models
# -------------------------------
class ConversationRequest(BaseModel):
    session_id: str
    message: str


class ConversationResponse(BaseModel):
    reply: str
    city: Optional[str] = None
    kids: Optional[int] = None
    housing: Optional[str] = None
    score: Optional[float] = None
    monthly_estimate: Optional[int] = None
    range_low: Optional[int] = None
    range_high: Optional[int] = None


# -------------------------------
# HELPERS
# -------------------------------

def clean_json(raw: str) -> dict:
    """Remove markdown ```json wrappers and parse JSON."""
    cleaned = (
        raw.replace("```json", "")
           .replace("```", "")
           .strip()
    )
    print("Cleaned response:", cleaned)

    try:
        return json.loads(cleaned)
    except Exception as e:
        print("JSON parse error:", e)
        return {"city": None, "kids": None, "housing": None}


def extract_info_with_gemini(text: str) -> Dict[str, Any]:
    """Extract city, kids, housing from user message."""

    prompt = f"""
    You MUST return ONLY JSON. No sentences.

    The JSON must contain:
    - city (string or null)
    - kids (int or null)
    - housing ("rent", "buy", or null)

    USER: "{text}"

    Example:
    {{
        "city": "Toronto",
        "kids": 2,
        "housing": "rent"
    }}

    Output ONLY JSON:
    """

    resp = gemini.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    raw = resp.text.strip()
    print("Gemini raw:", raw)

    data = clean_json(raw)

    # Type fix
    city = data.get("city")
    kids = data.get("kids")
    housing = data.get("housing")

    if isinstance(kids, str):
        try: kids = int(kids)
        except: kids = None

    if housing:
        housing = housing.lower()
        if housing not in ("rent", "buy"):
            housing = None

    return {"city": city, "kids": kids, "housing": housing}


def build_features(city: str, kids: int, housing: Optional[str]) -> np.ndarray:
    """
    Your ML model expects 6 normalized inputs.
    But city-specific norms are NOT available.

    For now, we let Gemini estimate norms per city.
    """

    prompt = f"""
    Give normalized cost-of-living category values (0–1) for:

    City: "{city}"

    Categories:
    - housing_norm
    - food_norm
    - restaurants_norm
    - transport_norm
    - internet_utils_norm
    - lifestyle_norm

    Output ONLY JSON like:
    {{
        "housing_norm": 0.58,
        "food_norm": 0.67,
        "restaurants_norm": 0.70,
        "transport_norm": 0.66,
        "internet_utils_norm": 0.53,
        "lifestyle_norm": 0.57
    }}
    """

    resp = gemini.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    raw = resp.text.strip()
    data = clean_json(raw)

    # extract
    h = float(data.get("housing_norm", 0.5))
    f = float(data.get("food_norm", 0.5))
    r = float(data.get("restaurants_norm", 0.5))
    t = float(data.get("transport_norm", 0.5))
    i = float(data.get("internet_utils_norm", 0.5))
    l = float(data.get("lifestyle_norm", 0.5))

    # Adjust for kids + housing
    if kids and kids > 0:
        f += 0.05 * kids
        t += 0.03 * kids

    if housing == "buy":
        h += 0.10
    elif housing == "rent":
        h += 0.03

    # Clip values
    vals = [h, f, r, t, i, l]
    vals = [max(0, min(1, v)) for v in vals]

    return np.array(vals, dtype=float)


def run_models(city: str, kids: int, housing: str) -> dict:
    """Build features → scale → predict."""
    feats = build_features(city, kids, housing).reshape(1, -1)
    scaled = scaler.transform(feats)
    score = float(ml_model.predict(scaled)[0])

    return {"score": score}


def score_to_cost(score: float):
    base = (score * 1100) + 2000
    monthly = round(base)
    return {
        "score": round(score, 4),
        "monthly": monthly,
        "range_low": round(monthly * 0.85),
        "range_high": round(monthly * 1.25)
    }


def build_gemini_reply(user_msg: str, city: str, kids, housing, preds: dict) -> str:
    score = preds["score"]
    monthly = score_to_cost(score)

    prompt = f"""
    User message: "{user_msg}"

    City: {city}
    Kids: {kids}
    Housing: {housing}
    Estimated monthly cost: {monthly['monthly']} CAD

    Write a friendly 3–5 sentence explanation.
    DO NOT mention machine learning or models.
    """

    resp = gemini.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    return resp.text


# -------------------------------
# MAIN ENDPOINT
# -------------------------------
@app.post("/ai-chat", response_model=ConversationResponse)
async def ai_chat(req: ConversationRequest):

    session_id = req.session_id
    user_msg = req.message

    # Create session if not exists
    if session_id not in session_memory:
        session_memory[session_id] = {"city": None, "kids": None, "housing": None}

    state = session_memory[session_id]

    # Extract info
    info = extract_info_with_gemini(user_msg)

    if info["city"]:
        state["city"] = info["city"]

    if info["kids"] is not None:
        state["kids"] = info["kids"]

    if info["housing"]:
        state["housing"] = info["housing"]

    city = state["city"]
    kids = state["kids"]
    housing = state["housing"]

    # Ask again if city missing
    if city is None:
        return ConversationResponse(
            reply="Sure! I can help with cost of living. Which city are you interested in?",
            city=None, kids=kids, housing=housing
        )

    # Run ML prediction
    preds = run_models(city, kids, housing)
    cost = score_to_cost(preds["score"])

    # Build human message
    reply = build_gemini_reply(user_msg, city, kids, housing, preds)

    return ConversationResponse(
        reply=reply,
        city=city,
        kids=kids,
        housing=housing,
        score=cost["score"],
        monthly_estimate=cost["monthly"],
        range_low=cost["range_low"],
        range_high=cost["range_high"]
    )
