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
session_memory: Dict[str, Dict[str, Any]] = {}

# -------------------------------
# LOAD ML MODELS (BASE MODEL ONLY)
# -------------------------------
MODEL1_PATH = "../model/model2-rf-weighted/rf_weighted_model.pkl"
SCALER1_PATH = "../model/model2-rf-weighted/robust_scaler_weighted.pkl"

ml_model = joblib.load(MODEL1_PATH)
scaler = joblib.load(SCALER1_PATH)

# --- OPTIONAL: weighted model (kept but commented out) ---
# MODEL2_PATH = "../model/model2-rf-weighted/rf_weighted_model.pkl"
# SCALER2_PATH = "../model/model2-rf-weighted/robust_scaler_weighted.pkl"
# weighted_model = joblib.load(MODEL2_PATH)
# weighted_scaler = joblib.load(SCALER2_PATH)

# -------------------------------
# LOAD CITY NORMS (IF AVAILABLE)
# -------------------------------
# Expect columns:
# City, housing_norm, food_norm, restaurants_norm,
# transport_norm, internet_utils_norm, lifestyle_norm

CITY_NORMS_PATHS = [
    "../dataset/city_norm_values.csv",
    "../model/city_norm_values.csv",
]

city_norms_df = None
for path in CITY_NORMS_PATHS:
    if os.path.exists(path):
        city_norms_df = pd.read_csv(path)
        city_norms_df["City"] = city_norms_df["City"].astype(str).str.strip()
        print(f"Loaded city norms from: {path}")
        break

if city_norms_df is None:
    print("⚠️ No city_norm_values.csv found. Will use Gemini for norms.")


# -------------------------------
# Request + Response Models
# -------------------------------
class ConversationRequest(BaseModel):
    session_id: str
    message: str


class ConversationResponse(BaseModel):
    session_id: str
    reply: str
    city: Optional[str] = None
    kids: Optional[int] = None
    housing: Optional[str] = None
    cars: Optional[int] = None
    score: Optional[float] = None
    monthly_estimate: Optional[int] = None
    range_low: Optional[int] = None
    range_high: Optional[int] = None


# -------------------------------
# JSON Helper
# -------------------------------
def clean_json(raw: str) -> dict:
    """Remove ```json ... ``` wrappers and parse JSON safely."""
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
        return {}


# -------------------------------
# 1) Extract city / kids / housing / cars
# -------------------------------
def extract_info_with_gemini(text: str) -> Dict[str, Any]:
    """
    Use Gemini to extract:
      - city (string or null)
      - kids (int or null)
      - housing ("rent" / "buy" / null)
      - cars (int or null)
    """

    prompt = f"""
    You MUST return ONLY JSON. No sentences.

    The JSON must contain exactly:
    - city (string or null)
    - kids (integer or null)
    - housing ("rent", "buy", or null)
    - cars (integer or null)

    USER MESSAGE: "{text}"

    Example:
    {{
        "city": "Toronto, Canada",
        "kids": 2,
        "housing": "rent",
        "cars": 1
    }}

    Output ONLY JSON:
    """

    resp = gemini.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    raw = resp.text.strip()
    print("Gemini raw (extract):", raw)

    data = clean_json(raw)

    city = data.get("city")
    kids = data.get("kids")
    housing = data.get("housing")
    cars = data.get("cars")

    # Type fixes
    if isinstance(kids, str):
        try:
            kids = int(kids)
        except Exception:
            kids = None

    if isinstance(cars, str):
        try:
            cars = int(cars)
        except Exception:
            cars = None

    if housing:
        housing = str(housing).lower()
        if housing not in ("rent", "buy"):
            housing = None

    return {
        "city": city,
        "kids": kids,
        "housing": housing,
        "cars": cars,
    }


# -------------------------------
# 2) Build base city features (6 norms)
# -------------------------------
def build_city_features(city: str) -> np.ndarray:
    """
    1) If city exists in CSV, use its 6 norm values.
    2) Otherwise, ask Gemini to approximate them (fallback).
    """

    # --- Try CSV first ---
    if city_norms_df is not None:
        mask = city_norms_df["City"].str.lower() == city.strip().lower()
        row = city_norms_df[mask]
        if not row.empty:
            base = row.iloc[0]
            vals = [
                float(base["housing_norm"]),
                float(base["food_norm"]),
                float(base["restaurants_norm"]),
                float(base["transport_norm"]),
                float(base["internet_utils_norm"]),
                float(base["lifestyle_norm"]),
            ]
            return np.array(vals, dtype=float)

    # --- Fallback to Gemini if city not found or CSV doesn't exist ---
    prompt = f"""
    We have 6 normalized cost-of-living values (0 to 1) for a city:

    - housing_norm
    - food_norm
    - restaurants_norm
    - transport_norm
    - internet_utils_norm
    - lifestyle_norm

    0 means very cheap, 1 means extremely expensive in that category.

    City: "{city}"

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
    print("Gemini raw (norms):", raw)
    data = clean_json(raw)

    h = float(data.get("housing_norm", 0.5))
    f = float(data.get("food_norm", 0.5))
    r = float(data.get("restaurants_norm", 0.5))
    t = float(data.get("transport_norm", 0.5))
    i = float(data.get("internet_utils_norm", 0.5))
    l = float(data.get("lifestyle_norm", 0.5))

    vals = [h, f, r, t, i, l]
    vals = [max(0, min(1, v)) for v in vals]
    return np.array(vals, dtype=float)


# -------------------------------
# 3) Run base ML model on city
# -------------------------------
def run_base_model(city: str) -> Dict[str, float]:
    feats = build_city_features(city).reshape(1, -1)
    scaled = scaler.transform(feats)
    score = float(ml_model.predict(scaled)[0])

    result = {
        "base_score": score,
    }

    # --- OPTIONAL second model (commented out) ---
    # feats2 = feats
    # scaled2 = weighted_scaler.transform(feats2)
    # weighted_score = float(weighted_model.predict(scaled2)[0])
    # result["weighted_score"] = weighted_score

    return result


# -------------------------------
# 4) Convert score → base monthly cost
# -------------------------------
def score_to_cost(score: float) -> Dict[str, float]:
    min=1000
    max =5000
    """
    Convert the ML score into a base monthly estimate.
    (You can tweak this mapping as you like.)
    """
    base = min+(score*(max - min))
    monthly = round(base)
    return {
        "score": round(score, 4),
        "monthly": monthly,
        "range_low": round(monthly * 0.85),
        "range_high": round(monthly * 1.25),
    }


# -------------------------------
# 5) Option C: manual rules + (optional) Gemini fallback
# -------------------------------
def adjust_monthly_with_rules(
    city: str,
    base_monthly: int,
    kids: Optional[int],
    housing: Optional[str],
    cars: Optional[int],
) -> int:
    """
    Option C:
    - Start from base_monthly (1 adult, neutral situation).
    - Apply manual multipliers for kids, housing, cars.
    - If we still have almost no info, we could call Gemini to refine (optional).
    """

    multiplier = 1.0

    # Kids → add ~20% per child, capped
    if kids is not None and kids > 0:
        extra_kids = min(0.20 * kids, 0.60)  # cap at +60%
        multiplier += extra_kids

    # Housing → assume base is closer to 'baseline rent'
    if housing == "rent":
        multiplier += 0.05  # slight bump vs base
    elif housing == "buy":
        multiplier += 0.35  # buying is significantly more

    # Cars → ~18% per car, capped
    if cars is not None and cars > 0:
        extra_cars = min(0.18 * cars, 0.54)  # cap at +54%
        multiplier += extra_cars

    adjusted = round(base_monthly * multiplier)

    # Optional: if we want Gemini to refine *further* when info is thin,
    # we could add a fallback here (kept simple for now).

    return adjusted


# -------------------------------
# 6) Build final human reply with Gemini
# -------------------------------
def build_gemini_reply(
    user_msg: str,
    city: str,
    kids: Optional[int],
    housing: Optional[str],
    cars: Optional[int],
    base_monthly: int,
    final_monthly: int,
) -> str:
    """
    Ask Gemini to write a friendly explanation.
    We do NOT mention ML or internal details.
    """

    kids_text = "no children mentioned"
    if kids is not None:
        kids_text = f"{kids} kid(s)"

    housing_text = "housing type not specified"
    if housing == "rent":
        housing_text = "planning to rent"
    elif housing == "buy":
        housing_text = "planning to buy a home"

    cars_text = "no cars mentioned"
    if cars is not None:
        cars_text = f"{cars} car(s)"

    prompt = f"""
    User message: "{user_msg}"

    City: {city}
    Family: {kids_text}
    Housing: {housing_text}
    Cars: {cars_text}

    A baseline monthly cost estimate for a single person in this city is about {base_monthly} CAD.
    After adjusting for kids, housing choice, and cars, the estimated monthly cost is about {final_monthly} CAD.

    Write a friendly 3–6 sentence explanation:
    - Confirm the city and talk about whether it's relatively cheap, moderate, or expensive.
    - Explain in simple terms why the cost changes when they have kids, cars, or are buying vs renting.
    - Give a realistic but soft range (e.g. "depending on lifestyle it could be a bit lower or higher").
    - Suggest what extra info (like salary range, whether both parents work, etc.) could refine the estimate.
    - Do NOT mention machine learning, models, formulas, or that another AI helped.
    """

    resp = gemini.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
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
        session_memory[session_id] = {
            "city": None,
            "kids": None,
            "housing": None,
            "cars": None,
        }

    state = session_memory[session_id]

    # 1) Extract structured info from this message
    info = extract_info_with_gemini(user_msg)

    if info.get("city"):
        state["city"] = info["city"]

    if info.get("kids") is not None:
        state["kids"] = info["kids"]

    if info.get("housing"):
        state["housing"] = info["housing"]

    if info.get("cars") is not None:
        state["cars"] = info["cars"]

    city = state["city"]
    kids = state["kids"]
    housing = state["housing"]
    cars = state["cars"]

    # If city still missing → ask user
    if city is None:
        return ConversationResponse(
            session_id=session_id,
            reply="Sure! I can help with cost of living. Which city are you interested in?",
            city=None,
            kids=kids,
            housing=housing,
            cars=cars,
        )

    # 2) Base model prediction for this city
    base_preds = run_base_model(city)
    base_score = base_preds["base_score"]
    base_cost = score_to_cost(base_score)
    base_monthly = base_cost["monthly"]

    # 3) Adjust monthly cost using Option C (manual rules)
    final_monthly = adjust_monthly_with_rules(
        city=city,
        base_monthly=base_monthly,
        kids=kids,
        housing=housing,
        cars=cars,
    )

    # Recompute range around final (keep 15–25% band)
    final_range_low = round(final_monthly * 0.85)
    final_range_high = round(final_monthly * 1.25)

    # 4) Build human explanation with Gemini
    reply = build_gemini_reply(
        user_msg=user_msg,
        city=city,
        kids=kids,
        housing=housing,
        cars=cars,
        base_monthly=base_monthly,
        final_monthly=final_monthly,
    )

    return ConversationResponse(
        session_id=session_id,
        reply=reply,
        city=city,
        kids=kids,
        housing=housing,
        cars=cars,
        score=base_cost["score"],          # base model score
        monthly_estimate=final_monthly,    # adjusted monthly
        range_low=final_range_low,
        range_high=final_range_high,
    )
