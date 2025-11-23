import os
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai

load_dotenv()

# --------------------------
# Load Gemini
# --------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini = genai.Client(api_key=GEMINI_API_KEY)

# --------------------------
# Load ML Model + Scaler
# --------------------------
MODEL_PATH = "model/model1-rf/rf_median_model.pkl"
SCALER_PATH = "model/model1-rf/robust_scaler.pkl"

ml_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --------------------------
# FastAPI App
# --------------------------
app = FastAPI()


# --------------------------
# Request/Response Schemas
# --------------------------
class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    reply: str


class PredictRequest(BaseModel):
    features: list[float]


class PredictResponse(BaseModel):
    prediction: float


class SmartRequest(BaseModel):
    text: str
    features: list[float]


class SmartResponse(BaseModel):
    gemini_summary: str
    ml_prediction: float


# --------------------------
# 1) Gemini Chat
# ----------AI-chat", response_model=ChatResponse)
async def gemini_chat(req: ChatRequest):
    response = gemini.models.generate_content(
        model="gemini-1.5-flash",
        contents=req.prompt,
    )
    return ChatResponse(reply=response.text)


# --------------------------
# 2) ML Model Prediction
# --------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):

    # scale input
    scaled = scaler.transform([req.features])

    # predict
    pred = ml_model.predict(scaled)[0]

    return PredictResponse(prediction=float(pred))


# --------------------------
# 3) Smart Endpoint (Gemini + ML Together)
# --------------------------
@app.post("/smart-endpoint", response_model=SmartResponse)
async def smart_endpoint(req: SmartRequest):

    scaled = scaler.transform([req.features])
    pred = ml_model.predict(scaled)[0]

    prompt = f"""
    A cost-of-living prediction model gave the value: {pred}.
    Context: {req.text}
    Explain this prediction clearly in 3–5 sentences.
    """

    gemini_resp = gemini.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt,
    )

    return SmartResponse(
        gemini_summary=gemini_resp.text,
        ml_prediction=float(pred)
    )
