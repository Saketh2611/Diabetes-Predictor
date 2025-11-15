from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("AE_RFC.joblib")

# Load templates
templates = Jinja2Templates(directory="templates")


# ----------- INPUT FORMAT EXPECTED FROM FRONTEND ------------
class ModelInput(BaseModel):
    Gender: int
    AGE: float
    Urea: float
    Cr: float
    HbA1c: float
    Chol: float
    TG: float
    HDL: float
    LDL: float
    VLDL: float
    BMI: float


# ----------------- ROUTES ------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_html(request: Request):
    return templates.TemplateResponse("frontend.html", {"request": request})


@app.post("/predict")
async def predict_disease(data: ModelInput):

    # Convert input into a 2D array matching training format
    features = np.array([[
        data.Gender,
        data.AGE,
        data.Urea,
        data.Cr,
        data.HbA1c,
        data.Chol,
        data.TG,
        data.HDL,
        data.LDL,
        data.VLDL,
        data.BMI
    ]])

    # Prediction
    pred = model.predict(features)[0]

    # Probability (assuming classifier has predict_proba)
    try:
        probs = model.predict_proba(features)[0]
        prob_N = float(probs[0])  # class 0
        prob_Y = float(probs[1])  # class 1
    except:
        # If no predict_proba method
        prob_Y = 1.0 if pred == 1 else 0.0
        prob_N = 1.0 - prob_Y

    # Create label
    prediction_label = "Y" if pred == 1 else "N"

    return {
        "prediction_label": prediction_label,
        "probability_Y": prob_Y,
        "probability_N": prob_N
    }
