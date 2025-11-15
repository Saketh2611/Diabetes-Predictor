from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import joblib
import numpy as np
import torch
import torch.nn as nn


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load RandomForest model
model = joblib.load("AE_RFC.joblib")


# ------------------------------------
# CORRECT ENCODER (MATCHES SAVED FILE)
# ------------------------------------
encoder = nn.Sequential(
    nn.Linear(11, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 4)
)

state = torch.load("encoder_model.pth", map_location="cpu")
encoder.load_state_dict(state, strict=True)
encoder.eval()


templates = Jinja2Templates(directory="templates")


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


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("frontend.html", {"request": request})


@app.post("/predict")
async def predict(data: ModelInput):

    raw_np = np.array([[ 
        data.Gender, data.AGE, data.Urea, data.Cr, data.HbA1c,
        data.Chol, data.TG, data.HDL, data.LDL, data.VLDL, data.BMI
    ]], dtype=np.float32)

    raw_tensor = torch.tensor(raw_np)

    # Encode into 4 latent features
    with torch.no_grad():
        encoded = encoder(raw_tensor).numpy()

    pred = model.predict(encoded)[0]

    try:
        probs = model.predict_proba(encoded)[0]
        prob_N, prob_Y = float(probs[0]), float(probs[1])
    except:
        prob_Y = float(pred)
        prob_N = 1 - prob_Y

    prediction_label = "Y" if pred == 1 else "N"

    return {
        "prediction_label": prediction_label,
        "probability_Y": prob_Y,
        "probability_N": prob_N
    }
