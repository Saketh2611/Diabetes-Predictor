# main.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import joblib
import numpy as np
import torch
import torch.nn as nn
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- paths (ensure these files exist in the project folder) ----------
RF_PATH = "AE_RFC.joblib"
ENCODER_PATH = "encoder_model.pth"
SCALER_PATH = "Scaler.joblib"

# ---------- load RandomForest ----------
if not os.path.exists(RF_PATH):
    raise FileNotFoundError(f"RandomForest model not found at {RF_PATH}")
model = joblib.load(RF_PATH)

# ---------- build encoder architecture (matches saved keys) ----------
encoder = nn.Sequential(
    nn.Linear(11, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 4)
)

# ---------- load encoder weights ----------
if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(f"Encoder weights not found at {ENCODER_PATH}")
state = torch.load(ENCODER_PATH, map_location="cpu")
encoder.load_state_dict(state, strict=True)
encoder.eval()

# ---------- load scaler ----------
scaler = None
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
else:
    # If scaler is missing, continue but warn in logs and DO NOT scale inputs.
    print(f"Warning: scaler file not found at {SCALER_PATH}. Inputs will NOT be scaled. This may break performance.")

templates = Jinja2Templates(directory="Templates")


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
    return Templates.TemplateResponse("frontend.html", {"request": request})


@app.post("/predict")
async def predict(data: ModelInput):
    # 1) assemble raw vector in the *CORRECT* training order
    raw = np.array([[  
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
    ]], dtype=np.float32)
    # 2) scale using saved scaler (if available)
    if scaler is not None:
        try:
            raw_scaled = scaler.transform(raw)
        except Exception as e:
            # If scaler fails for any reason, log and return error to client
            return JSONResponse({"error": "Scaler transform failed", "detail": str(e)}, status_code=500)
    else:
        raw_scaled = raw  # fallback (not recommended)

    # 3) encode via encoder
    with torch.no_grad():
        tensor = torch.tensor(raw_scaled, dtype=torch.float32)
        encoded = encoder(tensor).detach().numpy()  # shape (1, latent_dim)

    # 4) RF predict
    try:
        pred = int(model.predict(encoded)[0])
    except Exception as e:
        return JSONResponse({"error": "Model predict failed", "detail": str(e)}, status_code=500)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(encoded)[0]
        prob_N = float(probs[0])
        prob_Y = float(probs[1])
    else:
        prob_Y = 1.0 if pred == 1 else 0.0
        prob_N = 1.0 - prob_Y

    prediction_label = "Y" if pred == 1 else "N"

    return {
        "prediction_label": prediction_label,
        "probability_Y": prob_Y,
        "probability_N": prob_N
    }
