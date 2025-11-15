from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import joblib
import numpy as np
import torch
import torch.nn as nn


# ==============================
# FASTAPI APP
# ==============================

app = FastAPI()

# Enable CORS so frontend can call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# LOAD MODELS
# ==============================

# 1️⃣ Load RandomForest
model = joblib.load("AE_RFC.joblib")


# 2️⃣ Rebuild the EXACT encoder architecture from your notebook
class Encoder(nn.Module):
    def __init__(self, input_dim=11, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)


# 3️⃣ Load trained encoder weights
encoder = Encoder()
encoder.load_state_dict(torch.load("encoder_model.pth", map_location="cpu"))
encoder.eval()


# ==============================
# TEMPLATE ENGINE
# ==============================

templates = Jinja2Templates(directory="templates")


# ==============================
# INPUT MODEL (11 FEATURES)
# ==============================

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


# ==============================
# ROUTES
# ==============================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("frontend.html", {"request": request})


@app.post("/predict")
async def predict(data: ModelInput):

    # STEP 1 → Convert raw inputs to numpy
    raw_np = np.array([[
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

    # Convert to tensor
    raw_tensor = torch.tensor(raw_np)

    # STEP 2 → Encode using your Autoencoder
    with torch.no_grad():
        encoded = encoder(raw_tensor).numpy()   # Shape: (1,4)

    # STEP 3 → Predict using RandomForest
    pred = model.predict(encoded)[0]

    try:
        probs = model.predict_proba(encoded)[0]
        prob_N = float(probs[0])
        prob_Y = float(probs[1])
    except:
        prob_Y = 1.0 if pred == 1 else 0.0
        prob_N = 1.0 - prob_Y

    # STEP 4 → Format label
    prediction_label = "Y" if pred == 1 else "N"

    return {
        "prediction_label": prediction_label,
        "probability_Y": prob_Y,
        "probability_N": prob_N
    }
