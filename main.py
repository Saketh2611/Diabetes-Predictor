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

# Enable CORS (required for your frontend fetch)
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


# 2️⃣ Define PyTorch Encoder class (IMPORTANT: match your notebook exactly)
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(11, 8)
        self.layer2 = nn.Linear(8, 4)  # latent size = 4
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


# 3️⃣ Load saved encoder weights
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

    # ---------------------------
    # STEP 1: RAW INPUT → NUMPY
    # ---------------------------
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

    # Convert to PyTorch tensor
    raw_tensor = torch.tensor(raw_np)

    # ---------------------------
    # STEP 2: ENCODE USING AE
    # ---------------------------
    with torch.no_grad():
        encoded = encoder(raw_tensor).numpy()  # shape = (1, latent_dim=4)

    # ---------------------------
    # STEP 3: RANDOM FOREST PREDICT
    # ---------------------------
    pred = model.predict(encoded)[0]

    try:
        probs = model.predict_proba(encoded)[0]
        prob_N = float(probs[0])
        prob_Y = float(probs[1])
    except:
        prob_Y = 1.0 if pred == 1 else 0.0
        prob_N = 1.0 - prob_Y

    # Convert label
    prediction_label = "Y" if pred == 1 else "N"

    # ---------------------------
    # RETURN RESPONSE
    # ---------------------------
    return {
        "prediction_label": prediction_label,
        "probability_Y": prob_Y,
        "probability_N": prob_N
    }
