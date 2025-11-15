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

# Allow frontend -> backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load RandomForest
model = joblib.load("AE_RFC.joblib")


# Encoder architecture from your Autoencoder
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


# Load encoder weights (you saved only encoder)
encoder = Encoder()
encoder.load_state_dict(torch.load("encoder_model.pth", map_location="cpu"))
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

    # Convert to tensor
    raw_np = np.array([[
        data.Gender, data.AGE, data.Urea, data.Cr, data.HbA1c,
        data.Chol, data.TG, data.HDL, data.LDL, data.VLDL, data.BMI
    ]], dtype=np.float32)

    raw_tensor = torch.tensor(raw_np)

    # Encode to 4 features
    with torch.no_grad():
        encoded = encoder(raw_tensor).numpy()

    # RF prediction
    pred = model.predict(encoded)[0]

    try:
        probs = model.predict_proba(encoded)[0]
        prob_N = float(probs[0])
        prob_Y = float(probs[1])
    except:
        prob_Y = 1.0 if pred == 1 else 0.0
        prob_N = 1.0 - prob_Y

    prediction_label = "Y" if pred == 1 else "N"

    return {
        "prediction_label": prediction_label,
        "probability_Y": prob_Y,
        "probability_N": prob_N
    }
