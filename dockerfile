# ---------- Base Image ----------
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# ---------- System dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ---------- Install CPU-only PyTorch (FAST + SMALL) ----------
RUN pip install --no-cache-dir torch==2.2.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# ---------- Install remaining Python dependencies ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy code ----------
COPY main.py .
COPY templates/ templates/
COPY AE_RFC.joblib .
COPY encoder_model.pth .
COPY Scaler.joblib .

ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
