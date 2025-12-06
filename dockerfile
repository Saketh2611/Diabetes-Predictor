# ---------- Base Image ----------
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# ---------- System dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ---------- Install Python dependencies ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy ONLY needed project files ----------
COPY main.py .
COPY templates/ templates/

# Copy ML model files explicitly
COPY AE_RFC.joblib .
COPY encoder_model.pth .
COPY Scaler.joblib .

# If you also need static files (CSS/JS)
# COPY static/ static/

# Render sets PORT automatically
ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
