"""
Domestic Violence Detection System - Backend API
FastAPI (Local-first, Cloud-ready)
Uses Transformer-based Zero-Shot Classification
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime
import numpy as np
import os
from tensorflow.keras.models import load_model
import numpy as np

ESCALATION_MODEL = load_model("models/escalation_lstm_model.h5")
MAX_SEQ_LEN = 10


# Speech-to-text
from speech import speech_to_text

# ML
from transformers import pipeline

# -------------------------
# App setup
# -------------------------

np.random.seed(42)

app = FastAPI(title="DV Detection API (ML Enabled)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# 🔒 Firestore DISABLED (local-first mode)
# -------------------------

db = None  # cloud-ready, but disabled for hackathon/demo

# -------------------------
# Load ML Model ONCE
# -------------------------

classifier = pipeline(
    task="zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1  # CPU (safe on Windows)
)

ABUSE_LABELS = [
    "controlling behavior",
    "verbal abuse",
    "threats",
    "physical abuse",
    "severe physical abuse"
]

# -------------------------
# Data Models
# -------------------------

class Message(BaseModel):
    timestamp: str
    text: str

class CaseInput(BaseModel):
    case_id: str
    messages: List[Message]

class AbuseClassification(BaseModel):
    control: float
    verbal: float
    threat: float
    physical: float
    severe_physical: float

class RiskAnalysis(BaseModel):
    case_id: str
    severity_latest: float
    escalation_probability: float
    escalation_speed: float
    risk_score: float
    risk_level: str
    flag_for_review: bool
    recommended_action: str
    timeline: List[Dict]

# -------------------------
# ML-based Classification
# -------------------------

def classify_message(text: str) -> AbuseClassification:
    """
    Transformer-based zero-shot classification.
    Handles unseen words & paraphrases.
    """
    result = classifier(text, ABUSE_LABELS, multi_label=True)
    scores = dict(zip(result["labels"], result["scores"]))

    return AbuseClassification(
        control=round(scores.get("controlling behavior", 0.0), 2),
        verbal=round(scores.get("verbal abuse", 0.0), 2),
        threat=round(scores.get("threats", 0.0), 2),
        physical=round(scores.get("physical abuse", 0.0), 2),
        severe_physical=round(scores.get("severe physical abuse", 0.0), 2),
    )

# -------------------------
# Severity Calculation
# -------------------------

def calculate_severity(cls: AbuseClassification) -> float:
    base = (
        cls.control * 0.8 +
        cls.verbal * 1.5 +
        cls.threat * 3.0 +
        cls.physical * 6.0 +
        cls.severe_physical * 8.0
    )

    # Domain rules
    if cls.physical > 0.25:
        base += 1.5
    if cls.severe_physical > 0.15:
        base += 2.0

    return round(min(base, 5.0), 2)

# -------------------------
# Escalation Detection
# -------------------------

def lstm_escalation(sequence):
    padded = [[0,0,0,0,0]] * max(0, MAX_SEQ_LEN - len(sequence)) + sequence[-MAX_SEQ_LEN:]
    X = np.array([padded])

    prob = float(ESCALATION_MODEL.predict(X, verbose=0)[0][0])

    speed = 0.0
    if len(sequence) >= 2:
        speed = abs(sequence[-1][3] - sequence[-2][3])  # physical abuse delta

    return {
        "escalation_probability": round(prob, 2),
        "escalation_speed": round(min(speed, 1.0), 2)
    }


# -------------------------
# Risk Scoring
# -------------------------

def calculate_risk(severity, esc_prob, esc_speed):
    score = round(
        0.55 * severity +
        0.30 * esc_prob +
        0.15 * esc_speed,
        2
    )

    if score < 1.2:
        return score, "LOW", "Continue monitoring"
    elif score < 2.2:
        return score, "MEDIUM", "NGO counselor outreach recommended"
    else:
        return score, "HIGH", "Immediate safety planning required"

# -------------------------
# Flagging Logic
# -------------------------

def should_flag(severity, esc_prob, history):
    if severity >= 2.5 and esc_prob >= 0.4:
        return True
    if len(history) >= 2 and history[-1] - history[-2] >= 1.5:
        return True
    if severity >= 4.0:
        return True
    return False

# -------------------------
# API Endpoints
# -------------------------

@app.get("/")
def root():
    return {"status": "DV Detection API running (local-first, ML-enabled)"}

@app.post("/analyze", response_model=RiskAnalysis)
async def analyze_case(case: CaseInput):
    try:
        messages = sorted(case.messages, key=lambda m: m.timestamp)

        timeline = []
        severities = []

        for msg in messages:
            cls = classify_message(msg.text)
            sev = calculate_severity(cls)
            severities.append(sev)

            timeline.append({
                "timestamp": msg.timestamp,
                "text": msg.text,
                "classification": cls.dict(),
                "severity": sev
            })

        sequence = [
                 [
        t["classification"]["control"],
        t["classification"]["verbal"],
        t["classification"]["threat"],
        t["classification"]["physical"],
        t["classification"]["severe_physical"]
             ]
            for t in timeline
                    ]

        esc = lstm_escalation(sequence)

        score, level, action = calculate_risk(
            severities[-1],
            esc["escalation_probability"],
            esc["escalation_speed"]
        )

        flag = should_flag(severities[-1], esc["escalation_probability"], severities)

        return RiskAnalysis(
            case_id=case.case_id,
            severity_latest=severities[-1],
            escalation_probability=esc["escalation_probability"],
            escalation_speed=esc["escalation_speed"],
            risk_score=score,
            risk_level=level,
            flag_for_review=flag,
            recommended_action=action,
            timeline=timeline
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/speech-analyze")
async def analyze_voice(file: UploadFile = File(...)):
    """
    Audio → Speech-to-Text → ML DV Analysis
    """
    temp_path = f"temp_{file.filename}"

    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        transcript = speech_to_text(temp_path)

        if not transcript:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")

        case = CaseInput(
            case_id="VOICE_CASE",
            messages=[
                Message(
                    timestamp=datetime.utcnow().isoformat(),
                    text=transcript
                )
            ]
        )

        analysis = await analyze_case(case)

        return {
            "transcript": transcript,
            "analysis": analysis
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
