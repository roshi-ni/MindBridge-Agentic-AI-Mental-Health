"""
Configuration module for IBM Watsonx / Granite integration.
Loads credentials from environment variables or .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── IBM Watsonx credentials ────────────────────────────────────────────────────
IBM_API_KEY = os.getenv("IBM_API_KEY", "GIDwr-JMzCwdmdUxCSjG_KFW9m1gPS_I8MVMJoJKrXuO")
IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID", "181d5190-0e53-4269-9dc5-7f24d6b7c9f9")
IBM_WATSONX_URL = os.getenv("IBM_WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

# ── Model identifiers ──────────────────────────────────────────────────────────
GRANITE_MODEL_ID = os.getenv("GRANITE_MODEL_ID", "ibm/granite-13b-chat-v2")
GRANITE_INSTRUCT_MODEL = "ibm/granite-13b-instruct-v2"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Generation parameters ──────────────────────────────────────────────────────
GENERATION_PARAMS = {
    "decoding_method": "greedy",
    "max_new_tokens": 800,
    "min_new_tokens": 50,
    "stop_sequences": ["<|endoftext|>", "Human:", "User:"],
    "repetition_penalty": 1.1,
    "temperature": 0.7,
}

# ── Application settings ───────────────────────────────────────────────────────
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "mental-health-agent-secret-2024")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Vector store path ──────────────────────────────────────────────────────────
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
KNOWLEDGE_BASE_PATH = "./data/knowledge_base"

# ── Risk thresholds ────────────────────────────────────────────────────────────
HIGH_RISK_THRESHOLD = 0.75
MEDIUM_RISK_THRESHOLD = 0.45
LOW_RISK_THRESHOLD = 0.20

# ── Crisis helplines (India + International) ──────────────────────────────────
CRISIS_HELPLINES = {
    "India": [
        {"name": "iCall (TISS)", "number": "9152987821", "available": "Mon–Sat, 8am–10pm"},
        {"name": "Vandrevala Foundation", "number": "1860-2662-345", "available": "24/7"},
        {"name": "Snehi", "number": "044-24640050", "available": "24/7"},
        {"name": "AASRA", "number": "9820466627", "available": "24/7"},
        {"name": "iCall WhatsApp", "number": "+91 9152987821", "available": "Mon–Sat"},
    ],
    "International": [
        {"name": "Crisis Text Line (US)", "number": "Text HOME to 741741", "available": "24/7"},
        {"name": "National Suicide Prevention Lifeline (US)", "number": "988", "available": "24/7"},
        {"name": "Samaritans (UK)", "number": "116 123", "available": "24/7"},
        {"name": "Lifeline (Australia)", "number": "13 11 14", "available": "24/7"},
        {"name": "International Association for Suicide Prevention",
         "number": "https://www.iasp.info/resources/Crisis_Centres/", "available": "24/7"},
    ],
}

# ── Distress keyword categories ────────────────────────────────────────────────
DISTRESS_KEYWORDS = {
    "suicidal_ideation": [
        "want to die", "kill myself", "end my life", "suicide", "suicidal",
        "don't want to live", "no reason to live", "better off dead",
        "end it all", "take my life", "not worth living", "wish i was dead",
        "want to disappear", "rather be dead",
    ],
    "self_harm": [
        "hurt myself", "cut myself", "self harm", "self-harm", "hurting myself",
        "burn myself", "injure myself", "harm myself",
    ],
    "severe_depression": [
        "hopeless", "worthless", "empty inside", "no point", "nothing matters",
        "can't go on", "can't cope", "give up", "pointless", "numb",
        "dark thoughts", "trapped", "burden to everyone",
    ],
    "crisis_signals": [
        "goodbye forever", "final message", "won't be here", "last day",
        "saying goodbye", "nobody cares", "no one would miss me",
        "farewell", "last time", "can't take it anymore",
    ],
    "moderate_distress": [
        "depressed", "anxious", "overwhelmed", "exhausted", "stressed",
        "can't sleep", "panic attack", "crying", "lonely", "isolated",
        "sad", "miserable", "scared", "afraid", "desperate",
    ],
    "mild_distress": [
        "worried", "nervous", "upset", "frustrated", "tired",
        "not okay", "struggling", "difficult", "hard time", "low mood",
    ],
}

# ── Safe messaging guidelines ──────────────────────────────────────────────────
SAFE_MESSAGING_DISCLAIMER = (
    "⚠️ **Important Notice**: This AI provides support and information only — "
    "it is NOT a substitute for professional mental health care. "
    "If you are in crisis, please contact a helpline immediately."
)
