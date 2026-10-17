"""
Distress Detection & Risk Prediction Engine
Analyses user messages for distress signals, classifies risk level,
and generates risk scores using keyword + sentiment analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from config import (
    DISTRESS_KEYWORDS,
    HIGH_RISK_THRESHOLD,
    LOW_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
)


@dataclass
class RiskAssessment:
    """Result of a single-message risk analysis."""

    risk_level: str                     # NONE | LOW | MEDIUM | HIGH | CRITICAL
    risk_score: float                   # 0.0 – 1.0
    detected_signals: list[str]         # categories triggered
    matched_phrases: list[str]          # exact phrases found
    requires_immediate_action: bool
    recommended_action: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "risk_score": round(self.risk_score, 3),
            "detected_signals": self.detected_signals,
            "matched_phrases": self.matched_phrases,
            "requires_immediate_action": self.requires_immediate_action,
            "recommended_action": self.recommended_action,
            "timestamp": self.timestamp.isoformat(),
        }


class DistressDetector:
    """
    Multi-layered distress signal detector that combines:
    - Keyword / phrase matching (weighted by severity)
    - Escalation detection (rising risk over conversation history)
    - Temporal pattern analysis (frequency of distress signals)
    """

    # ── Severity weights per keyword category ────────────────────────────────
    CATEGORY_WEIGHTS = {
        "suicidal_ideation": 1.0,
        "crisis_signals": 0.95,
        "self_harm": 0.85,
        "severe_depression": 0.55,
        "moderate_distress": 0.30,
        "mild_distress": 0.10,
    }

    def __init__(self) -> None:
        self._history: list[RiskAssessment] = []

    # ── Core analysis ─────────────────────────────────────────────────────────
    def analyse(self, text: str) -> RiskAssessment:
        """Analyse a single message and return a RiskAssessment."""
        normalised = text.lower().strip()
        normalised = re.sub(r"[^\w\s']", " ", normalised)

        detected_signals: list[str] = []
        matched_phrases: list[str] = []
        weighted_score = 0.0

        for category, phrases in DISTRESS_KEYWORDS.items():
            weight = self.CATEGORY_WEIGHTS.get(category, 0.1)
            for phrase in phrases:
                if phrase in normalised:
                    detected_signals.append(category)
                    matched_phrases.append(phrase)
                    weighted_score = max(weighted_score, weight)

        # De-duplicate signals list
        detected_signals = list(dict.fromkeys(detected_signals))

        # Boost score if multiple high-severity categories co-occur
        high_severity = {"suicidal_ideation", "crisis_signals", "self_harm"}
        if len(high_severity & set(detected_signals)) >= 2:
            weighted_score = min(weighted_score * 1.2, 1.0)

        # Contextual escalation: boost if recent history shows rising risk
        if self._history:
            last_score = self._history[-1].risk_score
            if last_score > MEDIUM_RISK_THRESHOLD and weighted_score > LOW_RISK_THRESHOLD:
                weighted_score = min(weighted_score + 0.10, 1.0)

        # Classify risk level
        risk_level = self._classify(weighted_score, detected_signals)
        requires_immediate = risk_level in ("HIGH", "CRITICAL")

        assessment = RiskAssessment(
            risk_level=risk_level,
            risk_score=weighted_score,
            detected_signals=detected_signals,
            matched_phrases=matched_phrases,
            requires_immediate_action=requires_immediate,
            recommended_action=self._recommend_action(risk_level),
        )

        self._history.append(assessment)
        return assessment

    # ── Classification logic ──────────────────────────────────────────────────
    def _classify(self, score: float, signals: list[str]) -> str:
        critical_categories = {"suicidal_ideation", "crisis_signals"}

        if critical_categories & set(signals):
            # Explicit suicidal/crisis → always at least HIGH
            return "CRITICAL" if score >= HIGH_RISK_THRESHOLD else "HIGH"

        if score >= HIGH_RISK_THRESHOLD:
            return "HIGH"
        if score >= MEDIUM_RISK_THRESHOLD:
            return "MEDIUM"
        if score >= LOW_RISK_THRESHOLD:
            return "LOW"
        return "NONE"

    # ── Action recommendation ─────────────────────────────────────────────────
    def _recommend_action(self, risk_level: str) -> str:
        actions = {
            "CRITICAL": (
                "IMMEDIATE: Provide crisis helpline numbers. Express deep empathy. "
                "Encourage user to call emergency services or a crisis line now. "
                "Do not leave them alone in the conversation."
            ),
            "HIGH": (
                "URGENT: Validate emotions strongly. Provide helpline numbers. "
                "Gently ask about safety. Encourage reaching out to a trusted person."
            ),
            "MEDIUM": (
                "ELEVATED: Be extra empathetic. Validate feelings thoroughly. "
                "Suggest professional support and coping strategies."
            ),
            "LOW": (
                "SUPPORTIVE: Acknowledge the difficulty. Offer coping resources. "
                "Maintain a warm, supportive tone."
            ),
            "NONE": (
                "STANDARD: Respond helpfully and warmly. Share relevant resources."
            ),
        }
        return actions.get(risk_level, actions["NONE"])

    # ── Session trend ─────────────────────────────────────────────────────────
    def get_session_trend(self) -> dict[str, Any]:
        """Return aggregated risk statistics for the current session."""
        if not self._history:
            return {"session_peak_risk": "NONE", "avg_score": 0.0, "total_messages": 0}

        scores = [a.risk_score for a in self._history]
        levels = [a.risk_level for a in self._history]
        severity_order = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        peak = max(levels, key=lambda l: severity_order.index(l))

        return {
            "session_peak_risk": peak,
            "avg_score": round(sum(scores) / len(scores), 3),
            "latest_score": round(scores[-1], 3),
            "total_messages": len(self._history),
            "high_risk_count": sum(1 for l in levels if l in ("HIGH", "CRITICAL")),
            "risk_trend": self._compute_trend(scores),
        }

    def _compute_trend(self, scores: list[float]) -> str:
        if len(scores) < 3:
            return "insufficient_data"
        recent = scores[-3:]
        if recent[-1] > recent[0] + 0.10:
            return "escalating"
        if recent[-1] < recent[0] - 0.10:
            return "de-escalating"
        return "stable"

    def reset_session(self) -> None:
        self._history = []


class RiskPredictor:
    """
    Proactive stress-peak forecaster.
    Tracks rolling average risk scores and raises an alert when
    the trend indicates a risk spike is imminent.
    """

    WINDOW = 5  # messages to consider for rolling average

    def __init__(self) -> None:
        self._scores: list[float] = []

    def update(self, score: float) -> dict[str, Any]:
        self._scores.append(score)
        window = self._scores[-self.WINDOW :]
        avg = sum(window) / len(window)

        alert = False
        alert_message = ""
        if len(window) >= 3 and avg >= MEDIUM_RISK_THRESHOLD:
            alert = True
            alert_message = (
                "I've noticed that our conversation has been touching on some "
                "difficult emotions. I want to check in — how are you really doing? "
                "Remember, it's okay to reach out for support."
            )
        elif len(window) >= 3:
            delta = window[-1] - window[0]
            if delta >= 0.25:
                alert = True
                alert_message = (
                    "I can sense things may be getting harder as we talk. "
                    "Would it help to take a breath together, or would you like me "
                    "to share some coping strategies?"
                )

        return {
            "rolling_avg": round(avg, 3),
            "proactive_alert": alert,
            "alert_message": alert_message,
            "window_size": len(window),
        }

    def reset(self) -> None:
        self._scores = []
