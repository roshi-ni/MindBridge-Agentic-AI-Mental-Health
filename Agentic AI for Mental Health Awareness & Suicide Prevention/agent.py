"""
Empathetic Conversational Agent — MindBridge
Orchestrates the RAG pipeline, distress detection, and crisis intervention
into a unified, session-aware mental health conversation flow.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from config import CRISIS_HELPLINES, SAFE_MESSAGING_DISCLAIMER
from distress_detector import DistressDetector, RiskPredictor
from rag_pipeline import GraniteRAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Single conversation turn."""

    role: str           # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    risk_level: str = "NONE"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "risk_level": self.risk_level,
            "metadata": self.metadata,
        }


class MindBridgeAgent:
    """
    Main conversational agent for MindBridge.

    Responsibilities:
    - Maintain session-level conversation history
    - Run distress detection on every user message
    - Trigger appropriate RAG-augmented Granite responses
    - Inject crisis resources when risk level is HIGH/CRITICAL
    - Issue proactive check-in alerts via RiskPredictor
    """

    WELCOME_MESSAGE = (
        "Hello, I'm **MindBridge** — your compassionate AI mental health support companion. 💙\n\n"
        "I'm here to listen, support, and share evidence-based mental health resources. "
        "You can talk to me about anything — stress, anxiety, depression, relationships, "
        "or just how your day is going.\n\n"
        "**A few things to know:**\n"
        "• Everything you share stays in this session only.\n"
        "• I'm an AI — not a replacement for professional care.\n"
        "• If you're ever in crisis, I'll connect you to real help immediately.\n\n"
        "How are you feeling today? 🌿"
    )

    def __init__(self) -> None:
        self.session_id = str(uuid.uuid4())
        self.history: list[Message] = []
        self.detector = DistressDetector()
        self.predictor = RiskPredictor()
        self.rag = GraniteRAGPipeline.get_instance()
        self._mood_log: list[dict] = []
        logger.info("MindBridgeAgent session started: %s", self.session_id)

    # ── Core respond method ───────────────────────────────────────────────────
    def respond(self, user_input: str) -> dict[str, Any]:
        """
        Process user input and return a structured response dict:
        {
            response: str,
            risk_level: str,
            risk_score: float,
            crisis_resources: list | None,
            proactive_alert: str | None,
            sources: list,
        }
        """
        # 1. Distress detection
        assessment = self.detector.analyse(user_input)
        predictor_result = self.predictor.update(assessment.risk_score)

        # 2. Record user message
        user_msg = Message(
            role="user",
            content=user_input,
            risk_level=assessment.risk_level,
            metadata=assessment.to_dict(),
        )
        self.history.append(user_msg)
        self._mood_log.append(
            {
                "timestamp": user_msg.timestamp.isoformat(),
                "risk_score": assessment.risk_score,
                "risk_level": assessment.risk_level,
            }
        )

        # 3. Build conversation history for RAG context
        history_dicts = [{"role": m.role, "content": m.content} for m in self.history[-10:]]

        # 4. Generate RAG response
        rag_result = self.rag.generate_response(
            user_message=user_input,
            conversation_history=history_dicts,
            risk_level=assessment.risk_level,
        )

        response_text = rag_result["response"]

        # 5. Append crisis resources for HIGH/CRITICAL
        crisis_resources = None
        if assessment.risk_level in ("HIGH", "CRITICAL"):
            crisis_resources = self._get_crisis_block()
            if crisis_resources not in response_text:
                response_text += f"\n\n{crisis_resources}"

        # 6. Proactive alert injection
        proactive_alert = None
        if predictor_result["proactive_alert"] and assessment.risk_level not in ("HIGH", "CRITICAL"):
            proactive_alert = predictor_result["alert_message"]

        # 7. Record assistant message
        assistant_msg = Message(
            role="assistant",
            content=response_text,
            risk_level=assessment.risk_level,
            metadata={"sources": rag_result["sources"]},
        )
        self.history.append(assistant_msg)

        return {
            "response": response_text,
            "risk_level": assessment.risk_level,
            "risk_score": assessment.risk_score,
            "detected_signals": assessment.detected_signals,
            "crisis_resources": crisis_resources,
            "proactive_alert": proactive_alert,
            "sources": rag_result["sources"],
            "requires_immediate_action": assessment.requires_immediate_action,
        }

    # ── Crisis resource block ─────────────────────────────────────────────────
    def _get_crisis_block(self) -> str:
        india = CRISIS_HELPLINES["India"]
        intl = CRISIS_HELPLINES["International"]
        lines = [
            "---",
            "🆘 **Crisis Support — You Are Not Alone**",
            "",
            "**🇮🇳 India:**",
        ]
        for h in india:
            lines.append(f"• **{h['name']}**: {h['number']} ({h['available']})")
        lines += ["", "**🌍 International:**"]
        for h in intl[:3]:
            lines.append(f"• **{h['name']}**: {h['number']} ({h['available']})")
        lines += [
            "",
            "🚨 **If you are in immediate danger, call emergency services: 112 (India) / 911 (US)**",
            "---",
        ]
        return "\n".join(lines)

    # ── Journaling analysis ───────────────────────────────────────────────────
    def analyse_journal(self, journal_text: str) -> dict[str, Any]:
        """
        Analyse a journal entry for distress signals and return
        a supportive RAG-augmented response.
        """
        assessment = self.detector.analyse(journal_text)
        prompt = (
            f"The following is a personal journal entry. "
            f"Please provide a compassionate, supportive response that acknowledges "
            f"the person's feelings and offers relevant coping strategies.\n\n"
            f"Journal entry:\n{journal_text}"
        )
        rag_result = self.rag.generate_response(
            user_message=prompt,
            conversation_history=[],
            risk_level=assessment.risk_level,
        )
        return {
            "analysis": rag_result["response"],
            "risk_level": assessment.risk_level,
            "risk_score": assessment.risk_score,
            "detected_signals": assessment.detected_signals,
            "sources": rag_result["sources"],
        }

    # ── Mood tracking ─────────────────────────────────────────────────────────
    def get_mood_timeline(self) -> list[dict]:
        """Return the mood/risk score timeline for this session."""
        return self._mood_log

    def get_session_summary(self) -> dict[str, Any]:
        """Return a summary of the current session's risk trend."""
        trend = self.detector.get_session_trend()
        return {
            "session_id": self.session_id,
            "total_messages": len([m for m in self.history if m.role == "user"]),
            "session_start": self.history[0].timestamp.isoformat() if self.history else None,
            **trend,
        }

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset_session(self) -> None:
        self.session_id = str(uuid.uuid4())
        self.history = []
        self._mood_log = []
        self.detector.reset_session()
        self.predictor.reset()
        logger.info("Session reset. New session ID: %s", self.session_id)

    # ── Resource lookup ───────────────────────────────────────────────────────
    def get_resources_by_topic(self, topic: str) -> dict[str, Any]:
        """Retrieve and generate a focused educational resource on the given topic."""
        rag_result = self.rag.generate_response(
            user_message=f"Tell me about {topic} — what is it and how can I cope?",
            conversation_history=[],
            risk_level="NONE",
        )
        return rag_result
