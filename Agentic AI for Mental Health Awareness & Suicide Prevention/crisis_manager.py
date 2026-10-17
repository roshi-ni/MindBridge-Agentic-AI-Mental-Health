"""
Crisis Intervention Module
Handles high-risk situations: provides helpline details, safety plans,
and (optionally) SMS alerts via Twilio when consent is given.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from config import CRISIS_HELPLINES

logger = logging.getLogger(__name__)


class CrisisInterventionManager:
    """
    Manages crisis-level responses including:
    - Surfacing relevant helpline resources
    - Generating personalised safety plan prompts
    - Optional SMS alert dispatch (with explicit user consent)
    """

    SAFETY_PLAN_TEMPLATE = """
## 🛡️ Your Personal Safety Plan

A safety plan is something you can use when you're having thoughts of suicide or self-harm.
Fill this out when you're feeling okay — so it's ready when you need it.

### Step 1 — Warning Signs
*What thoughts, feelings, or behaviours tell me a crisis may be developing?*
- ___________________________________________
- ___________________________________________

### Step 2 — Internal Coping Strategies
*Things I can do on my own to distract myself or feel better:*
- ___________________________________________
- ___________________________________________
- ___________________________________________

### Step 3 — People & Places That Provide Distraction
*People I can talk to or places I can go to take my mind off things:*
1. Name: _____________  Contact: _____________
2. Name: _____________  Contact: _____________

### Step 4 — People I Can Ask For Help
*Trusted people I can tell I'm in crisis:*
1. Name: _____________  Contact: _____________
2. Name: _____________  Contact: _____________

### Step 5 — Professionals & Agencies I Can Contact
| Name | Phone | Hours |
|------|-------|-------|
| iCall (TISS) | 9152987821 | Mon–Sat 8am–10pm |
| Vandrevala Foundation | 1860-2662-345 | 24/7 |
| AASRA | 9820466627 | 24/7 |
| Crisis Text Line (US) | Text HOME → 741741 | 24/7 |
| Emergency Services | 112 / 911 | 24/7 |

### Step 6 — Making My Environment Safe
*Things I can do to make my space safer (reduce access to means):*
- ___________________________________________
- ___________________________________________

---
*Source: Based on the Stanley-Brown Safety Planning Intervention*
*Review this plan with a mental health professional.*
"""

    COPING_CARDS = {
        "breathing": (
            "**Box Breathing** — do this right now:\n"
            "1. Breathe IN for 4 counts 🌬️\n"
            "2. HOLD for 4 counts ⏸️\n"
            "3. Breathe OUT for 4 counts 💨\n"
            "4. HOLD for 4 counts ⏸️\n"
            "Repeat 4 times. You've got this. 💙"
        ),
        "grounding": (
            "**5-4-3-2-1 Grounding** — bring yourself back:\n"
            "👀 Name **5** things you can see\n"
            "✋ Feel **4** things you can touch\n"
            "👂 Notice **3** things you can hear\n"
            "👃 Find **2** things you can smell\n"
            "👅 Notice **1** thing you can taste\n\n"
            "Take your time. You are safe right now. 💙"
        ),
        "affirmation": (
            "**Remind yourself:**\n\n"
            "💙 This pain is temporary — it will pass.\n"
            "💙 You have survived every difficult moment so far.\n"
            "💙 Reaching out is an act of courage.\n"
            "💙 You are not a burden — you are loved.\n"
            "💙 Help is available and you deserve it.\n\n"
            "*Take a deep breath. You matter.*"
        ),
        "distraction": (
            "**Quick Distraction Techniques:**\n\n"
            "• Count backwards from 100 by 7s\n"
            "• Name 10 animals starting with each letter of your name\n"
            "• Look around and categorise objects by colour\n"
            "• Squeeze an ice cube tightly\n"
            "• Hum your favourite song\n"
            "• Text or call someone you trust\n\n"
            "*These create a temporary bridge to get through the hardest moments.*"
        ),
    }

    def get_helplines(self, region: str = "India") -> list[dict]:
        """Return helplines for the specified region."""
        return CRISIS_HELPLINES.get(region, CRISIS_HELPLINES["International"])

    def get_all_helplines(self) -> dict[str, list[dict]]:
        return CRISIS_HELPLINES

    def get_safety_plan(self) -> str:
        return self.SAFETY_PLAN_TEMPLATE

    def get_coping_card(self, coping_type: str = "breathing") -> str:
        return self.COPING_CARDS.get(coping_type, self.COPING_CARDS["breathing"])

    def format_crisis_response(self, risk_level: str, user_name: str = "friend") -> str:
        """
        Build a contextually appropriate crisis response message.
        """
        base_empathy = (
            f"I hear you, {user_name}, and I'm really glad you're talking to me. "
            "What you're feeling right now is real and valid — you don't have to carry this alone.\n\n"
        )

        if risk_level == "CRITICAL":
            return (
                f"{base_empathy}"
                "Right now, the most important thing is your safety. "
                "Please reach out to a crisis counselor **immediately** — they are trained "
                "to help and will not judge you:\n\n"
                f"{self._format_top_helplines()}\n\n"
                "🚨 **If you are in immediate danger, call 112 (India) or 911 (US/Canada) now.**\n\n"
                "Are you able to call one of these numbers right now? I'll stay here with you. 💙"
            )
        elif risk_level == "HIGH":
            return (
                f"{base_empathy}"
                "I'm concerned about how you're feeling. "
                "You deserve real human support right now. Here are people who can help:\n\n"
                f"{self._format_top_helplines()}\n\n"
                "Would you like me to share some immediate coping techniques while you "
                "gather the courage to reach out? 💙"
            )
        else:
            return (
                f"{base_empathy}"
                "I want you to know that support is always available:\n\n"
                f"{self._format_top_helplines()}\n\n"
                "You're not alone in this. 💙"
            )

    def _format_top_helplines(self, count: int = 3) -> str:
        lines = ["**🇮🇳 India:**"]
        for h in CRISIS_HELPLINES["India"][:count]:
            lines.append(f"• **{h['name']}**: `{h['number']}` — {h['available']}")
        lines.append("\n**🌍 International:**")
        for h in CRISIS_HELPLINES["International"][:2]:
            lines.append(f"• **{h['name']}**: `{h['number']}` — {h['available']}")
        return "\n".join(lines)

    def send_sms_alert(
        self, to_number: str, message: str, user_consent: bool = False
    ) -> dict[str, Any]:
        """
        Send an SMS alert via Twilio (requires explicit user consent).
        Returns a dict with success status and message.
        """
        if not user_consent:
            return {
                "success": False,
                "reason": "User consent not provided. SMS alert not sent.",
            }

        try:
            from twilio.rest import Client  # optional dependency

            account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            auth_token = os.getenv("TWILIO_AUTH_TOKEN")
            from_number = os.getenv("TWILIO_FROM_NUMBER")

            if not all([account_sid, auth_token, from_number]):
                return {"success": False, "reason": "Twilio credentials not configured."}

            client = Client(account_sid, auth_token)
            msg = client.messages.create(body=message, from_=from_number, to=to_number)
            logger.info("SMS alert sent. SID: %s", msg.sid)
            return {"success": True, "sid": msg.sid}

        except ImportError:
            return {"success": False, "reason": "Twilio package not installed."}
        except Exception as exc:
            logger.error("SMS send failed: %s", exc)
            return {"success": False, "reason": str(exc)}
