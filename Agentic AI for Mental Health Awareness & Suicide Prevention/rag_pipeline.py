"""
RAG Pipeline — IBM Granite (REST) + TF-IDF vector store
Python 3.15 compatible: uses requests-based WatsonxClient and
scikit-learn TF-IDF cosine similarity (no torch/chromadb needed).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import SAFE_MESSAGING_DISCLAIMER
from knowledge_base import get_knowledge_base
from watsonx_client import WatsonxClient

logger = logging.getLogger(__name__)


# ── TF-IDF vector store ────────────────────────────────────────────────────────

class _TFIDFVectorStore:
    """
    TF-IDF cosine-similarity vector store.
    Uses scikit-learn (already installed) — no torch or chromadb needed.
    Works on Python 3.15+.
    """

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            stop_words="english",
            sublinear_tf=True,
        )
        self._docs: list[dict] = []
        self._matrix = None   # sparse TF-IDF matrix

    def add(self, documents: list[dict]) -> None:
        """Index documents list of {id, content, title, category, tags}."""
        self._docs = documents
        texts = [d["content"] for d in documents]
        self._matrix = self._vectorizer.fit_transform(texts)
        logger.info("TF-IDF vector store: indexed %d documents.", len(documents))

    def query(self, text: str, n: int = 4) -> list[dict]:
        """Return top-n documents by cosine similarity."""
        if self._matrix is None or not self._docs:
            return []
        q_vec = self._vectorizer.transform([text])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        top_idx = np.argsort(sims)[::-1][:n]
        results = []
        for i in top_idx:
            doc = dict(self._docs[i])
            doc["score"] = float(sims[i])
            results.append(doc)
        return results

    def count(self) -> int:
        return len(self._docs)


# ── Main pipeline ──────────────────────────────────────────────────────────────

class GraniteRAGPipeline:
    """
    End-to-end RAG pipeline:
    1. Index mental-health knowledge base into in-process vector store.
    2. Retrieve relevant context for each query.
    3. Augment prompt and generate via IBM Granite REST API.
    """

    _instance: "GraniteRAGPipeline | None" = None

    @classmethod
    def get_instance(cls) -> "GraniteRAGPipeline":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        logger.info("Initialising GraniteRAGPipeline…")
        self._init_vector_store()
        self._check_granite()

    # ── Vector store ──────────────────────────────────────────────────────────
    def _init_vector_store(self) -> None:
        try:
            self._store = _TFIDFVectorStore()
            docs = get_knowledge_base()
            self._store.add(docs)
            self.vector_store_ready = True
        except Exception as exc:
            logger.warning("Vector store init failed: %s — using keyword fallback.", exc)
            self._store = None
            self.vector_store_ready = False

    # ── Granite availability check ────────────────────────────────────────────
    def _check_granite(self) -> None:
        """Mark granite as potentially available — actual check happens on first call."""
        self.granite_available = True   # optimistic; flipped to False on first error

    # ── Retrieval ─────────────────────────────────────────────────────────────
    def retrieve(self, query: str, n_results: int = 4) -> list[dict[str, Any]]:
        if not self.vector_store_ready or self._store is None:
            return self._keyword_fallback_retrieve(query)
        results = self._store.query(query, n=n_results)
        return [
            {
                "content": r["content"],
                "title": r["title"],
                "category": r["category"],
                "score": r["score"],
            }
            for r in results
        ]

    def _keyword_fallback_retrieve(self, query: str) -> list[dict[str, Any]]:
        """Simple keyword overlap retrieval when embeddings are unavailable."""
        from knowledge_base import get_knowledge_base
        query_words = set(query.lower().split())
        scored = []
        for doc in get_knowledge_base():
            doc_words = set(doc["content"].lower().split())
            overlap = len(query_words & doc_words)
            if overlap:
                scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"content": d["content"], "title": d["title"], "category": d["category"], "score": s}
            for s, d in scored[:4]
        ] or [{"content": get_knowledge_base()[0]["content"],
               "title": get_knowledge_base()[0]["title"],
               "category": "general", "score": 0.0}]

    # ── Prompt construction ───────────────────────────────────────────────────
    def _build_prompt(
        self,
        user_message: str,
        retrieved_docs: list[dict],
        conversation_history: list[dict],
        risk_level: str,
    ) -> str:
        context_block = "\n\n".join(
            f"[Resource: {d['title']}]\n{d['content']}" for d in retrieved_docs
        )

        history_block = ""
        for turn in conversation_history[-6:]:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_block += f"{role}: {turn['content']}\n"

        crisis_instruction = ""
        if risk_level in ("HIGH", "CRITICAL"):
            crisis_instruction = (
                "\n[CRITICAL INSTRUCTION: The user may be in crisis. "
                "Express deep empathy, validate their pain, STRONGLY encourage them to "
                "contact a crisis helpline immediately, and provide helpline numbers. "
                "Do NOT minimise or dismiss their feelings.]\n"
            )
        elif risk_level == "MEDIUM":
            crisis_instruction = (
                "\n[INSTRUCTION: The user seems distressed. Be extra gentle, "
                "validate their feelings, and gently suggest professional support.]\n"
            )

        prompt = f"""<|system|>
You are MindBridge, a compassionate AI mental health support assistant powered by IBM Granite.
Your role is to provide empathetic, evidence-based mental health support and information.

CORE PRINCIPLES:
- Listen actively and respond with genuine empathy
- Validate emotions without judgment
- Share relevant coping strategies and resources
- Always remind users you are not a substitute for professional care
- For any crisis signals, immediately provide helpline numbers
- Never provide a method for self-harm or suicide
- Use warm, accessible language — never clinical or cold
{crisis_instruction}
RELEVANT KNOWLEDGE BASE CONTEXT:
{context_block}

CONVERSATION HISTORY:
{history_block}
<|user|>
{user_message}
<|assistant|>
"""
        return prompt

    # ── Generation ────────────────────────────────────────────────────────────
    def generate_response(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
        risk_level: str = "LOW",
    ) -> dict[str, Any]:
        if conversation_history is None:
            conversation_history = []

        retrieved_docs = self.retrieve(user_message, n_results=4)
        prompt = self._build_prompt(user_message, retrieved_docs, conversation_history, risk_level)

        response_text = self._call_granite(prompt, risk_level, retrieved_docs)

        if risk_level in ("MEDIUM", "HIGH", "CRITICAL") and SAFE_MESSAGING_DISCLAIMER not in response_text:
            response_text += f"\n\n{SAFE_MESSAGING_DISCLAIMER}"

        return {
            "response": response_text,
            "sources": [{"title": d["title"], "category": d["category"]} for d in retrieved_docs],
            "risk_level": risk_level,
        }

    def _call_granite(
        self,
        prompt: str,
        risk_level: str,
        retrieved_docs: list[dict],
    ) -> str:
        if self.granite_available:
            try:
                text = WatsonxClient.generate(prompt)
                return text
            except Exception as exc:
                logger.warning("Granite call failed (%s) — using fallback.", exc)
                self.granite_available = False

        return self._fallback_response(retrieved_docs, risk_level)

    # ── Fallback ──────────────────────────────────────────────────────────────
    def _fallback_response(self, retrieved_docs: list[dict], risk_level: str) -> str:
        if risk_level in ("HIGH", "CRITICAL"):
            return (
                "I can hear that you're going through something incredibly painful right now, "
                "and I want you to know that your feelings are completely valid. You are not alone.\n\n"
                "Please reach out to a crisis helpline right now — trained counselors are "
                "available 24/7 and will listen without judgment:\n\n"
                "🇮🇳 **India:**\n"
                "• iCall (TISS): **9152987821** (Mon–Sat, 8am–10pm)\n"
                "• Vandrevala Foundation: **1860-2662-345** (24/7)\n"
                "• AASRA: **9820466627** (24/7)\n\n"
                "🌍 **International:**\n"
                "• Crisis Text Line (US): Text HOME to **741741**\n"
                "• Samaritans (UK): **116 123**\n\n"
                "If you are in immediate danger, please call emergency services (112 / 911).\n\n"
                f"{SAFE_MESSAGING_DISCLAIMER}"
            )
        if retrieved_docs:
            top = retrieved_docs[0]
            return (
                f"Thank you for sharing that with me. I'm here to listen and support you.\n\n"
                f"**{top['title']}**\n\n{top['content']}\n\n"
                "Remember, reaching out is always a sign of strength. "
                "How are you feeling right now?\n\n"
                f"{SAFE_MESSAGING_DISCLAIMER}"
            )
        return (
            "Thank you for reaching out. I'm here to listen and support you. "
            "Could you tell me a little more about what you're experiencing? "
            "I want to make sure I understand how you're feeling and provide "
            "the most helpful response I can.\n\n"
            f"{SAFE_MESSAGING_DISCLAIMER}"
        )

    # ── Runtime document addition ─────────────────────────────────────────────
    def add_document(self, doc: dict) -> None:
        """Add a single document dict to the vector store at runtime."""
        if self.vector_store_ready and self._store:
            all_docs = get_knowledge_base() + [doc]
            self._store.add(all_docs)
