"""
IBM Watsonx REST Client — Python 3.15 compatible
Wraps the Watsonx.ai REST API directly using `requests`,
avoiding the ibm-watsonx-ai SDK which caps at Python <3.15.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from config import (
    GENERATION_PARAMS,
    GRANITE_MODEL_ID,
    IBM_API_KEY,
    IBM_PROJECT_ID,
    IBM_WATSONX_URL,
)

logger = logging.getLogger(__name__)

# IAM token endpoint
_IAM_URL = "https://iam.cloud.ibm.com/identity/token"
# Watsonx generation endpoint
_GENERATE_URL = f"{IBM_WATSONX_URL}/ml/v1/text/generation?version=2023-05-29"


class WatsonxClient:
    """
    Lightweight IBM Watsonx.ai REST client.
    Handles IAM token acquisition with automatic refresh.
    """

    _token: str | None = None
    _token_expiry: float = 0.0

    @classmethod
    def _get_iam_token(cls) -> str:
        """Fetch (or return cached) IAM bearer token."""
        if cls._token and time.time() < cls._token_expiry - 60:
            return cls._token

        resp = requests.post(
            _IAM_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": IBM_API_KEY,
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        cls._token = payload["access_token"]
        cls._token_expiry = time.time() + payload.get("expires_in", 3600)
        logger.info("IAM token acquired (expires in %ds)", payload.get("expires_in", 3600))
        return cls._token

    @classmethod
    def generate(cls, prompt: str, model_id: str = GRANITE_MODEL_ID) -> str:
        """
        Call the Watsonx text generation endpoint and return the generated text.
        Raises on HTTP errors so the caller can fall back gracefully.
        """
        token = cls._get_iam_token()

        body: dict[str, Any] = {
            "model_id": model_id,
            "input": prompt,
            "parameters": {
                "decoding_method": GENERATION_PARAMS["decoding_method"],
                "max_new_tokens": GENERATION_PARAMS["max_new_tokens"],
                "min_new_tokens": GENERATION_PARAMS["min_new_tokens"],
                "stop_sequences": GENERATION_PARAMS["stop_sequences"],
                "repetition_penalty": GENERATION_PARAMS["repetition_penalty"],
                "temperature": GENERATION_PARAMS["temperature"],
            },
            "project_id": IBM_PROJECT_ID,
        }

        resp = requests.post(
            _GENERATE_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=body,
            timeout=60,
        )
        resp.raise_for_status()

        data = resp.json()
        generated = data["results"][0]["generated_text"]
        logger.debug("Granite generated %d chars.", len(generated))
        return generated.strip()
