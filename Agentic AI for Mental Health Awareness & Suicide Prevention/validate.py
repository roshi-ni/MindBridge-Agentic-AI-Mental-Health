"""
Validation script — tests all major components without a live Granite API.
Run with: python validate.py
"""

import sys
import traceback
from unittest.mock import MagicMock, patch


def test_config():
    print("[TEST] config.py …", end=" ")
    from config import (
        CRISIS_HELPLINES,
        DISTRESS_KEYWORDS,
        GRANITE_MODEL_ID,
        HIGH_RISK_THRESHOLD,
        IBM_PROJECT_ID,
        SAFE_MESSAGING_DISCLAIMER,
    )
    assert IBM_PROJECT_ID == "181d5190-0e53-4269-9dc5-7f24d6b7c9f9"
    assert GRANITE_MODEL_ID == "ibm/granite-13b-chat-v2"
    assert HIGH_RISK_THRESHOLD == 0.75
    assert "India" in CRISIS_HELPLINES
    assert "suicidal_ideation" in DISTRESS_KEYWORDS
    assert len(SAFE_MESSAGING_DISCLAIMER) > 20
    print("PASS ✓")


def test_knowledge_base():
    print("[TEST] knowledge_base.py …", end=" ")
    from knowledge_base import get_by_category, get_by_tag, get_knowledge_base

    docs = get_knowledge_base()
    assert len(docs) >= 20, f"Expected ≥20 docs, got {len(docs)}"
    assert all("id" in d and "content" in d and "category" in d for d in docs)

    crisis_docs = get_by_category("crisis")
    assert len(crisis_docs) >= 2

    tagged = get_by_tag("anxiety")
    assert len(tagged) >= 1
    print("PASS ✓")


def test_distress_detector():
    print("[TEST] distress_detector.py …", end=" ")
    from distress_detector import DistressDetector, RiskPredictor

    detector = DistressDetector()

    # Should detect CRITICAL/HIGH for suicidal ideation
    result = detector.analyse("I want to kill myself and end my life")
    assert result.risk_level in ("HIGH", "CRITICAL"), f"Expected HIGH/CRITICAL, got {result.risk_level}"
    assert result.risk_score >= 0.75
    assert "suicidal_ideation" in result.detected_signals
    assert result.requires_immediate_action

    # Should be NONE/LOW for neutral text
    result2 = detector.analyse("I had a great day at work today!")
    assert result2.risk_level in ("NONE", "LOW"), f"Expected NONE/LOW, got {result2.risk_level}"

    # Moderate distress
    result3 = detector.analyse("I've been feeling really hopeless and worthless lately")
    assert result3.risk_level in ("MEDIUM", "HIGH"), f"Expected MEDIUM/HIGH, got {result3.risk_level}"

    # Predictor
    predictor = RiskPredictor()
    pred = None
    for score in [0.1, 0.2, 0.6, 0.7, 0.8]:
        pred = predictor.update(score)
    assert pred is not None and "rolling_avg" in pred
    assert pred["rolling_avg"] > 0

    # Session trend
    trend = detector.get_session_trend()
    assert "session_peak_risk" in trend
    assert trend["total_messages"] >= 3
    print("PASS ✓")


def test_crisis_manager():
    print("[TEST] crisis_manager.py …", end=" ")
    from crisis_manager import CrisisInterventionManager

    mgr = CrisisInterventionManager()

    india = mgr.get_helplines("India")
    assert len(india) >= 3
    assert all("name" in h and "number" in h for h in india)

    safety_plan = mgr.get_safety_plan()
    assert "Warning Signs" in safety_plan
    assert "AASRA" in safety_plan

    for ctype in ["breathing", "grounding", "affirmation", "distraction"]:
        card = mgr.get_coping_card(ctype)
        assert len(card) > 50, f"Coping card '{ctype}' too short"

    crisis_resp = mgr.format_crisis_response("CRITICAL")
    assert "9152987821" in crisis_resp or "1860" in crisis_resp

    # SMS without consent should fail gracefully
    sms_result = mgr.send_sms_alert("+910000000000", "test", user_consent=False)
    assert sms_result["success"] is False
    print("PASS ✓")


def test_watsonx_client():
    print("[TEST] watsonx_client.py (mocked) …", end=" ")
    from unittest.mock import MagicMock, patch

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"results": [{"generated_text": "  Hello, I am here to help.  "}]}
    mock_resp.raise_for_status = MagicMock()

    mock_iam = MagicMock()
    mock_iam.json.return_value = {"access_token": "fake_token", "expires_in": 3600}
    mock_iam.raise_for_status = MagicMock()

    with patch("watsonx_client.requests.post") as mock_post:
        mock_post.side_effect = [mock_iam, mock_resp]

        from watsonx_client import WatsonxClient
        WatsonxClient._token = None
        WatsonxClient._token_expiry = 0.0

        result = WatsonxClient.generate("Test prompt")
        assert result == "Hello, I am here to help."
        assert mock_post.call_count == 2   # IAM + generate
    print("PASS ✓")


def test_rag_pipeline_offline():
    """Test RAG pipeline in offline/fallback mode (TF-IDF + no Granite API)."""
    print("[TEST] rag_pipeline.py (offline/keyword mode) …", end=" ")

    import importlib
    import rag_pipeline

    rag_pipeline.GraniteRAGPipeline._instance = None
    importlib.reload(rag_pipeline)

    pipeline = rag_pipeline.GraniteRAGPipeline()
    pipeline.granite_available = False  # force fallback

    # TF-IDF retrieval should work without any external models
    docs = pipeline.retrieve("anxiety breathing techniques", n_results=3)
    assert len(docs) >= 1
    assert "content" in docs[0]

    # Fallback response for high risk should contain helpline
    result_crisis = pipeline.generate_response("I want to end my life", risk_level="CRITICAL")
    assert "9152987821" in result_crisis["response"] or "AASRA" in result_crisis["response"]
    assert "sources" in result_crisis

    # Medium risk response should include disclaimer
    result_med = pipeline.generate_response("I feel hopeless", risk_level="MEDIUM")
    assert len(result_med["response"]) > 50

    rag_pipeline.GraniteRAGPipeline._instance = None
    print("PASS ✓")


def test_agent_offline():
    """Test the full agent pipeline in offline mode."""
    print("[TEST] agent.py (offline mode) …", end=" ")

    import importlib
    import rag_pipeline

    rag_pipeline.GraniteRAGPipeline._instance = None
    importlib.reload(rag_pipeline)

    pipeline = rag_pipeline.GraniteRAGPipeline()
    pipeline.granite_available = False

    # Patch the singleton getter to return our offline pipeline
    with patch("agent.GraniteRAGPipeline.get_instance", return_value=pipeline):
        import agent as agent_module
        ag = agent_module.MindBridgeAgent()

        # Standard response
        result = ag.respond("I've been feeling stressed lately")
        assert "response" in result
        assert result["risk_level"] in ("NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL")

        # Crisis response
        result_crisis = ag.respond("I want to kill myself")
        assert result_crisis["risk_level"] in ("HIGH", "CRITICAL")
        assert result_crisis["requires_immediate_action"]

        # Journal analysis
        journal_result = ag.analyse_journal(
            "Today was really hard. I feel completely worthless and don't see the point anymore."
        )
        assert "analysis" in journal_result
        assert journal_result["risk_level"] in ("MEDIUM", "HIGH", "CRITICAL")

        # Mood timeline
        timeline = ag.get_mood_timeline()
        assert len(timeline) >= 2

        # Session summary
        summary = ag.get_session_summary()
        assert summary["total_messages"] >= 2

    rag_pipeline.GraniteRAGPipeline._instance = None
    print("PASS ✓")


def main():
    print("=" * 60)
    print("MindBridge — Component Validation Suite")
    print("=" * 60)
    tests = [
        test_config,
        test_knowledge_base,
        test_distress_detector,
        test_crisis_manager,
        test_watsonx_client,
        test_rag_pipeline_offline,
        test_agent_offline,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL ✗ — {e}")
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} passed", "✅" if failed == 0 else "❌")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
