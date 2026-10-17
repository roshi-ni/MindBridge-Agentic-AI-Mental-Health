"""
MindBridge — Agentic AI for Mental Health Awareness & Suicide Prevention
Main Streamlit Application
IBM Granite + ChromaDB + RAG Pipeline
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

# ── Page config (MUST be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="MindBridge — AI Mental Health Support",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://icallhelpline.org/",
        "About": "MindBridge — Agentic AI for Mental Health powered by IBM Granite",
    },
)

from agent import MindBridgeAgent  # noqa: E402  (must come after set_page_config)
from crisis_manager import CrisisInterventionManager  # noqa: E402

logging.basicConfig(level=logging.INFO)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Global */
    .main { background-color: #f0f4f8; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }

    /* Header banner */
    .mindbridge-header {
        background: linear-gradient(135deg, #1a56db 0%, #7c3aed 100%);
        color: white;
        padding: 1.2rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .mindbridge-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .mindbridge-header p  { margin: 0.3rem 0 0; font-size: 0.95rem; opacity: 0.9; }

    /* Chat bubbles */
    .chat-user {
        background: #dbeafe;
        border-left: 4px solid #2563eb;
        padding: 0.75rem 1rem;
        border-radius: 0 12px 12px 12px;
        margin: 0.5rem 0;
        max-width: 85%;
    }
    .chat-assistant {
        background: #ffffff;
        border-left: 4px solid #7c3aed;
        padding: 0.75rem 1rem;
        border-radius: 0 12px 12px 12px;
        margin: 0.5rem 0;
        max-width: 95%;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* Risk badges */
    .risk-none     { background:#d1fae5; color:#065f46; padding:2px 8px; border-radius:8px; font-size:0.75rem; font-weight:600; }
    .risk-low      { background:#dbeafe; color:#1e40af; padding:2px 8px; border-radius:8px; font-size:0.75rem; font-weight:600; }
    .risk-medium   { background:#fef3c7; color:#92400e; padding:2px 8px; border-radius:8px; font-size:0.75rem; font-weight:600; }
    .risk-high     { background:#fee2e2; color:#991b1b; padding:2px 8px; border-radius:8px; font-size:0.75rem; font-weight:600; }
    .risk-critical { background:#fce7f3; color:#9d174d; padding:2px 8px; border-radius:8px; font-size:0.75rem; font-weight:600; animation: pulse 1s infinite; }

    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }

    /* Crisis banner */
    .crisis-banner {
        background: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.75rem 0;
    }

    /* Disclaimer */
    .disclaimer {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.82rem;
        color: #78350f;
        margin: 0.5rem 0 1rem;
    }

    /* Resource card */
    .resource-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Sidebar */
    .css-1d391kg { background-color: #1e1b4b; }
    section[data-testid="stSidebar"] { background: #1e1b4b; }
    section[data-testid="stSidebar"] * { color: white !important; }

    /* Input */
    .stTextArea textarea {
        border-radius: 10px;
        border: 1.5px solid #a5b4fc;
        font-size: 0.95rem;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state initialisation ───────────────────────────────────────────────
def _init_session() -> None:
    if "agent" not in st.session_state:
        with st.spinner("Initialising MindBridge AI (loading models…)"):
            st.session_state.agent = MindBridgeAgent()
            st.session_state.crisis_mgr = CrisisInterventionManager()
    if "messages" not in st.session_state:
        st.session_state.messages = []          # list of {role, content, risk_level, ts}
    if "mood_data" not in st.session_state:
        st.session_state.mood_data = []
    if "session_started" not in st.session_state:
        st.session_state.session_started = datetime.now()
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "chat"
    if "user_consent_sms" not in st.session_state:
        st.session_state.user_consent_sms = False
    if "journal_result" not in st.session_state:
        st.session_state.journal_result = None


_init_session()

agent: MindBridgeAgent = st.session_state.agent
crisis_mgr: CrisisInterventionManager = st.session_state.crisis_mgr

# ── Helper: Risk badge HTML ────────────────────────────────────────────────────
def risk_badge(level: str) -> str:
    icons = {"NONE": "✅", "LOW": "💙", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "🆘"}
    css_class = f"risk-{level.lower()}"
    return f'<span class="{css_class}">{icons.get(level, "❓")} {level}</span>'


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:1rem 0'>"
        "<span style='font-size:2.5rem'>💙</span>"
        "<h2 style='margin:0.3rem 0 0'>MindBridge</h2>"
        "<p style='font-size:0.8rem;opacity:0.7'>Powered by IBM Granite</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Navigation
    nav = st.radio(
        "Navigate",
        ["💬 Chat", "📓 Journal", "📚 Resources", "📊 Insights", "🆘 Crisis Help"],
        label_visibility="collapsed",
    )

    st.divider()

    # Session info
    session_summary = agent.get_session_summary()
    st.markdown("**Session Info**")
    col_a, col_b = st.columns(2)
    col_a.metric("Messages", session_summary.get("total_messages", 0))
    col_b.metric(
        "Peak Risk",
        session_summary.get("session_peak_risk", "NONE"),
    )

    st.divider()

    # Quick crisis access
    st.markdown("**🆘 Quick Help**")
    st.markdown(
        "**iCall (India):** `9152987821`  \n"
        "**Vandrevala:** `1860-2662-345`  \n"
        "**Emergency:** `112`",
        unsafe_allow_html=True,
    )

    st.divider()

    if st.button("🔄 New Session", use_container_width=True):
        agent.reset_session()
        st.session_state.messages = []
        st.session_state.mood_data = []
        st.session_state.journal_result = None
        st.rerun()

    st.markdown(
        "<p style='font-size:0.7rem;opacity:0.5;text-align:center;margin-top:1rem'>"
        "⚠️ Not a substitute for professional care</p>",
        unsafe_allow_html=True,
    )


# ── Main header ────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="mindbridge-header">
    <h1>💙 MindBridge</h1>
    <p>Agentic AI for Mental Health Awareness & Suicide Prevention &nbsp;|&nbsp; Powered by IBM Granite</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Disclaimer bar ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> MindBridge is an AI support tool — '
    "it is <strong>NOT</strong> a substitute for professional mental health care. "
    "In a crisis, please call a helpline or emergency services immediately.</div>",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: CHAT
# ═══════════════════════════════════════════════════════════════════════════════
if nav == "💬 Chat":
    # Show welcome message on first load
    if not st.session_state.messages:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": agent.WELCOME_MESSAGE,
                "risk_level": "NONE",
                "ts": datetime.now().strftime("%H:%M"),
            }
        )

    # Render chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">'
                    f'<strong>You</strong> <small style="color:#6b7280">{msg.get("ts","")}</small>'
                    f'<br>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                badge = risk_badge(msg.get("risk_level", "NONE"))
                st.markdown(
                    f'<div class="chat-assistant">'
                    f'<strong>MindBridge 💙</strong> {badge} '
                    f'<small style="color:#6b7280">{msg.get("ts","")}</small>'
                    f'<br><br>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

    st.divider()

    # Input area
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        user_input = col1.text_area(
            "Message",
            placeholder="How are you feeling today? Share anything on your mind…",
            height=80,
            label_visibility="collapsed",
        )
        submitted = col2.form_submit_button("Send 💙", use_container_width=True)

    if submitted and user_input.strip():
        ts = datetime.now().strftime("%H:%M")

        # Add user message
        st.session_state.messages.append(
            {"role": "user", "content": user_input.strip(), "risk_level": "NONE", "ts": ts}
        )

        # Show spinner while generating
        with st.spinner("MindBridge is thinking…"):
            result = agent.respond(user_input.strip())

        # Track mood data
        st.session_state.mood_data.append(
            {
                "time": ts,
                "score": result["risk_score"],
                "level": result["risk_level"],
            }
        )

        # Proactive alert (non-crisis)
        if result.get("proactive_alert"):
            st.info(f"💙 **Check-In:** {result['proactive_alert']}")

        # Crisis banner
        if result["risk_level"] in ("HIGH", "CRITICAL"):
            st.markdown(
                f'<div class="crisis-banner">'
                f"🆘 <strong>High-Risk Signal Detected</strong><br>"
                f"{crisis_mgr.format_crisis_response(result['risk_level'])}"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Add assistant message
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["response"],
                "risk_level": result["risk_level"],
                "ts": ts,
            }
        )

        # Show sources
        if result.get("sources"):
            with st.expander("📚 Knowledge sources used", expanded=False):
                for src in result["sources"]:
                    st.markdown(f"- **{src['title']}** _{src['category']}_")

        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: JOURNAL
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "📓 Journal":
    st.subheader("📓 Journaling Space")
    st.markdown(
        "Use this private space to write freely. MindBridge will analyse your entry "
        "and offer compassionate, personalised insights."
    )

    journal_text = st.text_area(
        "Your journal entry",
        placeholder=(
            "Write about how you're feeling, what happened today, what's on your mind… "
            "There's no right or wrong way to journal."
        ),
        height=250,
    )

    col1, col2 = st.columns([2, 1])
    analyse_btn = col1.button("✨ Analyse & Get Insights", type="primary", use_container_width=True)
    clear_btn = col2.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        st.session_state.journal_result = None
        st.rerun()

    if analyse_btn and journal_text.strip():
        with st.spinner("Analysing your entry with empathy…"):
            result = agent.analyse_journal(journal_text.strip())
            st.session_state.journal_result = result

    if st.session_state.journal_result:
        res = st.session_state.journal_result
        st.divider()

        # Risk level display
        level = res["risk_level"]
        badge_html = risk_badge(level)
        st.markdown(f"**Emotional Tone Assessment:** {badge_html}", unsafe_allow_html=True)

        if res["detected_signals"]:
            st.markdown(
                f"**Signals detected:** `{'`, `'.join(res['detected_signals'])}`"
            )

        # Crisis banner for high-risk journal
        if level in ("HIGH", "CRITICAL"):
            st.error(
                "🆘 Your journal entry contains signals of significant distress. "
                "Please reach out to a crisis helpline — you don't have to carry this alone.\n\n"
                "**iCall (TISS): 9152987821** | **Vandrevala: 1860-2662-345** | **Emergency: 112**"
            )

        st.markdown("### 💙 Personalised Insights")
        st.markdown(res["analysis"])

        if res.get("sources"):
            with st.expander("📚 Resources used"):
                for src in res["sources"]:
                    st.markdown(f"- **{src['title']}** _{src['category']}_")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: RESOURCES
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "📚 Resources":
    st.subheader("📚 Mental Health Resources & Education")

    topics = {
        "🧘 Mindfulness & Meditation": "mindfulness meditation techniques",
        "😰 Managing Anxiety": "anxiety management coping strategies",
        "😔 Understanding Depression": "depression symptoms and treatment",
        "😴 Sleep & Mental Health": "sleep hygiene mental health",
        "🏃 Exercise & Wellbeing": "exercise mental health benefits",
        "🍎 Nutrition & Mood": "nutrition diet mental health",
        "🤝 Social Connection": "social connection loneliness mental health",
        "🔥 Burnout Recovery": "burnout recovery workplace stress",
        "💬 Finding a Therapist": "types of therapy finding professional help",
        "🛡️ Crisis & Safety Planning": "safety planning suicide prevention crisis",
    }

    st.markdown("Select a topic to get evidence-based insights powered by IBM Granite + RAG:")

    cols = st.columns(2)
    for idx, (label, query) in enumerate(topics.items()):
        if cols[idx % 2].button(label, use_container_width=True):
            with st.spinner(f"Loading resources for '{label}'…"):
                res = agent.get_resources_by_topic(query)
            st.markdown(f"### {label}")
            st.markdown(res["response"])
            if res.get("sources"):
                with st.expander("📚 Sources"):
                    for src in res["sources"]:
                        st.markdown(f"- **{src['title']}** _{src['category']}_")
            st.markdown(
                '<div class="disclaimer">⚠️ Information provided for educational purposes only. '
                "Consult a mental health professional for personalised advice.</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("🚨 Crisis Coping Cards")
    st.markdown("Immediate techniques for difficult moments:")

    coping_cols = st.columns(2)
    coping_types = [
        ("🌬️ Box Breathing", "breathing"),
        ("🌳 5-4-3-2-1 Grounding", "grounding"),
        ("💬 Affirmations", "affirmation"),
        ("🎯 Distraction Techniques", "distraction"),
    ]
    for i, (label, ctype) in enumerate(coping_types):
        with coping_cols[i % 2]:
            with st.expander(label):
                st.markdown(crisis_mgr.get_coping_card(ctype))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "📊 Insights":
    st.subheader("📊 Session Insights & Mood Trends")

    summary = agent.get_session_summary()
    mood_data = st.session_state.mood_data

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Messages", summary.get("total_messages", 0))
    m2.metric("Peak Risk", summary.get("session_peak_risk", "NONE"))
    m3.metric(
        "Avg Risk Score",
        f"{summary.get('avg_score', 0.0):.2f}",
    )
    m4.metric(
        "Risk Trend",
        summary.get("risk_trend", "—").replace("_", " ").title(),
    )

    if mood_data:
        st.markdown("### 📈 Risk Score Over Conversation")
        times = [d["time"] for d in mood_data]
        scores = [d["score"] for d in mood_data]
        levels = [d["level"] for d in mood_data]

        # Colour-coded scatter + line
        colour_map = {
            "NONE": "#10b981",
            "LOW": "#3b82f6",
            "MEDIUM": "#f59e0b",
            "HIGH": "#ef4444",
            "CRITICAL": "#9d174d",
        }
        colours = [colour_map.get(l, "#6b7280") for l in levels]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(scores) + 1)),
                y=scores,
                mode="lines+markers",
                line=dict(color="#6366f1", width=2),
                marker=dict(color=colours, size=10, line=dict(width=1, color="white")),
                text=levels,
                hovertemplate="<b>Message %{x}</b><br>Score: %{y:.3f}<br>Level: %{text}<extra></extra>",
                name="Risk Score",
            )
        )
        # Risk threshold bands
        fig.add_hrect(y0=0.75, y1=1.0, fillcolor="#fce7f3", opacity=0.3, line_width=0, annotation_text="CRITICAL/HIGH")
        fig.add_hrect(y0=0.45, y1=0.75, fillcolor="#fef3c7", opacity=0.3, line_width=0, annotation_text="MEDIUM")
        fig.add_hrect(y0=0.20, y1=0.45, fillcolor="#dbeafe", opacity=0.3, line_width=0, annotation_text="LOW")

        fig.update_layout(
            xaxis_title="Message Number",
            yaxis_title="Risk Score (0–1)",
            yaxis=dict(range=[0, 1.05]),
            height=380,
            plot_bgcolor="#f9fafb",
            paper_bgcolor="#ffffff",
            margin=dict(t=30, b=40, l=50, r=30),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Risk level distribution pie
        st.markdown("### 🎯 Risk Level Distribution")
        from collections import Counter

        dist = Counter(levels)
        pie_colours = [colour_map.get(k, "#6b7280") for k in dist.keys()]
        fig2 = go.Figure(
            go.Pie(
                labels=list(dist.keys()),
                values=list(dist.values()),
                marker_colors=pie_colours,
                hole=0.4,
            )
        )
        fig2.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Start a conversation in the Chat tab to see your mood insights here. 💙")

    # Safety plan download
    st.divider()
    st.subheader("🛡️ Personal Safety Plan")
    st.markdown(
        "A safety plan is a tool created with your mental health professional. "
        "Below is a template you can print and fill in:"
    )
    st.markdown(crisis_mgr.get_safety_plan())


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: CRISIS HELP
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "🆘 Crisis Help":
    st.subheader("🆘 Crisis Help & Immediate Support")

    st.error(
        "**If you are in immediate danger, call emergency services: 112 (India) | 911 (US) | 999 (UK)**"
    )

    st.markdown("### You Are Not Alone 💙")
    st.markdown(
        "Suicidal thoughts and overwhelming pain are more common than you might think — "
        "and they are treatable. Crisis counselors are available 24/7 and are trained to "
        "listen without judgment. Calling a helpline is one of the bravest things you can do."
    )

    # Helplines tabs
    tab_india, tab_intl = st.tabs(["🇮🇳 India", "🌍 International"])

    with tab_india:
        for h in crisis_mgr.get_helplines("India"):
            with st.container():
                st.markdown(
                    f'<div class="resource-card">'
                    f"<strong>{h['name']}</strong><br>"
                    f"📞 <code>{h['number']}</code><br>"
                    f"🕐 {h['available']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    with tab_intl:
        for h in crisis_mgr.get_helplines("International"):
            with st.container():
                st.markdown(
                    f'<div class="resource-card">'
                    f"<strong>{h['name']}</strong><br>"
                    f"📞 <code>{h['number']}</code><br>"
                    f"🕐 {h['available']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.divider()

    # Immediate coping
    st.markdown("### ⚡ Immediate Coping Techniques")
    st.markdown("While you gather the courage to reach out, try one of these:")

    coping_cols = st.columns(2)
    with coping_cols[0]:
        st.markdown("#### 🌬️ Box Breathing")
        st.markdown(crisis_mgr.get_coping_card("breathing"))
    with coping_cols[1]:
        st.markdown("#### 🌳 Grounding")
        st.markdown(crisis_mgr.get_coping_card("grounding"))

    with st.expander("💬 Affirmations"):
        st.markdown(crisis_mgr.get_coping_card("affirmation"))
    with st.expander("🎯 Distraction Techniques"):
        st.markdown(crisis_mgr.get_coping_card("distraction"))

    st.divider()
    st.markdown("### 🛡️ Safety Planning")
    st.info(
        "A safety plan created with a therapist is one of the most effective tools "
        "for preventing suicide. Go to **📊 Insights** to see the full template."
    )
