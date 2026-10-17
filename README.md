# MindBridge — Agentic AI for Mental Health Awareness & Suicide Prevention
# IBM Granite-powered RAG + Empathetic Conversational Agent

## 🏗️ Architecture Overview

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    MindBridge App (Streamlit)                │
│                                                             │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────┐  │
│  │  Chat Tab    │  │ Journal Tab   │  │ Resources Tab   │  │
│  └──────┬───────┘  └──────┬────────┘  └────────┬────────┘  │
│         └────────────────┬─┘                   │           │
│                          ▼                     │           │
│              ┌───────────────────────┐         │           │
│              │   MindBridgeAgent     │◄────────┘           │
│              │  (agent.py)           │                      │
│              └─────┬────────┬────────┘                      │
│                    │        │                               │
│         ┌──────────▼──┐ ┌───▼──────────────┐               │
│         │  Distress   │ │  RAG Pipeline    │               │
│         │  Detector   │ │  (Granite+Chroma)│               │
│         │  + Predictor│ │                  │               │
│         └──────────┬──┘ └───┬──────────────┘               │
│                    │        │                               │
│                    ▼        ▼                               │
│         ┌───────────────────────────────┐                  │
│         │    Crisis Intervention Mgr    │                  │
│         │  Helplines + Safety Plans     │                  │
│         └───────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
         IBM Watsonx (Granite 13B Chat)
         ChromaDB (Vector Store)
         Sentence Transformers (Embeddings)
```

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- IBM Watsonx account with API key

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure credentials
Edit `config.py` or set environment variables:
```bash
IBM_API_KEY=your_api_key
IBM_PROJECT_ID=your_project_id
IBM_WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

### 4. Run the application
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

---

## 📁 Project Structure

```
├── app.py                  # Main Streamlit UI (5 tabs)
├── agent.py                # MindBridgeAgent — session orchestration
├── rag_pipeline.py         # IBM Granite + ChromaDB RAG pipeline
├── distress_detector.py    # Risk analysis & prediction engine
├── crisis_manager.py       # Crisis resources, safety plans, SMS alerts
├── knowledge_base.py       # Curated mental health knowledge base
├── config.py               # Centralised configuration
├── requirements.txt        # Python dependencies
├── .streamlit/config.toml  # Streamlit server config
└── data/
    └── chroma_db/          # Persistent vector store (auto-created)
```

---

## 🧩 Key Components

### 1. RAG Pipeline (`rag_pipeline.py`)
- **ChromaDB** persistent vector store with cosine similarity
- **Sentence Transformers** (`all-MiniLM-L6-v2`) for embedding mental health documents
- **IBM Granite 13B Chat** for empathetic, knowledge-grounded response generation
- Graceful fallback when Granite API is unavailable
- 22 curated mental health knowledge base documents across 8 categories

### 2. Distress Detector (`distress_detector.py`)
- Multi-category keyword matching with severity weights
- 6 risk categories: `suicidal_ideation`, `crisis_signals`, `self_harm`,
  `severe_depression`, `moderate_distress`, `mild_distress`
- 5-tier risk classification: NONE → LOW → MEDIUM → HIGH → CRITICAL
- **RiskPredictor**: rolling window trend analysis for proactive alerts
- Contextual escalation detection across conversation history

### 3. MindBridgeAgent (`agent.py`)
- Session-aware conversation history management
- Integrates distress detection + RAG generation + crisis intervention
- Journal entry analysis with emotional tone assessment
- Mood timeline tracking for session insights
- Proactive check-in alerts when escalating risk trends detected

### 4. Crisis Intervention (`crisis_manager.py`)
- Complete India + International helpline database
- Stanley-Brown Safety Plan template
- 4 immediate coping cards (breathing, grounding, affirmations, distraction)
- Optional SMS alert dispatch via Twilio (with explicit consent)

### 5. Streamlit UI (`app.py`)
- **💬 Chat**: Real-time empathetic conversation with risk badges
- **📓 Journal**: Private journaling with emotional analysis
- **📚 Resources**: 10 topic-based RAG-powered educational cards
- **📊 Insights**: Interactive Plotly mood timeline + risk distribution chart
- **🆘 Crisis Help**: Helplines, coping cards, safety plan

---

## 🛡️ Safety & Ethics

- **Safe messaging guidelines** followed throughout (no method disclosure)
- **Explicit disclaimer** on every response: not a substitute for professional care
- **Consent-based** SMS alerts (Twilio integration)
- **Data minimisation**: conversation data stays in-session only (not persisted)
- Crisis resources displayed **immediately** for HIGH/CRITICAL risk signals
- Emergency services number (112/911) prominently featured

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | IBM Granite 13B Chat (`ibm/granite-13b-chat-v2`) |
| Cloud | IBM Watsonx.ai |
| Vector DB | ChromaDB (persistent, cosine similarity) |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Frontend | Streamlit 1.37 |
| Charts | Plotly |
| SMS Alerts | Twilio (optional) |
| Language | Python 3.10+ |

---

## ⚠️ Important Disclaimers

1. MindBridge is an **AI support tool**, not a licensed therapist or medical device.
2. It is **not a crisis service** — always call a helpline for real emergencies.
3. **Accuracy of risk detection** depends on explicit language — subtle signals may be missed.
4. Always work with a licensed mental health professional for diagnosis and treatment.

---

*Built for IBM Hackathon — Problem Statement No.2: Agentic AI for Mental Health Awareness & Suicide Prevention*

