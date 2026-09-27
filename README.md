# Mizo 3.0 (Mikaza) — AI-Powered Personalized Learning Tutor & ESP32-S3 Cloud Server

![Mizo 3.0 Architecture](https://img.shields.io/badge/Architecture-FastAPI%20%7C%20ESP32--S3%20%7C%20Llama%203.3%2070B%20%7C%20Edge--TTS-6366f1?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-06b6d4?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-10b981?style=for-the-badge)

**Mizo 3.0 (Mikaza)** is a full-stack, production-grade AI educational assistant and cloud server designed to power robotic tutors built on the **ESP32-S3 microcontroller**. It combines real-time speech recognition, multi-provider LLM reasoning (with **Llama 3.3 70B** as primary), persistent long-term student memory, dynamic speech assessment, RAG document search, and low-latency audio streaming.

---

## 🌟 Key Features & Production Architecture

- **Primary LLM: Llama 3.3 70B**: Powered by Groq Cloud (`llama-3.3-70b-versatile`) with automatic dynamic fallback cascade to **OpenAI** (`gpt-4o-mini`), **Qwen 2.5 72B** (OpenRouter), and **Local Ollama**. Zero hardcoded or mock bot responses.
- **Single Source of Truth Database & Memory Engine**: Persistent SQLite store (`data/mizo.db`) tracking unified learner profiles, onboarding calibration, active session lifecycles, mistake logs, and conversational history.
- **Session Lifecycle & Metric Separation**: Starting a new session creates a fresh session ID and initializes current-session metrics (fluency, grammar accuracy, vocabulary richness, pacing) to zero while preserving cumulative historical metrics to guide AI personalization.
- **Dynamic First-Session Onboarding**: First-time learners undergo greeting, interests/goals discovery, and baseline speech calibration before unlocking main coaching modes.
- **Multi-Tier Voice Pipeline**:
  - **STT**: Groq Whisper (`whisper-large-v3`) $\rightarrow$ OpenAI Whisper (`whisper-1`) $\rightarrow$ Faster-Whisper local fallback.
  - **TTS**: Edge-TTS neural voice generator delivering crisp **16kHz 16-bit Mono WAV** audio directly streamable to the ESP32 **MAX98357A I2S DAC amplifier**.
- **Dynamic Intent & Core Modes**:
  - 🎙️ **Communication Coach**: Conversational fluency, gentle grammar repair, rich vocabulary suggestions.
  - 📖 **Academic Subject Tutor**: Dynamic concept explanations with RAG PDF textbook retrieval.
  - 🎯 **Speech Trainer**: Presentation rehearsal, filler word elimination, and pacing feedback.
- **Modern Admin Dashboard & ESP32 Simulator**: Web UI with real-time API connection health checks, session management, mistake inspectors, and a virtual voice testbench.

---

## 🏗️ System Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │               MIZO CLOUD SERVER              │
                       │                                              │
                       │  ┌────────────────────────────────────────┐  │
                       │  │   Admin Dashboard & ESP32 Simulator    │  │
                       │  │   • Live AI Provider Health Checks     │  │
                       │  │   • Student Memory & Weakness Logs     │  │
                       │  │   • Current vs Historical Progress     │  │
                       │  │   • Session Lifecycle Management       │  │
                       │  │   • RAG Document Indexing (PDF/Text)   │  │
                       │  └────────────────────────────────────────┘  │
                       │                                              │
                       │  ┌────────────────────────────────────────┐  │
                       │  │   FastAPI Backend (Async Core)         │  │
                       │  │   • STT: Groq Whisper / Faster-Whisper │  │
                       │  │   • Intent Engine & Dynamic Onboarding │  │
                       │  │   • LLM Router: Llama 3.3 Primary      │  │
                       │  │     (Fallback: OpenAI / Qwen / Ollama) │  │
                       │  │   • Personalization & Memory Engine    │  │
                       │  │   • TTS: Edge-TTS 16kHz PCM WAV        │  │
                       │  │   • ESP32 Endpoints: /audio & /chat    │  │
                       │  └────────────────────────────────────────┘  │
                       │                                              │
                       │  ┌────────────────────────────────────────┐  │
                       │  │   SQLite Single Source of Truth        │  │
                       │  │   (Students, Sessions, Mistakes, Logs) │  │
                       │  └────────────────────────────────────────┘  │
                       └──────────────────────▲───────────────────────┘
                                              │
                                  Wi-Fi / HTTPS (REST / Audio)
                                              │
                       ┌──────────────────────▼───────────────────────┐
                       │                   ESP32-S3                   │
                       │                                              │
                       │  🎙️ INMP441 Microphone (I2S input)           │
                       │  🔊 MAX98357A Amp + Speaker (I2S output)     │
                       └──────────────────────────────────────────────┘
```

---

## 🚀 Local Software-Only Onboarding & Web Application

Mizo / Mikaza 3.0 provides a complete software-only onboarding and voice coaching experience that runs locally on your laptop without requiring physical ESP32 or robot hardware.

### 1. Configure API Keys Safely

Create a local `.env` file from the example template:

```bash
cp .env.example .env
```

Configure your preferred LLM and Speech providers in `.env`:
```ini
# Primary High-Speed LLM & Whisper STT
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_WHISPER_MODEL=whisper-large-v3

# Fallback LLMs
OPENAI_API_KEY=sk-your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
QWEN_API_KEY=sk-or-v1-your_openrouter_key_here
OLLAMA_BASE_URL=http://localhost:11434

# Edge-TTS Neural Voice
DEFAULT_TTS_VOICE=en-US-GuyNeural
DEFAULT_TTS_RATE=+0%
```

> [!NOTE]
> **API Key Safety Guarantee**:
> - Real API keys are never printed in logs, test output, or API responses.
> - The web dashboard displays only masked credentials (`gsk_...****...`).
> - Empty API key form submissions in the UI never overwrite or erase existing keys.
> - `.env` is included in `.gitignore` and is never committed to Git.

---

### 2. Start the Local Server & Open the Browser

Start the FastAPI application:

```bash
python run_server.py
```

Open your web browser and navigate to:
**[http://localhost:8000](http://localhost:8000)**

---

### 3. Browser Microphone & Audio Pipeline

The application uses standard Web Audio APIs (`navigator.mediaDevices.getUserMedia` & `MediaRecorder`) to capture voice directly from your laptop's microphone:

1. **Microphone Permission**: When clicking the **Record** button for the first time, allow browser microphone access.
2. **Audio Format**: Voice is recorded as standard Opus-encoded WebM or WAV. The backend dynamic MIME mapper auto-detects and packages the audio bytes appropriately for Groq/OpenAI Whisper STT without format mismatch errors.
3. **Recording Timer HUD**: A live timer shows progress towards the **~60-second assessment window** (up to 120s max).
4. **Short Recording Warning**: If you stop speaking in under 25 seconds, an alert dialog will notify you that the sample is short and give you the choice to *Continue Recording* or *Submit Anyway*.
5. **Text-Only Fallback**: If your microphone is unavailable or permission is denied, type your responses in the chat input or click **"Skip Audio Assessment"** to complete text profile onboarding without generating a fake baseline.

---

### 4. Persistent Learner Data & Storage

All learner information, conversation messages, speaking baselines, and mistakes are stored in the persistent SQLite database:
`data/mizo.db`

- **Database Safety**: The database schema uses non-destructive additive migrations (`_ensure_column`). The database is never dropped or recreated on application startup.
- **Backup Location**: A safe pre-migration backup is stored at `data/mizo.db.bak`.
- **Audio Files**: Microphone audio uploads are processed in memory / temporary cache and deleted immediately after transcription.

---

### 5. Data Preservation & Lifecycle Actions

| Action | What It Does | Profile & Baseline | Conversation History | API Keys & Settings |
| :--- | :--- | :--- | :--- | :--- |
| **Server Restart** | Resumes application execution without changing database state. | Preserved. Incomplete onboarding resumes exact step. | Preserved | Preserved |
| **New Session** (`⚡ New Session`) | Creates a fresh session ID. Current session metrics reset to 0%. | Preserved intact | Preserved and viewable in History | Preserved |
| **Restart Onboarding** (`🔄 Restart Onboarding`) | Archives a snapshot of the current profile into `profile_archives`, increments version, and restarts onboarding from Step 1. | Reset to fresh `ask_name` step | **Preserved intact** (past sessions & assessments remain reviewable) | Preserved |
| **Delete Learner Data** (`🗑️ Delete Learner Data`) | Destructive action requiring explicit confirmation modal. Permanently wipes learner profile, conversation history, assessments, and mistakes. | Reset to default | Deleted | **Strictly Preserved** (.env and provider keys never erased) |

---

### 6. Speaking Assessment: Capabilities & Limitations

The 1-minute speaking assessment provides **formative, diagnostic feedback** across 6 linguistic dimensions:

- **Grammar Accuracy (0–100%)**: Evaluates sentence structure, tense consistency, and verb agreement from the transcript.
- **Vocabulary Richness (0–100%)**: Evaluates word variety, technical terminology, and contextual word choice.
- **Fluency & Flow (0–100%)**: Evaluates speaking pace, word density, and presence of verbal fillers.
- **Pronunciation & Intelligibility (0–100%)**: *Note: Evaluated from speech recognition intelligibility and transcript flow. Acoustic phonetics are not measured directly from transcript alone.*
- **Delivery & Confidence (0–100%)**: Evaluates observable delivery features (sentence completeness, assertiveness of structure) without psychological speculation.
- **Overall Communication Level**: Formative calibration (Beginner, Elementary, Intermediate, Upper Intermediate, Advanced).

---

## 🧪 Automated Testing

To run the complete automated test suite (including mocked LLM/STT/TTS unit tests and API integration tests):

```bash
pytest -v tests/
```

Run specific test modules:
```bash
pytest -v tests/test_software_onboarding_and_history.py
```

---

## 📄 License
MIT License. Built for educational AI robotics innovation.

