# Mizo 3.0 — AI-Powered Personalized Learning Tutor & ESP32-S3 Cloud Server

![Mizo 3.0 Architecture](https://img.shields.io/badge/Architecture-FastAPI%20%7C%20ESP32--S3%20%7C%20Groq%20%7C%20Edge--TTS-6366f1?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-06b6d4?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-10b981?style=for-the-badge)

**Mizo 3.0 (Mikaza)** is a full-stack, real-time AI educational assistant and cloud server designed to power robotic tutors built on the **ESP32-S3 microcontroller**. It combines speech recognition, multi-provider LLM reasoning, long-term student memory, dynamic speech assessment, RAG document search, and low-latency audio streaming.

---

## 🌟 Key Features

- **Hardware-Ready Audio Pipeline**: Direct binary stream endpoint accepting audio from the **INMP441 I2S microphone** and returning **16kHz 16-bit Mono WAV** audio directly streamable to the **MAX98357A I2S DAC amplifier**.
- **Multi-Provider AI Brain**: Switch between **Groq** (`llama-3.3-70b`), **OpenAI** (`gpt-4o-mini`), **Qwen** (OpenRouter), and **Ollama** dynamically from the web dashboard.
- **Ultra-Fast Speech-to-Text (STT)**: Powered by **Groq Whisper** (`whisper-large-v3`) with local **Faster-Whisper** fallback.
- **Neural Text-to-Speech (TTS)**: Natural voice synthesis with **Edge-TTS** (US, UK, and Indian English neural voices).
- **Personalized Learning Modes**:
  - 🎙️ **Communication & Conversation Coach**: Live grammar analysis, vocabulary enrichment, and fluency metrics.
  - 📖 **Academic Subject Tutor**: Conceptual breakdown, analogies, and Socratic questioning.
  - 🎯 **Speech Trainer**: Pacing and confidence coaching.
- **RAG Knowledge Base**: Upload PDF textbooks or study notes for semantic similarity grounding.
- **Glassmorphic Admin Dashboard**: Web UI for monitoring learner progress, editing prompts, managing API keys, and testing with a **Virtual ESP32 Simulator**.

---

## 🏗️ System Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │               MIZO CLOUD SERVER              │
                       │                                              │
                       │  ┌────────────────────────────────────────┐  │
                       │  │   Admin Dashboard (Glassmorphic Web)   │  │
                       │  │   • AI behavior/rules & prompts        │  │
                       │  │   • Multi-provider API keys (Groq/OAI) │  │
                       │  │   • Student profiles & metrics         │  │
                       │  │   • Conversation history & feedback    │  │
                       │  │   • Knowledge base (PDFs/RAG)          │  │
                       │  │   • Live ESP32 Audio Simulator         │  │
                       │  └────────────────────────────────────────┘  │
                       │                                              │
                       │  ┌────────────────────────────────────────┐  │
                       │  │   FastAPI Backend (Async Core)         │  │
                       │  │   • STT: Groq Whisper / Faster-Whisper │  │
                       │  │   • RAG: Semantic doc search / chunker │  │
                       │  │   • LLM Router: Groq/OpenAI/Qwen/Ollama│  │
                       │  │   • Personalization & Memory Engine    │  │
                       │  │   • TTS: Edge-TTS (16kHz WAV)          │  │
                       │  │   • ESP32 Endpoint: Audio stream in/out│  │
                       │  └────────────────────────────────────────┘  │
                       │                                              │
                       │  ┌────────────────────────────────────────┐  │
                       │  │   SQLite Database + Vector Index       │  │
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

## 🚀 Getting Started

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/sameersam648/mizo.git
cd mizo
pip install -r requirements.txt
```

### 2. Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Add your API keys in `.env` (or configure them later via the Admin Dashboard):
```ini
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...
```

### 3. Start the Server

```bash
python run_server.py
```

- **Admin Dashboard**: [http://localhost:8000/](http://localhost:8000/)
- **Interactive API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ESP32 Audio Endpoint**: `http://localhost:8000/api/v1/esp32/audio`

---

## 🔌 ESP32-S3 Hardware Client

Full Arduino C++ firmware and wiring diagrams are in the [`firmware/esp32_s3_mizo/`](firmware/esp32_s3_mizo/) directory.

### Pinout Summary:
- **INMP441 (Microphone)**: `WS -> GPIO 5`, `SCK -> GPIO 6`, `SD -> GPIO 4`, `L/R -> GND`
- **MAX98357A (Speaker DAC)**: `BCLK -> GPIO 15`, `LRC -> GPIO 16`, `DIN -> GPIO 7`
- **Push-to-Talk Button**: `GPIO 0` (BOOT button)

---

## 🧪 Running Tests

```bash
python test_runner.py
```

---

## 📄 License
MIT License. Built for educational and AI robotics innovation.
