<div align="center">

# GenVQA — Generative Visual Question Answering

**A neuro-symbolic VQA system that detects objects with a neural model, retrieves structured facts from Wikidata, and generates grounded answers with Groq.**

</div>

---

## Architecture

```
Image ──► VQA Model ──► detected objects          [Neural]
               │
               ▼
          Wikidata API ──► structured facts        [Symbolic]
          (P31 category, P2101 melting point,
           P2054 density, P2777 flash point,
           P186 material, P366 use, ...)
               │
               ▼
          Groq LLM ──► natural-language answer     [Verbalize]
          (answers ONLY from Wikidata facts,
           no free reasoning)
```

| Layer         | What it does                                                         |
| ------------- | -------------------------------------------------------------------- |
| **Neural**    | Custom VQA model (Attention + GRU) detects objects in the image      |
| **Symbolic**  | Wikidata fetches live structured properties — no hardcoded knowledge |
| **Verbalize** | Groq (Llama 3.3 70B) generates an answer using _only_ those facts    |

---

## Features

- 🔍 **Visual Question Answering** — trained on VQAv2, fine-tuned on custom data
- 🧠 **Neuro-Symbolic Routing** — CLIP semantically classifies questions as _reasoning_ vs _visual_, routes accordingly
- 🌐 **Live Wikidata Facts** — queries physical properties, categories, materials, uses in real time
- 🤖 **Groq Verbalization** — Llama 3.3 70B answers from structured facts, not hallucination
- 💬 **Conversational Support** — multi-turn conversation manager with context tracking
- 📱 **Expo Mobile UI** — React Native app for iOS/Android/Web
- ♿ **Accessibility** — Groq generates spoken-friendly descriptions for every answer

---

## Quick Start

### 1 — Backend

```bash
# Clone and install
git clone https://github.com/DevaRajan8/Generative-vqa.git
cd Generative-vqa
pip install -r requirements_api.txt

# Set your Groq API key
cp .env.example .env
# Edit .env → GROQ_API_KEY=your_key_here

# Start API
python backend_api.py
# → http://localhost:8000
```

### 2 — Mobile UI

```bash
cd ui
npm install
npx expo start --clear
```

> Scan the QR code with Expo Go, or press `w` for browser.

---

## API

| Endpoint                | Method | Description                               |
| ----------------------- | ------ | ----------------------------------------- |
| `/api/answer`           | POST   | Answer a question about an uploaded image |
| `/api/health`           | GET    | Health check                              |
| `/api/conversation/new` | POST   | Start a new conversation session          |

**Example:**

```bash
curl -X POST http://localhost:8000/api/answer \
  -F "image=@photo.jpg" \
  -F "question=Can this melt?"
```

**Response:**

```json
{
  "answer": "ice",
  "model_used": "neuro-symbolic",
  "kg_enhancement": "Yes — ice can melt. [Wikidata P2101: melting point = 0.0 °C]",
  "knowledge_source": "VQA (neural) + Wikidata (symbolic) + Groq (verbalize)",
  "wikidata_entity": "Q86"
}
```

---

## Project Structure

```
├── backend_api.py                  # FastAPI server
├── ensemble_vqa_app.py             # VQA orchestrator (routing + inference)
├── semantic_neurosymbolic_vqa.py   # Wikidata KB + Groq verbalizer
├── groq_service.py                 # Groq accessibility descriptions
├── conversation_manager.py         # Multi-turn conversation tracking
├── model.py                        # VQA model definition
├── train.py                        # Training pipeline
├── ui/                             # Expo React Native app
│   └── src/screens/HomeScreen.js
└── .github/
    ├── workflows/                  # CI — backend lint + UI build
    └── ISSUE_TEMPLATE/
```

---

## Environment Variables

| Variable       | Required | Description                                             |
| -------------- | -------- | ------------------------------------------------------- |
| `GROQ_API_KEY` | ✅       | Groq API key — [get one free](https://console.groq.com) |
| `MODEL_PATH`   | optional | Path to VQA checkpoint (default: `vqa_checkpoint.pt`)   |
| `PORT`         | optional | API server port (default: `8000`)                       |

---

## Requirements

- Python 3.10+
- CUDA GPU recommended (CPU works but is slow)
- Node.js 20+ (for UI)
- Groq API key (free tier available)

---

## License

MIT © [DevaRajan8](https://github.com/DevaRajan8)
