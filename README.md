<div align="center">

# 🧠 GenVQA — Generative Visual Question Answering

**A hybrid neuro-symbolic VQA system that intelligently routes between pure neural networks and knowledge-grounded reasoning**

</div>

---

## Overview

GenVQA is an advanced Visual Question Answering system that combines the best of both worlds:

- **Neural networks** for perception-based visual questions
- **Symbolic reasoning** for knowledge-intensive reasoning questions

The system automatically classifies incoming questions and routes them to the optimal processing pipeline, ensuring accurate and grounded answers.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        CLIENT                           │
│         Expo React Native App (iOS/Android/Web)                  │
│         • Image upload via camera/gallery                        │
│         • Question input with suggested prompts                  │
│         • Multi-turn conversational interface                    │
│         • Google OAuth authentication                            │
└───────────────────────────┬──────────────────────────────────────┘
                            │ HTTP POST /api/answer
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    BACKEND API LAYER                          │
│                    FastAPI (backend_api.py)                      │
│         • Request handling & validation                          │
│         • Session management & authentication                    │
│         • Multi-turn conversation tracking                       │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                   INTELLIGENT ROUTING LAYER                   │
│                  (ensemble_vqa_app.py)                           │
│                                                                  │
│   CLIP Semantic Classifier:                                     │
│   Encodes question → Compares similarity:                       │
│   "This is a reasoning question about facts"                    │
│              vs                                                  │
│   "This is a visual perception question"                        │
│                                                                  │
│           Similarity > threshold?
                                │
│                     ├─────────┬────────┐                        │
│                     │         │        │                        │
│               REASONING    VISUAL   SPATIAL                      │
│                     │         │        │                        │
└─────────────────────┼─────────┼────────┼─────────────────────────┘
                      │         │        │
        ┌─────────────┘         │        └─────────────┐
        ▼                       ▼                      ▼
┌──────────────────┐   ┌───────────────────┐   ┌─────────────────┐
│ NEURO-SYMBOLIC   │   │  NEURAL VQA PATH  │   │ SPATIAL ADAPTER │
│    PIPELINE      │   │                   │   │      PATH       │
│                  │   │  CLIP + GRU +     │   │                 │
│ ① VQA Model      │   │  Attention        │   │  Enhanced with  │
│    Detects       │   │                   │   │  spatial        │
│    Objects       │   │  Direct answer    │   │  self-attention │
│    (e.g. "soup") │   │  prediction from  │   │  for left/right │
│                  │   │  image features   │   │  above/below    │
│ ② Wikidata API   │   │                   │   │  questions      │
│    Fetches Facts │   │  Outputs:         │   │                 │
│    P31: category │   │  "red"            │   │  Outputs:       │
│    P186: material│   └───────┬───────────┘   │  "on the left"  │
│    P2101: melting│           │               └────────┬────────┘
│    P366: use     │           │                        │
│    P2054: density│           │                        │
│                  │           │                        │
│ ③ Groq LLM       │           │                        │
│    Verbalizes    │           │                        │
│    from facts    │           │                        │
│    (instead
      of free      │           │                        │
│     reasoning)   │           │                        │
│                  │           │                        │
│ Outputs:         │           │                        │
│ "Soup is made of │           │                        │
│  water and       │           │                        │
│  vegetables,     │           │                        │
│  used for eating"│           │                        │
└────────┬─────────┘           │                        │
         │                     │                        │
         └──────────┬──────────┴────────────────────────┘
                    ▼
         ┌──────────────────────┐
         │  GROQ ACCESSIBILITY  │
         │       SERVICE        │
         │                      │
         │  Generates 2-sentence│
         │  screen-reader       │
         │  friendly description│
         │  for every answer    │
         └──────────┬───────────┘
                    │
                    ▼
              JSON Response
         {
           "answer": "...",
           "model_used": "neuro_symbolic|base|spatial",
           "confidence": 0.85,
           "kg_enhancement": true/false,
           "wikidata_entity": "Q123456",
           "description": "...",
           "session_id": "..."
         }
```

---

## Neural vs Neuro-Symbolic: Deep Dive

### Neural Pathway

**When Used**: Perceptual questions about what's directly visible

- _"What color is the car?"_
- _"How many people are in the image?"_
- _"Is the dog sitting or standing?"_

**Architecture**:

```
Image Input
    │
    ▼
┌─────────────────────────────┐
│    CLIP Vision Encoder      │
│    (ViT-B/16)               │
│    • Pre-trained on 400M    │
│      image-text pairs       │
│    • 512-dim embeddings     │
└──────────┬──────────────────┘
           │
           ▼
      [512-dim vector] ────────────┐
                                   │
Question Input                     │
    │                              │
    ▼                              │
┌─────────────────────────────┐   │
│   GPT-2 Text Encoder        │   │
│   (distilgpt2)              │   │
│   • Contextual embeddings   │   │
│   • 768-dim output          │   │
└──────────┬──────────────────┘   │
           │                       │
           ▼                       │
      [768-dim vector]             │
           │                       │
           ▼                       │
    ┌──────────────┐               │
    │ Linear Proj  │               │
    │ 768 → 512    │               │
    └──────┬───────┘               │
           │                       │
           └───────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  Multimodal Fusion   │
            │  • Gated combination │
            │  • 3-layer MLP       │
            │  • ReLU + Dropout    │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  GRU Decoder with    │
            │  Attention Mechanism │
            │                      │
            │  • Hidden: 512-dim   │
            │  • 2 layers          │
            │  • Seq2seq decoding  │
            │  • Attention over    │
            │    fused features    │
            └──────────┬───────────┘
                       │
                       ▼
                 Answer Tokens
                 "red car"
```

**Key Components**:

- **CLIP**: Zero-shot image understanding, robust to domain shift
- **GPT-2**: Contextual question encoding
- **Attention**: Decoder focuses on relevant image regions per word
- **GRU**: Sequential answer generation with memory

**Training**:

- Dataset: VQA v2 (curated, balanced subset)
- Loss: Cross-entropy over answer vocabulary
- Fine-tuning: Last 2 CLIP layers + full decoder
- Accuracy: ~39% on general VQA, ~28% on spatial questions

---

### Neuro-Symbolic Pathway (Knowledge-Grounded Reasoning)

**When Used**: Questions requiring external knowledge or reasoning

- _"Can soup melt?"_
- _"What is ice cream made of?"_
- _"Does this float in water?"_

**Architecture**:

```
Step 1: NEURAL DETECTION
─────────────────────────
Image + Question
    │
    ▼
┌──────────────────────┐
│   VQA Model          │
│   (same as above)    │
│                      │
│   Predicts: "soup"   │
└──────────┬───────────┘
           │
           ▼
    Detected Object
       "soup"

Step 2: SYMBOLIC FACT RETRIEVAL
────────────────────────────────
    "soup"
       │
       ▼
┌──────────────────────────────────┐
│    Wikidata SPARQL Queries       │
│                                  │
│ ① Entity Resolution:             │
│    "soup" → Q41415 (Wikidata ID) │
│                                  │
│ ② Fetch ALL Relevant Properties: │
│                                  │
│    P31  (instance of):           │
│         → "food"                 │
│         → "liquid food"          │
│         → "dish"                 │
│                                  │
│    P186 (made of):               │
│         → "water"                │
│         → "vegetables"           │
│         → "broth"                │
│                                  │
│    P366 (used for):              │
│         → "consumption"          │
│         → "nutrition"            │
│                                  │
│    P2101 (melting point):        │
│         → (not found)            │
│                                  │
│    P2054 (density):              │
│         → ~1000 kg/m³            │
│         → (floats/sinks calc)    │
│                                  │
│    P2777 (flash point):          │
│         → (not found)            │
└──────────────┬───────────────────┘
               │
               ▼
    Structured Knowledge Graph
    {
      "entity": "soup (Q41415)",
      "categories": ["food", "liquid"],
      "materials": ["water", "vegetables"],
      "uses": ["consumption"],
      "density": 1000,
      "melting_point": null
    }

Step 3: LLM VERBALIZATION (NOT REASONING!)
───────────────────────────────────────────
    Knowledge Graph
         │
         ▼
┌────────────────────────────────────┐
│         Groq API                   │
│     (Llama 3.3 70B)                │
│                                    │
│  System Prompt:                    │
│  "You are a fact verbalizer.      │
│   Answer ONLY from provided        │
│   Wikidata facts. Do NOT use       │
│   your training knowledge.         │
│   If facts don't contain the       │
│   answer, say 'unknown from        │
│   available data'."                │
│                                    │
│  User Input:                       │
│  Question: "Can soup melt?"        │
│  Facts: {structured data above}    │
└────────────┬───────────────────────┘
             │
             ▼
    Natural Language Answer
    "According to Wikidata, soup is
     a liquid food made of water and
     vegetables. Since it's already
     liquid, it doesn't have a melting
     point like solids do. It can
     freeze, but not melt."
```

**Critical Design Principle**:

> Groq is a **verbalizer**, NOT a reasoner. All reasoning happens in the symbolic layer (Wikidata facts). Groq only translates structured facts into natural language.

**Why This Matters**:

- **Without facts**: Groq hallucinates from training data
- **With facts**: Groq grounds answers in real-time data
- **Result**: Factual accuracy, no made-up information

**Knowledge Base Properties Fetched**:
| Property | Wikidata Code | Example Value |
|----------|---------------|---------------|
| Category | P31 | "food", "tool", "animal" |
| Material | P186 | "metal", "wood", "plastic" |
| Melting Point | P2101 | 273.15 K (0°C) |
| Density | P2054 | 917 kg/m³ (floats/sinks) |
| Use | P366 | "eating", "transportation" |
| Flash Point | P2777 | 310 K (flammable) |
| Location | P276 | "ocean", "forest" |

---

### Spatial Reasoning Pathway

**When Used**: Questions about relative positions

- _"What is to the left of the car?"_
- _"Is the cat above or below the table?"_

**Architecture Enhancement**:

```
Base VQA Model
    │
    ▼
┌──────────────────────────────┐
│  Spatial Self-Attention      │
│  • Multi-head attention (8)  │
│  • Learns spatial relations  │
│  • Position-aware weighting  │
└──────────┬───────────────────┘
           │
           ▼
    Spatial-aware answer
    "on the left side"
```

**Keyword Triggering**:

- Detects: `left`, `right`, `above`, `below`, `top`, `bottom`, `next to`, `behind`, `between`, etc.
- Routes to spatial adapter model
- Enhanced accuracy on positional questions

---

## Intelligent Routing System

**CLIP-Based Semantic Routing**:

```python
# Encode question with CLIP
question_embedding = clip.encode_text(question)

# Compare against two templates
reasoning_prompt = "This is a reasoning question about facts and knowledge"
visual_prompt = "This is a visual perception question about what you see"

reasoning_similarity = cosine_similarity(question_embedding,
                                         clip.encode_text(reasoning_prompt))
visual_similarity = cosine_similarity(question_embedding,
                                      clip.encode_text(visual_prompt))

# Route decision
if reasoning_similarity > visual_similarity + THRESHOLD:
    route_to_neuro_symbolic()
elif contains_spatial_keywords(question):
    route_to_spatial_adapter()
else:
    route_to_base_neural()
```

**Routing Logic**:

1. **Neuro-Symbolic** if CLIP classifies as reasoning (>0.6 similarity)
2. **Spatial** if contains spatial keywords (`left`, `right`, `above`, etc.)
3. **Base Neural** for all other visual perception questions

---

## Multi-Turn Conversation Support

**Conversation Manager Features**:

- Session tracking with UUID
- Context retention across turns
- Pronoun resolution (`it`, `this`, `that` → previous object)
- Automatic session expiry (30 min timeout)

**Example Conversation**:

```
Turn 1:
User: "What is this?"
VQA: "A red car"
Objects: ["car"]

Turn 2:
User: "Can it float?"              # "it" = "car"
System: Resolves "it" → "car"
VQA: [Neuro-Symbolic] "According to Wikidata, cars are made
      of metal and plastic with density around 800-1000 kg/m³,
      which is close to water. Most cars would sink."

Turn 3:
User: "What color is it again?"    # Still referring to car
VQA: [Neural] "red"                # From Turn 1 context
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA GPU (recommended, 4GB+ VRAM)
- Node.js 16+ (for mobile UI)
- Groq API key ([get one free](https://console.groq.com))

### Backend Setup

```bash
# 1. Clone repository
git clone https://github.com/YourUsername/vqa_coes.git
cd vqa_coes

# 2. Install dependencies
pip install -r requirements_api.txt

# 3. Set environment variables
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# 4. Download model checkpoints (if not included)
# Ensure these files exist in project root:
#   - vqa_checkpoint.pt (base model)
#   - vqa_spatial_checkpoint.pt (spatial model)

# 5. Start API server
python backend_api.py

# Server will start at http://localhost:8000
```

### Mobile UI Setup

```bash
# 1. Navigate to UI folder
cd ui

# 2. Install dependencies
npm install

# 3. Configure API endpoint
# Edit ui/src/config/api.js
# Change: export const API_BASE_URL = 'http://YOUR_LOCAL_IP:8000';

# 4. Start Expo
npx expo start --clear

# Scan QR code with Expo Go app, or press 'w' for web
```

---

## 🔧 API Reference

### POST `/api/answer`

Answer a visual question with optional conversation context.

**Request**:

```bash
curl -X POST http://localhost:8000/api/answer \
  -F "image=@photo.jpg" \
  -F "question=Can this float in water?" \
  -F "session_id=optional-uuid-here"
```

**Response**:

```json
{
  "answer": "According to Wikidata, this object has a density of 917 kg/m³, which is less than water (1000 kg/m³), so it would float.",
  "model_used": "neuro_symbolic",
  "confidence": 0.87,
  "kg_enhancement": true,
  "wikidata_entity": "Q41576",
  "description": "The object appears to be made of ice. Based on its physical properties from scientific data, it would float on water due to lower density.",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "conversation_turn": 2
}


## 📄 License

MIT License - see LICENSE file for details

---
```
