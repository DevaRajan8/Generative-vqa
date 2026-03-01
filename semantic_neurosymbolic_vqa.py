"""
Semantic Neuro-Symbolic VQA

Architecture:
  NEURAL    -> VQA model detects objects from the image
  SYMBOLIC  -> Wikidata fetches structured facts about those objects
               (physical properties, categories, materials, uses, etc.)
  ANSWER    -> Groq generates a natural-language answer using ONLY
               the Wikidata facts — not from its own training knowledge.

Flow:
  1. VQA model detects objects (e.g. "soup")
  2. Wikidata lookup: fetch ALL relevant properties for "soup"
       P31  -> instance of    : food, liquid food, dish
       P186 -> material       : water, vegetable
       P2101-> melting point  : (none)
       P2054-> density        : (none)
       P2777-> flash point    : (none)
       P366 -> use            : consumption
       P18  -> physical state : liquid
  3. Groq receives: question + ALL Wikidata facts
     Groq is instructed to answer ONLY from those facts.
     Groq is the verbalizer, NOT the reasoner.
"""

import os
import torch
import clip
from transformers import GPT2Tokenizer
import requests
from typing import Dict, List, Optional
from functools import lru_cache
from groq import Groq


# ---------------------------------------------------------------------------
# Wikidata property definitions (what to fetch for every object)
# ---------------------------------------------------------------------------
WIKIDATA_PROPERTIES = {
    "P31":   "instance of (category)",
    "P279":  "subclass of",
    "P186":  "material / ingredient",
    "P366":  "use / purpose",
    "P2101": "melting point (K)",
    "P2054": "density (kg/m³)",
    "P2777": "flash point (K)",
    "P276":  "location",
    "P17":   "country of origin",
    "P921":  "main subject",
}


class WikidataKnowledgeBase:
    """
    Fetches comprehensive structured facts from Wikidata for any concept.
    This is the SYMBOLIC part of the neuro-symbolic pipeline.
    No hardcoded knowledge — everything comes from Wikidata at runtime.
    """
    SPARQL = "https://query.wikidata.org/sparql"
    API    = "https://www.wikidata.org/w/api.php"

    def __init__(self, session: requests.Session, timeout: int = 10):
        self.session = session
        self.timeout = timeout

    @lru_cache(maxsize=500)
    def get_entity_id(self, concept: str) -> Optional[str]:
        """Resolve any concept string to its Wikidata Q-ID."""
        try:
            r = self.session.get(self.API, params={
                "action": "wbsearchentities", "format": "json",
                "language": "en", "type": "item",
                "search": concept, "limit": 1,
            }, timeout=self.timeout)
            r.raise_for_status()
            hits = r.json().get("search", [])
            return hits[0]["id"] if hits else None
        except Exception:
            return None

    def get_property_values(self, entity_id: str, prop: str,
                            limit: int = 5) -> List[str]:
        """Fetch the English labels of all values of a property."""
        query = f"""
        SELECT ?valueLabel WHERE {{
          wd:{entity_id} wdt:{prop} ?value.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }} LIMIT {limit}
        """
        try:
            r = self.session.get(self.SPARQL,
                                  params={"query": query, "format": "json"},
                                  timeout=self.timeout)
            r.raise_for_status()
            rows = r.json().get("results", {}).get("bindings", [])
            return [row["valueLabel"]["value"] for row in rows
                    if "valueLabel" in row]
        except Exception:
            return []

    def get_numeric_property(self, entity_id: str, prop: str) -> Optional[float]:
        """Fetch the first numeric value of a property (e.g. melting point in K)."""
        query = f"""
        SELECT ?value WHERE {{
          wd:{entity_id} wdt:{prop} ?value.
        }} LIMIT 1
        """
        try:
            r = self.session.get(self.SPARQL,
                                  params={"query": query, "format": "json"},
                                  timeout=self.timeout)
            r.raise_for_status()
            rows = r.json().get("results", {}).get("bindings", [])
            if rows:
                return float(rows[0]["value"]["value"])
        except Exception:
            pass
        return None

    def fetch_all_facts(self, concept: str) -> Optional[Dict]:
        """
        Fetch ALL Wikidata facts for a concept.
        Returns a structured dict of facts, or None if concept not found.
        """
        entity_id = self.get_entity_id(concept)
        if not entity_id:
            return None

        facts = {"entity_id": entity_id, "concept": concept}

        # Categorical facts (label-based)
        for prop, desc in [
            ("P31",  "categories"),
            ("P279", "parent_classes"),
            ("P186", "materials"),
            ("P366", "uses"),
            ("P276", "locations"),
            ("P17",  "countries"),
        ]:
            values = self.get_property_values(entity_id, prop, limit=5)
            if values:
                facts[desc] = values

        # Numeric / physical properties
        melting_k = self.get_numeric_property(entity_id, "P2101")
        if melting_k is not None:
            facts["melting_point_celsius"] = round(melting_k - 273.15, 1)
            facts["melting_point_kelvin"]  = melting_k

        density = self.get_numeric_property(entity_id, "P2054")
        if density is not None:
            facts["density_kg_m3"] = density

        flash_k = self.get_numeric_property(entity_id, "P2777")
        if flash_k is not None:
            facts["flash_point_celsius"] = round(flash_k - 273.15, 1)

        return facts if len(facts) > 2 else None   # must have more than just entity_id + concept

    def format_facts_for_prompt(self, facts: Dict) -> str:
        """
        Format facts into a human-readable block for the Groq prompt.
        This is what Groq will reason over.
        """
        concept = facts.get("concept", "object")
        lines   = [f"Wikidata facts about '{concept}':"]

        if "categories" in facts:
            lines.append(f"  - Category (P31): {', '.join(facts['categories'])}")
        if "parent_classes" in facts:
            lines.append(f"  - Subclass of (P279): {', '.join(facts['parent_classes'])}")
        if "materials" in facts:
            lines.append(f"  - Made of (P186): {', '.join(facts['materials'])}")
        if "uses" in facts:
            lines.append(f"  - Used for (P366): {', '.join(facts['uses'])}")
        if "locations" in facts:
            lines.append(f"  - Found at (P276): {', '.join(facts['locations'])}")
        if "countries" in facts:
            lines.append(f"  - Origin (P17): {', '.join(facts['countries'])}")
        if "melting_point_celsius" in facts:
            lines.append(f"  - Melting point (P2101): {facts['melting_point_celsius']} °C")
        if "density_kg_m3" in facts:
            d = facts["density_kg_m3"]
            floats = "floats on water" if d < 1000 else "sinks in water"
            lines.append(f"  - Density (P2054): {d} kg/m³ ({floats})")
        if "flash_point_celsius" in facts:
            lines.append(f"  - Flash point (P2777): {facts['flash_point_celsius']} °C (flammable)")

        if len(lines) == 1:
            return f"No Wikidata facts found for '{concept}'."
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Groq answer generator — verbalizes Wikidata facts, does not reason freely
# ---------------------------------------------------------------------------

class WikidataGroqAnswerer:
    """
    Uses Groq to generate a natural-language answer to a question
    using ONLY the Wikidata facts provided. Groq is the verbalizer,
    not the reasoner — it cannot use knowledge beyond what's provided.
    """

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "llama-3.3-70b-versatile"):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY not set")
        self.client = Groq(api_key=key)
        self.model  = model

    def answer(self, question: str, facts_text: str,
               object_name: str) -> str:
        """
        Generate an answer to the question using only the provided Wikidata facts.
        """
        system_prompt = (
            "You are a neuro-symbolic reasoning assistant. "
            "Your job is to answer the question using commonsense inference from the Wikidata facts given. "
            "Rules:\n"
            "1. If the object is an animal, organism, or mammal → it is biological: it cannot melt, dissolve, "
            "   or catch fire like a material. It CAN walk/run/swim/eat depending on its class.\n"
            "2. If the object is food, drink, or plant → it is edible/organic. It can decay but not melt.\n"
            "3. If the object is metal, plastic, wax, or glass → reason about physical properties normally.\n"
            "4. If the object is tableware or a container (bowl, cup, plate) → it holds food/drink. "
            "   Its melting depends on material (ceramic/glass won't melt at normal temperatures).\n"
            "5. NEVER say 'cannot be determined'. Always give a concrete commonsense answer "
            "   inferred from the category or subclass of the object.\n"
            "6. Keep your answer to 1-2 sentences. Be direct and conversational."
        )

        user_prompt = (
            f"{facts_text}\n\n"
            f"Question about '{object_name}': {question}\n\n"
            f"Using the Wikidata facts and commonsense reasoning about the object's category, answer:"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.1,   # low temperature = more factual, less creative
                max_tokens=180,
                top_p=0.9,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Groq unavailable: {e}"


# ===========================================================================

class SemanticNeurosymbolicVQA:
    """
    TRUE Neuro-Symbolic VQA:
      NEURAL    -> VQA model detects objects (what is in the image?)
      SYMBOLIC  -> WikidataKnowledgeBase fetches structured facts
      VERBALIZE -> WikidataGroqAnswerer answers using only those facts
    """
    WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
    WIKIDATA_API    = "https://www.wikidata.org/w/api.php"

    def __init__(self, device="cuda", timeout=15):
        self.device  = device
        self.timeout = timeout

        print("  -> Loading CLIP (question routing — neural)...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()

        print("  -> Loading GPT-2 tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SemanticVQA/1.0 (Educational)"})

        # Symbolic knowledge base
        self.knowledge_base = WikidataKnowledgeBase(self.session, timeout)
        print("  -> Wikidata knowledge base ready")

        # Groq verbalizer
        self.groq_answerer  = None
        self.groq_enabled   = False
        try:
            self.groq_answerer = WikidataGroqAnswerer()
            self.groq_enabled  = True
            print("  -> Groq verbalizer ready (answers from Wikidata facts only)")
        except Exception as e:
            print(f"  -> Groq unavailable ({e}), will return raw facts")

        # Legacy flag (backward compat)
        self.llm_enabled = self.groq_enabled

        print("OK Neuro-Symbolic VQA ready")
        print("   [Neural: VQA+CLIP | Symbolic: Wikidata | Verbalize: Groq]")

    # ------------------------------------------------------------------
    # CLIP zero-shot image → object detection
    # ------------------------------------------------------------------

    # Vocabulary of ~80 common concrete nouns that have Wikidata entries.
    # Kept deliberately concrete (no adjectives, no verbs) so Wikidata lookup works.
    CLIP_OBJECT_VOCAB = [
        # People
        "person", "man", "woman", "child", "baby",
        # Animals
        "dog", "cat", "bird", "horse", "cow", "elephant", "lion", "tiger",
        "bear", "zebra", "giraffe", "sheep", "pig", "rabbit", "fish",
        # Vehicles
        "car", "truck", "bus", "bicycle", "motorcycle", "airplane", "boat",
        "train", "helicopter",
        # Furniture / indoor
        "chair", "table", "sofa", "bed", "desk", "lamp", "shelf", "door",
        # Electronics
        "laptop", "phone", "television", "camera", "keyboard", "monitor",
        # Food / drink
        "apple", "banana", "orange", "pizza", "cake", "sandwich", "coffee",
        "bread", "bottle", "cup", "bowl",
        # Nature / outdoor
        "tree", "flower", "grass", "mountain", "river", "sky", "cloud",
        "rock", "leaf",
        # Materials / objects
        "book", "paper", "bag", "box", "ball", "knife", "fork",
        "glass", "plastic", "metal", "wood", "stone", "ice", "fire",
        # Buildings
        "house", "building", "bridge", "road",
    ]

    def detect_objects_with_clip(self, image_path: str, top_k: int = 3) -> List[str]:
        """
        Use CLIP zero-shot classification to detect the top-k objects in an image.

        Instead of asking the VQA model (which can hallucinate), we encode the image
        with CLIP's vision encoder and score it against every label in CLIP_OBJECT_VOCAB
        using cosine similarity.  The highest-scoring labels are returned.

        Args:
            image_path: Absolute path to the image file.
            top_k:      Number of top objects to return (default 3).

        Returns:
            List of object name strings, e.g. ["person", "chair"]
        """
        try:
            from PIL import Image as PILImage
            img = PILImage.open(image_path).convert("RGB")
            img_tensor = self.clip_preprocess(img).unsqueeze(0).to(self.device)

            # Wrap each label in a natural prompt for better CLIP alignment
            prompts = [f"a photo of a {label}" for label in self.CLIP_OBJECT_VOCAB]
            text_tokens = clip.tokenize(prompts).to(self.device)

            with torch.no_grad():
                img_features  = self.clip_model.encode_image(img_tensor)
                text_features = self.clip_model.encode_text(text_tokens)

                # Normalise → cosine similarity
                img_features  = img_features  / img_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                similarities  = (img_features @ text_features.T).squeeze(0)  # (N_labels,)

            # Pick top-k
            top_indices = similarities.topk(top_k).indices.cpu().tolist()
            detected    = [self.CLIP_OBJECT_VOCAB[i] for i in top_indices]
            print(f"      [CLIP] Top-{top_k} objects detected: {detected}")
            return detected

        except Exception as e:
            print(f"      [CLIP] Object detection failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Routing — CLIP decides if question needs neuro-symbolic reasoning
    # ------------------------------------------------------------------

    def should_use_neurosymbolic(self, image_features, question: str,
                                  vqa_confidence: float = 0.0,
                                  image_path: str = None) -> bool:
        """
        Routing via CLIP text-to-anchor similarity — zero pattern matching, image-independent.

        The question is compared against two descriptive anchor sentences:
          VISUAL anchor   → "A question asking what can be seen, observed, or counted in the image"
          KNOWLEDGE anchor→ "A question asking about facts, properties, or behaviour of objects"

        If the question is closer to the VISUAL anchor   → neural VQA 👁️
        If the question is closer to the KNOWLEDGE anchor → neuro-symbolic 🧠

        This works for ANY image because the question text alone carries the intent.
        Generic visual questions ("what is there?", "what animal?") always score closer
        to the visual anchor regardless of image content.
        """
        # Visual questions — about what IS in the image
        VISUAL_ANCHOR = (
            "What is this? What is in the image? What animal is shown? "
            "What color is it? How many are there? What object is visible? "
            "What is the person doing? What is in the background?"
        )
        # Knowledge questions — about properties/capabilities of the subject
        KNOWLEDGE_ANCHOR = (
            "Can this melt? Can this walk? Can this swim? Can this fly? "
            "Is this edible? Can this be eaten? Is this safe? Can this burn? "
            "What is this used for? What is this made of? Is this alive? "
            "Can this float? Does this conduct electricity? Is this poisonous?"
        )

        try:
            q_tok = clip.tokenize([question]).to(self.device)
            a_tok = clip.tokenize([VISUAL_ANCHOR, KNOWLEDGE_ANCHOR]).to(self.device)

            with torch.no_grad():
                q_feat = self.clip_model.encode_text(q_tok)
                q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)

                a_feat = self.clip_model.encode_text(a_tok)
                a_feat = a_feat / a_feat.norm(dim=-1, keepdim=True)

                # Raw cosine similarities
                sims = (q_feat @ a_feat.T).squeeze()   # [visual_sim, knowledge_sim]

                # Temperature scaling (×10) + softmax to amplify the gap
                probs = torch.softmax(sims * 10, dim=0)

            visual_prob    = probs[0].item()
            knowledge_prob = probs[1].item()
            use_ns         = knowledge_prob > visual_prob

            route_reason = "knowledge/capability question" if use_ns else "visual/perceptual question"
            print(f"      [Routing] visual={visual_prob:.3f}  knowledge={knowledge_prob:.3f} "
                  f"→ {route_reason} → {'neuro-symbolic 🧠' if use_ns else 'neural VQA 👁️'}")
            return use_ns

        except Exception as e:
            print(f"      [Routing] CLIP routing failed ({e}) → defaulting to neuro-symbolic")
            return True

    # ------------------------------------------------------------------
    # CLIP question intent (kept for backward compat / analytics)
    # ------------------------------------------------------------------

    def _analyze_question_semantics(self, question: str) -> Dict:
        intent_templates = {
            "capability":  "Can this object do something?",
            "property":    "What properties does this have?",
            "purpose":     "What is this used for?",
            "composition": "What is this made of?",
            "location":    "Where is this found?",
            "safety":      "Is this safe or dangerous?",
            "edibility":   "Can this be eaten?",
            "state":       "What state or condition is this?",
        }
        try:
            q_tok = clip.tokenize([question]).to(self.device)
            i_tok = clip.tokenize(list(intent_templates.values())).to(self.device)
            with torch.no_grad():
                q_feat = self.clip_model.encode_text(q_tok)
                q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)
                i_feat = self.clip_model.encode_text(i_tok)
                i_feat = i_feat / i_feat.norm(dim=-1, keepdim=True)
                sims   = (q_feat @ i_feat.T).squeeze()
                probs  = torch.softmax(sims * 10, dim=0)
            return {k: probs[i].item() for i, k in enumerate(intent_templates)}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Backward-compat Wikidata helpers (used by ensemble_vqa_app.py)
    # ------------------------------------------------------------------

    @lru_cache(maxsize=200)
    def _get_wikidata_id(self, concept: str) -> Optional[str]:
        return self.knowledge_base.get_entity_id(concept)

    def _get_wikidata_knowledge(self, concept: str, intent: Dict) -> Optional[Dict]:
        return self.knowledge_base.fetch_all_facts(concept)

    def _query_wikidata_property(self, entity_id: str, prop: str) -> List[str]:
        return self.knowledge_base.get_property_values(entity_id, prop)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def answer_with_clip_features(
        self,
        image_features,
        question: str,
        image_path: str = None,
        detected_objects: List[str] = None,
    ) -> Optional[Dict]:
        """
        Full neuro-symbolic pipeline:
          Step 1 (Neural  — done externally): VQA detects objects
          Step 2 (Symbolic): Wikidata fetches ALL structured facts for the object
          Step 3 (Verbalize): Groq answers the question using ONLY those facts
        """
        if not detected_objects:
            print("      No objects supplied — skipping neuro-symbolic")
            return None

        question_intent = self._analyze_question_semantics(question)

        for obj in detected_objects[:5]:
            print(f"      [Symbolic] Fetching all Wikidata facts for '{obj}'...")
            facts = self.knowledge_base.fetch_all_facts(obj)

            if not facts:
                print(f"      [Symbolic] No Wikidata entity for '{obj}', skipping")
                continue

            entity_id  = facts["entity_id"]
            facts_text = self.knowledge_base.format_facts_for_prompt(facts)
            print(f"      [Symbolic] {entity_id} | {len(facts)-2} fact groups fetched")
            print(f"      [Symbolic] Facts:\n{facts_text}")

            # Step 3: Groq answers using ONLY the Wikidata facts
            if self.groq_enabled and self.groq_answerer:
                print(f"      [Groq] Generating answer from Wikidata facts...")
                answer_text = self.groq_answerer.answer(question, facts_text, obj)
            else:
                # Fallback: return the raw facts summary
                answer_text = facts_text

            return {
                "kg_enhancement":   answer_text,
                "reasoning_type":   "neuro-symbolic",
                "knowledge_source": "VQA (neural) + Wikidata (symbolic) + Groq (verbalize)",
                "objects_detected": detected_objects,
                "question_intent":  question_intent,
                "wikidata_entity":  entity_id,
                "wikidata_facts":   facts,
            }

        return None

    # ------------------------------------------------------------------
    # Backward-compat aliases
    # ------------------------------------------------------------------

    def _generate_semantic_answer(self, objects, question, intent, knowledge):
        """Legacy alias — delegates to the new pipeline."""
        result = self.answer_with_clip_features(
            image_features=None,
            question=question,
            detected_objects=objects,
        )
        return result["kg_enhancement"] if result else None

    def _detect_objects_with_clip(self, image_features, image_path=None):
        """DEPRECATED — object detection done by VQA model."""
        print("      _detect_objects_with_clip is deprecated.")
        return []