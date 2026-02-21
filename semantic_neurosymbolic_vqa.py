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
            "You answer questions STRICTLY using the Wikidata facts provided below. "
            "Do NOT use any outside knowledge or assumptions. "
            "If the facts do not contain enough information to answer, say: "
            "'The Wikidata knowledge base does not have enough facts to answer this question.' "
            "Keep your answer to 1-2 sentences. Be direct and factual."
        )

        user_prompt = (
            f"{facts_text}\n\n"
            f"Question: {question}\n\n"
            f"Answer using ONLY the Wikidata facts above (no outside knowledge):"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.1,   # low temperature = more factual, less creative
                max_tokens=120,
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
    # Routing — CLIP decides if question needs neuro-symbolic reasoning
    # ------------------------------------------------------------------

    def should_use_neurosymbolic(self, image_features, question: str,
                                  vqa_confidence: float = 0.0) -> bool:
        """
        CLIP compares the question against two natural-language descriptions:
          - reasoning/knowledge questions  (route to neuro-symbolic)
          - visual/perceptual questions    (stay neural)
        No hardcoded keyword lists.
        """
        reasoning_desc = (
            "A question about what an object is made of, what it can do, "
            "its physical properties like melting or floating, whether it is "
            "edible or safe, what it is used for, or how it is classified."
        )
        visual_desc = (
            "A question about what is visible in the image, the color, "
            "number, location, or spatial position of objects."
        )
        try:
            q_tok = clip.tokenize([question]).to(self.device)
            d_tok = clip.tokenize([reasoning_desc, visual_desc]).to(self.device)
            with torch.no_grad():
                q_feat = self.clip_model.encode_text(q_tok)
                q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)
                d_feat = self.clip_model.encode_text(d_tok)
                d_feat = d_feat / d_feat.norm(dim=-1, keepdim=True)
                sims   = (q_feat @ d_feat.T).squeeze()
            return bool(sims[0].item() > sims[1].item())
        except Exception:
            return False

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