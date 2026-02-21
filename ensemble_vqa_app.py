"""
Production Ensemble VQA Application
Combines base model (general VQA) and spatial adapter (spatial reasoning)
for optimal performance on all question types.
NEW: Neuro-Symbolic VQA with Knowledge Graph integration
NEW: Multi-turn Conversational VQA with context management
"""
import os
import torch
from PIL import Image
from transformers import GPT2Tokenizer
from models.model import VQAModel
from model_spatial import VQAModelWithSpatialAdapter
from experiments.train import Vocab
from knowledge_graph_service import KnowledgeGraphService
from typing import Optional
import time
class ProductionEnsembleVQA:

    SPATIAL_KEYWORDS = [
        'right', 'left', 'above', 'below', 'top', 'bottom',
        'up', 'down', 'upward', 'downward',
        'front', 'behind', 'back', 'next to', 'beside', 'near', 'between',
        'in front', 'in back', 'across from', 'opposite', 'adjacent',
        'closest', 'farthest', 'nearest', 'furthest', 'closer', 'farther',
        'where is', 'where are', 'which side', 'what side', 'what direction',
        'on the left', 'on the right', 'at the top', 'at the bottom',
        'to the left', 'to the right', 'in the middle', 'in the center',
        'under', 'over', 'underneath', 'on top of', 'inside', 'outside'
    ]
    def __init__(self, base_checkpoint, spatial_checkpoint, device='cuda'):
      
        self.device = device if torch.cuda.is_available() else 'cpu'
        print("="*80)
        print("🚀 INITIALIZING ENSEMBLE VQA SYSTEM")
        print("="*80)
        print(f"\n⚙️  Device: {self.device}")
        print("\n📥 Loading models...")
        start_time = time.time()
        print("  [1/2] Loading base model (general VQA)...")
        self.base_model, self.vocab, self.tokenizer = self._load_base_model(base_checkpoint)
        print("      ✓ Base model loaded")
        print("  [2/2] Loading spatial model (spatial reasoning)...")
        self.spatial_model, _, _ = self._load_spatial_model(spatial_checkpoint)
        print("      ✓ Spatial model loaded")
        load_time = time.time() - start_time
        print("  [3/3] Initializing Semantic Neuro-Symbolic VQA...")
        try:
            from semantic_neurosymbolic_vqa import SemanticNeurosymbolicVQA
            self.kg_service = SemanticNeurosymbolicVQA(device=self.device)
            print("      ✓ Semantic Neuro-Symbolic VQA ready (CLIP + Wikidata, no pattern matching)")
            self.kg_enabled = True
        except Exception as e:
            print(f"      ⚠️  Semantic Neuro-Symbolic VQA unavailable: {e}")
            print("      → Falling back to neural-only mode")
            self.kg_service = None
            self.kg_enabled = False
        print(f"\n✅ Ensemble ready! (loaded in {load_time:.1f}s)")
        print(f"📊 Memory: ~2x single model (~4GB GPU)")
        print(f"🎯 Routing: Automatic based on question type")
        print(f"🧠 Neuro-Symbolic: {'Enabled' if self.kg_enabled else 'Disabled (neural-only)'}")
        print(f"💬 Conversation: Initializing multi-turn support...")
        try:
            from conversation_manager import ConversationManager
            self.conversation_manager = ConversationManager(session_timeout_minutes=30)
            self.conversation_enabled = True
            print(f"      ✓ Conversational VQA ready (multi-turn with context)")
        except Exception as e:
            print(f"      ⚠️  Conversation manager unavailable: {e}")
            print(f"      → Single-shot Q&A only")
            self.conversation_manager = None
            self.conversation_enabled = False
        print("="*80)
    def _load_base_model(self, checkpoint_path):
        """Load base VQA model."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        vocab = Vocab()
        vocab.vocab = checkpoint['vocab']
        vocab.vocab_size = len(checkpoint['vocab'])
        vocab.word2idx = checkpoint['word2idx']
        vocab.idx2word = checkpoint['idx2word']
        vocab.pad_token_id = checkpoint['pad_token_id']
        vocab.bos_token_id = checkpoint['bos_token_id']
        vocab.eos_token_id = checkpoint['eos_token_id']
        vocab.unk_token_id = checkpoint['unk_token_id']
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model = VQAModel(
            vocab_size=len(checkpoint['vocab']),
            device=self.device,
            question_max_len=checkpoint.get('question_max_len', 20),
            answer_max_len=checkpoint.get('answer_max_len', 12),
            pad_token_id=checkpoint['pad_token_id'],
            bos_token_id=checkpoint['bos_token_id'],
            eos_token_id=checkpoint['eos_token_id'],
            unk_token_id=checkpoint['unk_token_id'],
            hidden_size=512,
            num_layers=2
        ).to(self.device)
        model.gpt2_model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        return model, vocab, tokenizer
    def _load_spatial_model(self, checkpoint_path):
        """Load spatial adapter model."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        vocab = Vocab()
        vocab.vocab = checkpoint['vocab']
        vocab.vocab_size = len(checkpoint['vocab'])
        vocab.word2idx = checkpoint['word2idx']
        vocab.idx2word = checkpoint['idx2word']
        vocab.pad_token_id = checkpoint['pad_token_id']
        vocab.bos_token_id = checkpoint['bos_token_id']
        vocab.eos_token_id = checkpoint['eos_token_id']
        vocab.unk_token_id = checkpoint['unk_token_id']
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        base_model = VQAModel(
            vocab_size=len(checkpoint['vocab']),
            device=self.device,
            question_max_len=checkpoint.get('question_max_len', 20),
            answer_max_len=checkpoint.get('answer_max_len', 12),
            pad_token_id=checkpoint['pad_token_id'],
            bos_token_id=checkpoint['bos_token_id'],
            eos_token_id=checkpoint['eos_token_id'],
            unk_token_id=checkpoint['unk_token_id'],
            hidden_size=512,
            num_layers=2
        ).to(self.device)
        base_model.gpt2_model.resize_token_embeddings(len(tokenizer))
        model = VQAModelWithSpatialAdapter(
            base_model=base_model,
            hidden_size=512,
            num_heads=8,
            dropout=0.3
        ).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        return model, vocab, tokenizer
    def is_spatial_question(self, question):
        """
        Classify if a question is spatial using keyword matching.
        Args:
            question: Question string
        Returns:
            bool: True if spatial, False otherwise
        """
        q_lower = question.lower()
        return any(keyword in q_lower for keyword in self.SPATIAL_KEYWORDS)
    def answer(self, image_path, question, use_beam_search=True, beam_width=5, verbose=False):
        """
        Answer a question by routing to appropriate model.
        Now with Neuro-Symbolic reasoning for common-sense questions!
        Args:
            image_path: Path to image file
            question: Question string
            use_beam_search: Whether to use beam search (better quality)
            beam_width: Beam width for beam search
            verbose: Print routing information
        Returns:
            dict: {
                'answer': str,
                'model_used': 'spatial' or 'base',
                'confidence': float,
                'kg_enhancement': str (optional),
                'reasoning_type': 'neural' or 'neuro-symbolic'
            }
        """
        is_spatial = self.is_spatial_question(question)
        model_used = 'spatial' if is_spatial else 'base'
        if verbose:
            print(f"🔍 Question type: {'Spatial' if is_spatial else 'General'}")
            print(f"🤖 Using: {model_used} model")
        model = self.spatial_model if is_spatial else self.base_model
        image = Image.open(image_path).convert('RGB')
        image = model.clip_preprocess(image).unsqueeze(0).to(self.device)
        question_tokens = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=model.question_max_len,
            return_tensors='pt'
        )
        questions = {
            'input_ids': question_tokens['input_ids'].to(self.device),
            'attention_mask': question_tokens['attention_mask'].to(self.device)
        }
        with torch.no_grad():
            if use_beam_search and hasattr(model, 'generate_with_beam_search'):
                generated = model.generate_with_beam_search(
                    image, questions, beam_width=beam_width
                )
            else:
                generated = model(image, questions)
        if self.kg_enabled and self.kg_service:
            if verbose:
                print("🔍 Analyzing question semantics...")
            should_use_ns = self.kg_service.should_use_neurosymbolic(
                image_features=None,
                question=question,
                vqa_confidence=0.0
            )
            if should_use_ns:
                if verbose:
                    print("🧠 Routing to Neuro-Symbolic (reasoning question)...")
                    print("   → Detecting objects using VQA model (neural part)...")

                # NEURAL: VQA model detects objects from the image
                detected_objects = self._detect_multiple_objects(image, model, top_k=5)

                if verbose:
                    print(f"   → Objects detected: {detected_objects}")
                    print("   → Fetching Wikidata facts + Groq verbalization...")

                # SYMBOLIC: rule engine queries Wikidata properties (P2101, P2054, P2777, P31...)
                ns_result = self.kg_service.answer_with_clip_features(
                    image_features=None,
                    question=question,
                    detected_objects=detected_objects
                )

                if ns_result:
                    answer_text = ns_result['kg_enhancement']
                    if verbose:
                        print(f"✨ Neuro-Symbolic Answer: {answer_text}")
                        entity = ns_result.get('wikidata_entity', '')
                        print(f"   → Wikidata entity: {entity}")
                    return {
                        'answer': ', '.join(detected_objects[:3]),
                        'model_used': 'neuro-symbolic',
                        'confidence': 1.0,
                        'kg_enhancement': answer_text,
                        'reasoning_type': 'neuro-symbolic',
                        'objects_detected': detected_objects,
                        'question_intent': ns_result.get('question_intent'),
                        'wikidata_entity': ns_result.get('wikidata_entity'),
                        'knowledge_source': ns_result.get('knowledge_source'),
                    }

                if verbose:
                    print("   → No Wikidata rule matched, falling back to neural VQA")

        if verbose:
            print("📝 Using neural VQA...")
        answer = self.vocab.decoder(generated[0].cpu().numpy())
        return {
            'answer': answer,
            'model_used': model_used,
            'confidence': 1.0,
            'kg_enhancement': None,
            'reasoning_type': 'neural'
        }
    def answer_conversational(
        self,
        image_path: str,
        question: str,
        session_id: Optional[str] = None,
        use_beam_search: bool = True,
        beam_width: int = 5,
        verbose: bool = False
    ) -> dict:
        """
        Answer a question with multi-turn conversation support.
        Handles pronoun resolution and context management.
        Args:
            image_path: Path to image file
            question: Question string (may contain pronouns like "it", "this")
            session_id: Optional session ID for continuing conversation
            use_beam_search: Whether to use beam search
            beam_width: Beam width for beam search
            verbose: Print routing information
        Returns:
            dict: {
                'answer': str,
                'session_id': str,
                'resolved_question': str,
                'conversation_context': dict,
                ... (other fields from answer())
            }
        """
        if not self.conversation_enabled or not self.conversation_manager:
            result = self.answer(image_path, question, use_beam_search, beam_width, verbose)
            result['session_id'] = None
            result['resolved_question'] = question
            result['conversation_context'] = {'has_context': False}
            return result
        session = self.conversation_manager.get_or_create_session(session_id, image_path)
        actual_session_id = session.session_id
        if verbose:
            print(f"💬 Session: {actual_session_id}")
            print(f"   Turn number: {len(session.history) + 1}")
        resolved_question = self.conversation_manager.resolve_references(question, session)
        if verbose and resolved_question != question:
            print(f"🔄 Pronoun resolution:")
            print(f"   Original: {question}")
            print(f"   Resolved: {resolved_question}")
        result = self.answer(
            image_path=image_path,
            question=resolved_question,
            use_beam_search=use_beam_search,
            beam_width=beam_width,
            verbose=verbose
        )
        self.conversation_manager.add_turn(
            session_id=actual_session_id,
            question=question,
            answer=result['answer'],
            objects_detected=result.get('objects_detected', []),
            reasoning_chain=result.get('reasoning_chain'),
            model_used=result.get('model_used')
        )
        context = self.conversation_manager.get_context_for_question(
            actual_session_id,
            question
        )
        result['session_id'] = actual_session_id
        result['resolved_question'] = resolved_question
        result['conversation_context'] = context
        return result
    def _detect_multiple_objects(self, image, vqa_model, top_k=5):
        """
        Detect multiple objects in the image using VQA's visual features.
        Uses different question variations to get multiple objects.
        """
        detected = []
        detection_questions = [
            "What is this?",
            "What food is this?",
            "What object is this?",
            "What is in the image?",
            "What do you see?",
        ]
        for question in detection_questions:
            try:
                question_tokens = self.tokenizer(
                    question,
                    padding='max_length',
                    truncation=True,
                    max_length=vqa_model.question_max_len,
                    return_tensors='pt'
                )
                questions = {
                    'input_ids': question_tokens['input_ids'].to(self.device),
                    'attention_mask': question_tokens['attention_mask'].to(self.device)
                }
                with torch.no_grad():
                    generated = vqa_model(image, questions)
                answer = self.vocab.decoder(generated[0].cpu().numpy())
                if answer and answer.strip() and answer not in detected:
                    if answer.lower() not in ['a', 'an', 'the', 'this', 'that', 'it', 'yes', 'no']:
                        detected.append(answer.strip())
                        if len(detected) >= top_k:
                            break
            except Exception as e:
                print(f"   ⚠️  Error detecting objects: {e}")
                continue
        if not detected:
            print("   ⚠️  No objects detected, using fallback")
            try:
                question_tokens = self.tokenizer(
                    "What is this?",
                    padding='max_length',
                    truncation=True,
                    max_length=vqa_model.question_max_len,
                    return_tensors='pt'
                )
                questions = {
                    'input_ids': question_tokens['input_ids'].to(self.device),
                    'attention_mask': question_tokens['attention_mask'].to(self.device)
                }
                with torch.no_grad():
                    generated = vqa_model(image, questions)
                answer = self.vocab.decoder(generated[0].cpu().numpy())
                if answer and answer.strip():
                    detected = [answer.strip()]
            except:
                pass
        return detected if detected else ["food"]
    def batch_answer(self, image_question_pairs, use_beam_search=True, verbose=False):
        """
        Answer multiple questions efficiently.
        Args:
            image_question_pairs: List of (image_path, question) tuples
            use_beam_search: Whether to use beam search
            verbose: Print progress
        Returns:
            List of result dicts
        """
        results = []
        total = len(image_question_pairs)
        for i, (image_path, question) in enumerate(image_question_pairs):
            if verbose:
                print(f"\n[{i+1}/{total}] Processing...")
            result = self.answer(image_path, question, use_beam_search, verbose=verbose)
            results.append(result)
        return results
def demo():
    """Demo usage of production ensemble VQA."""
    BASE_CHECKPOINT = "./output2/continued_training/vqa_checkpoint.pt"
    SPATIAL_CHECKPOINT = "./output2/spatial_adapter_v2_2/vqa_spatial_checkpoint.pt"
    IMAGE = "./im2.jpg"
    ensemble = ProductionEnsembleVQA(BASE_CHECKPOINT, SPATIAL_CHECKPOINT)
    test_cases = [
        ("what is to the right of the soup?", True),
        ("what is on the left side?", True),
        ("what is above the table?", True),
        ("what is next to the bowl?", True),
        ("what color is the bowl?", False),
        ("how many items are there?", False),
        ("what room is this?", False),
        ("is there a spoon?", False),
    ]
    print("\n" + "="*80)
    print("🧪 TESTING ENSEMBLE VQA SYSTEM")
    print("="*80)
    print(f"\n📷 Image: {IMAGE}\n")
    for question, expected_spatial in test_cases:
        result = ensemble.answer(IMAGE, question, verbose=False)
        is_spatial = result['model_used'] == 'spatial'
        routing_correct = "✓" if is_spatial == expected_spatial else "✗"
        print(f"Q: {question}")
        print(f"A: {result['answer']}")
        print(f"Model: {result['model_used']} {routing_correct}")
        print()
    print("="*80)
    print("✅ Demo complete!")
def interactive_mode():
    """Interactive mode for testing."""
    BASE_CHECKPOINT = "./output2/continued_training/vqa_checkpoint.pt"
    SPATIAL_CHECKPOINT = "./output2/spatial_adapter_v2_2/vqa_spatial_checkpoint.pt"
    ensemble = ProductionEnsembleVQA(BASE_CHECKPOINT, SPATIAL_CHECKPOINT)
    print("\n" + "="*80)
    print("🎮 INTERACTIVE MODE")
    print("="*80)
    print("\nCommands:")
    print("  - Enter image path and question")
    print("  - Type 'quit' to exit")
    print("="*80 + "\n")
    while True:
        try:
            image_path = input("📷 Image path: ").strip()
            if image_path.lower() == 'quit':
                break
            question = input("❓ Question: ").strip()
            if question.lower() == 'quit':
                break
            result = ensemble.answer(image_path, question, verbose=True)
            print(f"\n💬 Answer: {result['answer']}\n")
            print("-"*80 + "\n")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        demo()