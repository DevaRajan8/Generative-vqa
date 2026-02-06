"""
Production Ensemble VQA Application
Combines base model (general VQA) and spatial adapter (spatial reasoning)
for optimal performance on all question types.
"""

import os
import torch
from PIL import Image
from transformers import GPT2Tokenizer
from model import VQAModel
from model_spatial import VQAModelWithSpatialAdapter
from train import Vocab
import time


class ProductionEnsembleVQA:
    """
    Production-ready ensemble VQA system.
    - Base model for general questions (39.4% accuracy)
    - Spatial model for spatial questions (28.5% accuracy)
    - Automatic question routing
    """
    
    # Comprehensive spatial keywords for classification
    SPATIAL_KEYWORDS = [
        # Directional
        'right', 'left', 'above', 'below', 'top', 'bottom',
        'up', 'down', 'upward', 'downward',
        
        # Positional
        'front', 'behind', 'back', 'next to', 'beside', 'near', 'between',
        'in front', 'in back', 'across from', 'opposite', 'adjacent',
        
        # Relational
        'closest', 'farthest', 'nearest', 'furthest', 'closer', 'farther',
        
        # Spatial queries
        'where is', 'where are', 'which side', 'what side', 'what direction',
        'on the left', 'on the right', 'at the top', 'at the bottom',
        'to the left', 'to the right', 'in the middle', 'in the center',
        
        # Spatial prepositions
        'under', 'over', 'underneath', 'on top of', 'inside', 'outside'
    ]
    
    def __init__(self, base_checkpoint, spatial_checkpoint, device='cuda'):
        """
        Initialize ensemble with both models.
        
        Args:
            base_checkpoint: Path to base model checkpoint
            spatial_checkpoint: Path to spatial adapter checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print("="*80)
        print("🚀 INITIALIZING ENSEMBLE VQA SYSTEM")
        print("="*80)
        print(f"\n⚙️  Device: {self.device}")
        
        # Load models
        print("\n📥 Loading models...")
        start_time = time.time()
        
        print("  [1/2] Loading base model (general VQA)...")
        self.base_model, self.vocab, self.tokenizer = self._load_base_model(base_checkpoint)
        print("      ✓ Base model loaded")
        
        print("  [2/2] Loading spatial model (spatial reasoning)...")
        self.spatial_model, _, _ = self._load_spatial_model(spatial_checkpoint)
        print("      ✓ Spatial model loaded")
        
        load_time = time.time() - start_time
        
        print(f"\n✅ Ensemble ready! (loaded in {load_time:.1f}s)")
        print(f"📊 Memory: ~2x single model (~4GB GPU)")
        print(f"🎯 Routing: Automatic based on question type")
        print("="*80)
    
    def _load_base_model(self, checkpoint_path):
        """Load base VQA model."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load vocabulary
        vocab = Vocab()
        vocab.vocab = checkpoint['vocab']
        vocab.vocab_size = len(checkpoint['vocab'])
        vocab.word2idx = checkpoint['word2idx']
        vocab.idx2word = checkpoint['idx2word']
        vocab.pad_token_id = checkpoint['pad_token_id']
        vocab.bos_token_id = checkpoint['bos_token_id']
        vocab.eos_token_id = checkpoint['eos_token_id']
        vocab.unk_token_id = checkpoint['unk_token_id']
        
        # Setup tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        # Create model
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
        
        # Load vocabulary
        vocab = Vocab()
        vocab.vocab = checkpoint['vocab']
        vocab.vocab_size = len(checkpoint['vocab'])
        vocab.word2idx = checkpoint['word2idx']
        vocab.idx2word = checkpoint['idx2word']
        vocab.pad_token_id = checkpoint['pad_token_id']
        vocab.bos_token_id = checkpoint['bos_token_id']
        vocab.eos_token_id = checkpoint['eos_token_id']
        vocab.unk_token_id = checkpoint['unk_token_id']
        
        # Setup tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        # Create base model
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
        
        # Create spatial adapter model
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
                'confidence': float (placeholder for future)
            }
        """
        # Classify question
        is_spatial = self.is_spatial_question(question)
        model_used = 'spatial' if is_spatial else 'base'
        
        if verbose:
            print(f"🔍 Question type: {'Spatial' if is_spatial else 'General'}")
            print(f"🤖 Using: {model_used} model")
        
        # Select model
        model = self.spatial_model if is_spatial else self.base_model
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = model.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        # Tokenize question
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
        
        # Generate answer
        with torch.no_grad():
            if use_beam_search and hasattr(model, 'generate_with_beam_search'):
                generated = model.generate_with_beam_search(
                    image, questions, beam_width=beam_width
                )
            else:
                generated = model(image, questions)
        
        answer = self.vocab.decoder(generated[0].cpu().numpy())
        
        return {
            'answer': answer,
            'model_used': model_used,
            'confidence': 1.0  # Placeholder for future confidence scoring
        }
    
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
    
    # Configuration
    BASE_CHECKPOINT = "./output2/continued_training/vqa_checkpoint.pt"
    SPATIAL_CHECKPOINT = "./output2/spatial_adapter_v2_2/vqa_spatial_checkpoint.pt"
    IMAGE = "./im2.jpg"
    
    # Initialize ensemble
    ensemble = ProductionEnsembleVQA(BASE_CHECKPOINT, SPATIAL_CHECKPOINT)
    
    # Test questions (mixed spatial and general)
    test_cases = [
        # Spatial questions (will use spatial model)
        ("what is to the right of the soup?", True),
        ("what is on the left side?", True),
        ("what is above the table?", True),
        ("what is next to the bowl?", True),
        
        # General questions (will use base model)
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
        
        # Verify routing
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
