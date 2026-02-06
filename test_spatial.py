import os
import torch
from PIL import Image
from transformers import GPT2Tokenizer
from model import VQAModel
from model_spatial import VQAModelWithSpatialAdapter
from train import Vocab
def load_spatial_model(checkpoint_path, device='cuda'):
    """Load the fine-tuned spatial adapter model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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
    
    # Setup tokenizer FIRST (before creating model)
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    # Create base model
    base_model = VQAModel(
        vocab_size=len(checkpoint['vocab']),
        device=device,
        question_max_len=checkpoint.get('question_max_len', 20),
        answer_max_len=checkpoint.get('answer_max_len', 12),
        pad_token_id=checkpoint['pad_token_id'],
        bos_token_id=checkpoint['bos_token_id'],
        eos_token_id=checkpoint['eos_token_id'],
        unk_token_id=checkpoint['unk_token_id'],
        hidden_size=512,
        num_layers=2
    ).to(device)
    
    # Resize GPT-2 embeddings to match tokenizer (before loading weights)
    base_model.gpt2_model.resize_token_embeddings(len(tokenizer))
    
    # Create spatial adapter model
    model = VQAModelWithSpatialAdapter(
        base_model=base_model,
        hidden_size=512,
        num_heads=8,
        dropout=0.3
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    return model, vocab, tokenizer
def answer_question(model, vocab, tokenizer, image_path, question, device='cuda', use_beam_search=True, beam_width=5):
    """Answer a question about an image using the spatial adapter model"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = model.clip_preprocess(image).unsqueeze(0).to(device)
    
    # Tokenize question
    question_tokens = tokenizer(
        question,
        padding='max_length',
        truncation=True,
        max_length=model.question_max_len,
        return_tensors='pt'
    )
    
    questions = {
        'input_ids': question_tokens['input_ids'].to(device),
        'attention_mask': question_tokens['attention_mask'].to(device)
    }
    
    # Generate answer
    with torch.no_grad():
        if use_beam_search and hasattr(model, 'generate_with_beam_search'):
            generated = model.generate_with_beam_search(image, questions, beam_width=beam_width)
        else:
            generated = model(image, questions)
    
    answer = vocab.decoder(generated[0].cpu().numpy())
    return answer
# Configuration
SPATIAL_CHECKPOINT = "./output2/spatial_adapter_v2_2/vqa_spatial_checkpoint.pt"  # V2 model (25.59%)
BASE_CHECKPOINT = "./output2/continued_training/vqa_checkpoint.pt"
IMAGE_PATH = r"./im2.jpg"
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 80)
    print("🧪 TESTING SPATIAL ADAPTER MODEL")
    print("=" * 80)
    
    # Test spatial questions
    spatial_questions = [
        "Whic"
    ]
    
    print(f"\n📷 Image: {IMAGE_PATH}\n")
    
    # Test with spatial model if available
    if os.path.exists(SPATIAL_CHECKPOINT):
        print("🔧 Loading SPATIAL ADAPTER model...")
        spatial_model, vocab, tokenizer = load_spatial_model(SPATIAL_CHECKPOINT, device)
        print("✓ Spatial model loaded!\n")
        
        print("-" * 80)
        print("SPATIAL ADAPTER MODEL RESULTS:")
        print("-" * 80)
        for question in spatial_questions:
            answer = answer_question(spatial_model, vocab, tokenizer, IMAGE_PATH, question, device, use_beam_search=True, beam_width=5)
            print(f"\nQ: {question}")
            print(f"A: {answer}")
        
        print("\n" + "=" * 80)
    else:
        print(f"⚠️  Spatial model not found at: {SPATIAL_CHECKPOINT}")
        print("   Run finetune2.py first to train the spatial adapter model.")
    
    # Compare with base model
    if os.path.exists(BASE_CHECKPOINT):
        print("\n🔧 Loading BASE model for comparison...")
        from test import load_model
        base_model, vocab, tokenizer = load_model(BASE_CHECKPOINT, device)
        print("✓ Base model loaded!\n")
        
        print("-" * 80)
        print("BASE MODEL RESULTS (for comparison):")
        print("-" * 80)
        for question in spatial_questions:
            answer = answer_question(base_model, vocab, tokenizer, IMAGE_PATH, question, device, use_beam_search=True, beam_width=5)
            print(f"\nQ: {question}")
            print(f"A: {answer}")
        
        print("\n" + "=" * 80)
