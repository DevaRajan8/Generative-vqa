import os
import torch
from PIL import Image
from transformers import GPT2Tokenizer
from model import VQAModel
from train import Vocab
def load_model(checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    vocab = Vocab()
    vocab.vocab = checkpoint['vocab']
    vocab.vocab_size = len(checkpoint['vocab'])
    vocab.word2idx = checkpoint['word2idx']
    vocab.idx2word = checkpoint['idx2word']
    vocab.pad_token_id = checkpoint['pad_token_id']
    vocab.bos_token_id = checkpoint['bos_token_id']
    vocab.eos_token_id = checkpoint['eos_token_id']
    vocab.unk_token_id = checkpoint['unk_token_id']
    
    model = VQAModel(
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
    
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.gpt2_model.resize_token_embeddings(len(tokenizer))
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    return model, vocab, tokenizer
def answer_question(model, vocab, tokenizer, image_path, question, device='cuda', use_beam_search=True, beam_width=5, temperature=0.8):
    image = Image.open(image_path).convert('RGB')
    image = model.clip_preprocess(image).unsqueeze(0).to(device)
    
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
    
    with torch.no_grad():
        if use_beam_search and hasattr(model, 'generate_with_beam_search'):
            generated = model.generate_with_beam_search(image, questions, beam_width=beam_width)
        else:
            generated = model(image, questions)
    
    answer = vocab.decoder(generated[0].cpu().numpy())
    return answer
CHECKPOINT = "./output2/spatial_adapter_v2_2/vqa_spatial_checkpoint.pt"
IMAGE_PATH = r"./im2.jpg"
QUESTION = ""
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading model...")
    model, vocab, tokenizer = load_model(CHECKPOINT, device)
    print("Model loaded!\n")
    
    test_questions = [
        "What is to the right of the soup?"
    ]
    
    print(f"Image: {IMAGE_PATH}\n")
    
    for question in test_questions:
        print(f"Question: {question}")
        answer = answer_question(model, vocab, tokenizer, IMAGE_PATH, question, device, use_beam_search=True, beam_width=5)
        print(f"Answer: {answer}\n")