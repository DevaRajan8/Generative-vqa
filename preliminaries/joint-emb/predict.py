import torch
from torchvision import transforms
from PIL import Image
import os
import sys
from model import JointEmbeddingVQA, EasyVQADataset
from main import process_data

def get_vocab_and_model(model_path, json_file, device):
    if not os.path.exists(json_file):
        print(f"Error: '{json_file}' not found. Needed to rebuild vocabulary.")
        sys.exit(1)

    print("Rebuilding vocabulary...")
    train_data, _ = process_data(json_file)
    
    dummy_dataset = EasyVQADataset('images', train_data)
    vocab = dummy_dataset.vocab
    ans_vocab = dummy_dataset.ans_vocab
    
    idx_to_ans = {v: k for k, v in ans_vocab.items()}
    
    model = JointEmbeddingVQA(len(vocab), len(ans_vocab))
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Model weights loaded successfully.")
    else:
        print(f"Error: '{model_path}' not found. You must train the model first!")
        sys.exit(1)
        
    return model, vocab, idx_to_ans

def predict_custom_image(model, image_path, question, vocab, idx_to_ans, device):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: The image file '{image_path}' was not found.")
        return None

    image_tensor = transform(image).unsqueeze(0).to(device)
    
    tokens = question.lower().split()
    indices = [vocab.get(word, vocab['<UNK>']) for word in tokens]
    
    max_len = 10
    if len(indices) < max_len:
        indices += [vocab['<PAD>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
        
    question_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor, question_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        confidence, predicted_idx = probabilities.max(1)
        answer = idx_to_ans[predicted_idx.item()]
        
    return answer, confidence.item()

if __name__ == '__main__':
    MY_IMAGE_PATH = r'C:\Users\rdeva\Downloads\sem6\SynerVQA\easy_vqa_data\images\18.png' 
    MY_QUESTION = "what is the shape?"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model, vocab, idx_to_ans = get_vocab_and_model(
        r'C:\Users\rdeva\Downloads\sem6\SynerVQA\preliminaries\joint-emb\best_vqa_model.pth', 
        r'C:\Users\rdeva\Downloads\sem6\SynerVQA\easy_vqa_data\questions_answers.json', 
        device
    )
    
    print(f"Testing on: {MY_IMAGE_PATH}")
    print(f"Question:   {MY_QUESTION}")
    
    result = predict_custom_image(model, MY_IMAGE_PATH, MY_QUESTION, vocab, idx_to_ans, device)
    
    if result:
        answer, conf = result
        print(f"Prediction: {answer.upper()}")