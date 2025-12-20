import torch
import numpy as np
from PIL import Image
from torchvision import transforms
def evaluate_on_test_set(model, test_images_dir, test_json, vocab, ans_vocab, device):

    import json
    
    
    with open(test_json, 'r') as f:
        test_data = json.load(f)
    
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    idx_to_ans = {v: k for k, v in ans_vocab.items()}
    
    model.eval()
    results = []
    
    print("\n")
    print("TESTING ON UNSEEN DATA")
    print("/n")
    
    with torch.no_grad():
        for item in test_data:
            img_path = f"{test_images_dir}/{item['image_filename']}"
            
            
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            
            for qa in item['qa_pairs']:
                question = qa['question']
                true_answer = qa['answer']
                
                
                tokens = question.lower().split()
                indices = [vocab.get(word, vocab['<UNK>']) for word in tokens]
                
                
                max_len = 10
                if len(indices) < max_len:
                    indices += [vocab['<PAD>']] * (max_len - len(indices))
                else:
                    indices = indices[:max_len]
                
                question_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
                
                
                logits = model(image_tensor, question_tensor)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                confidence, predicted_idx = probabilities.max(1)
                
                predicted_answer = idx_to_ans[predicted_idx.item()]
                confidence_score = confidence.item()
                
                
                top3_probs, top3_indices = probabilities.topk(3, dim=1)
                top3_answers = [(idx_to_ans[idx.item()], prob.item()) 
                               for idx, prob in zip(top3_indices[0], top3_probs[0])]
                
                is_correct = (predicted_answer == true_answer)
                
                results.append({
                    'image': item['image_filename'],
                    'question': question,
                    'true_answer': true_answer,
                    'predicted_answer': predicted_answer,
                    'confidence': confidence_score,
                    'correct': is_correct,
                    'top3': top3_answers
                })
                
                
                status = "ok" if is_correct else "No"
                print(f"\n{status} Image: {item['image_filename']}")
                print(f"   Q: {question}")
                print(f"   True: {true_answer}")
                print(f"   Pred: {predicted_answer} ({confidence_score*100:.1f}% confidence)")
                print(f"   Top 3: {', '.join([f'{ans}({prob*100:.1f}%)' for ans, prob in top3_answers])}")
    
    
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = 100.0 * correct / total if total > 0 else 0
    avg_confidence = np.mean([r['confidence'] for r in results]) * 100
    
    print("\n")
    print(f"TEST SET RESULTS")
    print("/n")
    print(f"Total Questions: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Confidence: {avg_confidence:.2f}%")
    print("/n")
    
    return results, accuracy