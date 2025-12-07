import torch
from model import contrastive_loss

def validate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for images, questions, answers in dataloader:
            images = images.to(device)
            questions = questions.to(device)
            answers = answers.to(device)
            
            logits = model(images, questions)
            loss = contrastive_loss(logits, answers)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += (predicted == answers).sum().item()
            total += answers.size(0)
    
    return 100.0 * correct / total, total_loss / len(dataloader)