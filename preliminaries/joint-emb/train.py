import torch
from model import contrastive_loss

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0
        
    def __call__(self, val_acc, model, path='best_model.pth'):
        score = val_acc
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model, path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model, path)
            self.counter = 0
            
    def save_checkpoint(self, val_acc, model, path):
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_acc_max:.2f}% --> {val_acc:.2f}%). Saving model...')
        torch.save(model.state_dict(), path)
        self.val_acc_max = val_acc

def train_one_epoch(model, dataloader, optimizer, device, class_weights=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, questions, answers in dataloader:
        images = images.to(device)
        questions = questions.to(device)
        answers = answers.to(device)
        
        optimizer.zero_grad()
        logits = model(images, questions)
        loss = contrastive_loss(logits, answers,class_weights=class_weights)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += (predicted == answers).sum().item()
        total += answers.size(0)
    
    return total_loss / len(dataloader), 100.0 * correct / total