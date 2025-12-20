import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import json
import random
from model import EasyVQADataset, JointEmbeddingVQA
from train import train_one_epoch, EarlyStopping
from test import validate

def process_data(json_file):
    with open(json_file, 'r') as f:
        raw_data = json.load(f)
    
    random.seed(42)
    random.shuffle(raw_data)
    
    split_idx = int(0.8 * len(raw_data))
    train_images = raw_data[:split_idx]
    val_images = raw_data[split_idx:]
    
    def flatten(image_list):
        flat_data = []
        for img_obj in image_list:
            filename = img_obj['image_filename']
            for qa in img_obj['qa_pairs']:
                flat_data.append({
                    'image_filename': filename,
                    'question': qa['question'],
                    'answer': qa['answer']
                })
        return flat_data

    return flatten(train_images), flatten(val_images)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    image_dir = r'C:\Users\rdeva\Downloads\sem6\SynerVQA\easy_vqa_data\images'
    json_file = r'C:\Users\rdeva\Downloads\sem6\SynerVQA\easy_vqa_data\questions_answers.json'
    
    train_data, val_data = process_data(json_file)
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = EasyVQADataset(
        image_dir, 
        train_data,
        transform=transform
    )
    
    val_dataset = EasyVQADataset(
        image_dir,
        val_data,
        vocab=train_dataset.vocab,
        ans_vocab=train_dataset.ans_vocab,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    vocab_size = len(train_dataset.vocab)
    num_answers = len(train_dataset.ans_vocab)
    
    from collections import Counter

    answer_counts = Counter()
    for item in train_data:
        answer_idx = train_dataset.ans_vocab[item['answer']]
        answer_counts[answer_idx] += 1

    total_samples = len(train_data)
    class_weights = torch.zeros(num_answers)
    for ans_idx in range(num_answers):
        count = answer_counts.get(ans_idx, 1)
        class_weights[ans_idx] = total_samples / (num_answers * count)
    
    class_weights = class_weights.to(device)
    
    print(f'Class weights calculated:')
    for ans, idx in sorted(train_dataset.ans_vocab.items(), key=lambda x: x[1]):
        print(f'  {ans}: {class_weights[idx]:.3f}')
    
    print(f'Vocabulary size: {vocab_size}')
    print(f'Number of answers: {num_answers}')
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    
    model = JointEmbeddingVQA(vocab_size, num_answers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=10, min_delta=0.5, verbose=True)
    
    num_epochs = 20
    best_val_acc = 0
    
    print('\nStarting training...\n')
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_acc, val_loss = validate(model, val_loader, device)
        
        scheduler.step(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Current LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f'New best validation accuracy: {best_val_acc:.2f}%')
            
        if epoch == 0:
            initial_train_acc = train_acc
            initial_val_acc = val_acc
        else:
            train_improvement = train_acc - initial_train_acc
            val_improvement = val_acc - initial_val_acc
            print(f'Improvement: Train +{train_improvement:.2f}%, Val +{val_improvement:.2f}%')
        
        early_stopping(val_acc, model, 'best_vqa_model.pth')
        
        if early_stopping.early_stop:
            print('\nEarly stopping triggered!')
            break
        
        print('\n')
    
    print(f'\nTraining completed!')
    print(f'Best Validation Accuracy: {best_val_acc:.2f}%')

if __name__ == '__main__':
    main()