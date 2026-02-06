import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from torchvision import transforms
from models.model import VQAModel
device = 'cuda'
class Vocab:
    def __init__(self):
        self.vocab = None
        self.vocab_size = None
        self.word2idx = None
        self.idx2word = None
        
        self.pad = '<pad>'
        self.bos = '<bos>'
        self.eos = '<eos>'
        self.unk = '<unk>'
    
    def build_vocab(self, df, min_freq=1):
        counter = Counter()
        for ans in df['answer']:
            tokens = word_tokenize(ans.lower())
            counter.update(tokens)
        
        vocab = sorted([word for word, freq in counter.items() if freq >= min_freq])
        vocab = [self.pad, self.bos, self.eos, self.unk] + vocab
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for word, idx in word2idx.items()}
        
        self.vocab = vocab
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.vocab_size = len(vocab)
        
        self.pad_token_id = self.word2idx["<pad>"]
        self.bos_token_id = self.word2idx["<bos>"]
        self.eos_token_id = self.word2idx["<eos>"]
        self.unk_token_id = self.word2idx["<unk>"]
    
    def encoder(self, text, max_len):
        tokens = word_tokenize(text.lower())
        
        token_ids = [self.word2idx.get(token, self.unk_token_id) for token in tokens]
        token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        
        if len(token_ids) < max_len:
            token_ids += [self.pad_token_id] * (max_len - len(token_ids))
        else:
            token_ids = token_ids[:max_len]
        
        return token_ids
    
    def decoder(self, token_ids):
        tokens = []
        for idx in token_ids:
            if idx == self.eos_token_id:
                break
            if idx in (self.pad_token_id, self.bos_token_id):
                continue
            tokens.append(self.idx2word.get(idx, "<unk>"))
        return ' '.join(tokens).strip()
class AugmentedVQADataset(Dataset):
    def __init__(self, df, img_dir, question_tokenizer, text_processor, clip_processor,
                 question_max_len=32, answer_max_len=16, augment=True):
        self.df = df
        self.img_dir = img_dir
        self.question_tokenizer = question_tokenizer
        self.text_processor = text_processor
        self.clip_processor = clip_processor
        self.question_max_len = question_max_len
        self.answer_max_len = answer_max_len
        self.augment = augment
        
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(10),
            ])
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_path'])
        image = Image.open(img_path).convert('RGB')
        question = row['question']
        answer = row['answer']
        
        if self.augment and self.transform:
            image = self.transform(image)
        
        question_tokenized = self.question_tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=self.question_max_len,
            return_tensors='pt'
        )
        
        answer_ids = self.text_processor.encoder(answer, max_len=self.answer_max_len)
        
        image = self.clip_processor(image)
        
        return {
            'image_path': img_path,
            'image': image,
            'question_ids': question_tokenized['input_ids'].squeeze(0),
            'question_mask': question_tokenized['attention_mask'].squeeze(0),
            'answer_ids': torch.tensor(answer_ids, dtype=torch.long)
        }
def save_checkpoint(model, optimizer, epoch, vocab, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab': vocab.vocab,
        'word2idx': vocab.word2idx,
        'idx2word': vocab.idx2word,
        'pad_token_id': vocab.pad_token_id,
        'bos_token_id': vocab.bos_token_id,
        'eos_token_id': vocab.eos_token_id,
        'unk_token_id': vocab.unk_token_id,
        'question_max_len': model.question_max_len,
        'answer_max_len': model.answer_max_len
    }, path)
def plot_losses(train_losses, val_losses, save_path="loss_plot.png"):
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
def train_one_epoch(model, dataloader, optimizer, device, scaler, vocab):
    model.train()
    total_loss = 0
    total_token_acc = 0
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_token_id, label_smoothing=0.1)
    
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        images = batch['image'].to(device)
        questions = {
            'input_ids': batch['question_ids'].to(device),
            'attention_mask': batch['question_mask'].to(device)
        }
        answers = batch['answer_ids'].to(device)
        
        with torch.amp.autocast(device):
            logits = model(images, questions, answer_input_ids=answers)
            
            shifted_logits = logits[:, :-1, :]
            shifted_answers = answers[:, 1:]
            
            loss = criterion(
                shifted_logits.reshape(-1, shifted_logits.size(-1)),
                shifted_answers.reshape(-1)
            )
            
            predicted_tokens = shifted_logits.argmax(dim=-1)
            correct = (predicted_tokens == shifted_answers).float()
            mask = (shifted_answers != vocab.pad_token_id).float()
            token_acc = (correct * mask).sum() / mask.sum()
            total_token_acc += token_acc.item()
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_token_acc = total_token_acc / len(dataloader)
    return avg_loss, avg_token_acc
def validate_one_epoch(model, dataloader, device, vocab):
    model.eval()
    total_loss = 0
    total_token_acc = 0
    exact_matches = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_token_id)
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            questions = {
                'input_ids': batch['question_ids'].to(device),
                'attention_mask': batch['question_mask'].to(device)
            }
            answers = batch['answer_ids'].to(device)
            
            logits = model(images, questions, answer_input_ids=answers)
            
            shifted_logits = logits[:, :-1, :]
            shifted_answers = answers[:, 1:]
            
            loss = criterion(
                shifted_logits.reshape(-1, shifted_logits.size(-1)),
                shifted_answers.reshape(-1)
            )
            total_loss += loss.item()
            
            predicted_tokens = shifted_logits.argmax(dim=-1)
            correct = (predicted_tokens == shifted_answers).float()
            mask = (shifted_answers != vocab.pad_token_id).float()
            token_acc = (correct * mask).sum() / mask.sum()
            total_token_acc += token_acc.item()
            
            if hasattr(model, 'generate_with_beam_search'):
                generated = model.generate_with_beam_search(images, questions, beam_width=3)
            else:
                generated = model(images, questions)
            
            for pred, true in zip(generated, answers):
                pred_text = vocab.decoder(pred.cpu().numpy())
                true_text = vocab.decoder(true.cpu().numpy())
                if pred_text.strip() == true_text.strip():
                    exact_matches += 1
                total_samples += 1
    
    avg_loss = total_loss / len(dataloader)
    avg_token_acc = total_token_acc / len(dataloader)
    exact_match_acc = exact_matches / total_samples
    
    return avg_loss, avg_token_acc, exact_match_acc
def main():
    print()
    print("# VQA: Training with Staged Unfreezing")
 
    print()
    
    import random
    import numpy as np
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    
    DATA_DIR = r"./gen_vqa_v2"
    CSV_PATH = os.path.join(DATA_DIR, "metadata.csv")
    OUTPUT_DIR = r"./output2/feature_extraction"
    CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "vqa_checkpoint.pt")
    LOG_CSV = os.path.join(OUTPUT_DIR, "train_log.csv")
    LOSS_GRAPH_PATH = os.path.join(OUTPUT_DIR, "loss_plot.png")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    batch_size = 64
    learning_rate = 1e-4
    num_epochs = 30
    patience = 8
    question_max_len = 20
    answer_max_len = 12
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    metadata = pd.read_csv(CSV_PATH)
    
    print(f"Using: question_max_len={question_max_len}, answer_max_len={answer_max_len}")
    
    vocab = Vocab()
    vocab.build_vocab(metadata, min_freq=3)
    answer_vocab_size = len(vocab.vocab)
    print(f"Answer Vocab Size: {answer_vocab_size}")
    
    word_freq = Counter()
    for ans in metadata['answer']:
        tokens = word_tokenize(ans.lower())
        word_freq.update(tokens)
    print("\nTop 20 most common answer words:")
    for word, freq in word_freq.most_common(20):
        print(f"  {word}: {freq}")
    
    train_df, test_df = train_test_split(metadata, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
    print(f"\nTrain size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    print()
    
    model = VQAModel(
        vocab_size=answer_vocab_size,
        device=device,
        question_max_len=question_max_len,
        answer_max_len=answer_max_len,
        pad_token_id=vocab.pad_token_id,
        bos_token_id=vocab.bos_token_id,
        eos_token_id=vocab.eos_token_id,
        unk_token_id=vocab.unk_token_id,
        hidden_size=512,
        num_layers=2
    ).to(device)
    
    print("STAGE 1: Training decoder with frozen encoders")
    print()
    
    clip_processor = model.clip_preprocess
    question_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    
    if question_tokenizer.pad_token is None:
        question_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.gpt2_model.resize_token_embeddings(len(question_tokenizer))
    
    train_dataset = AugmentedVQADataset(
        train_df, DATA_DIR, question_tokenizer, vocab,
        clip_processor=clip_processor,
        question_max_len=question_max_len,
        answer_max_len=answer_max_len,
        augment=True
    )
    val_dataset = AugmentedVQADataset(
        val_df, DATA_DIR, question_tokenizer, vocab,
        clip_processor=clip_processor,
        question_max_len=question_max_len,
        answer_max_len=answer_max_len,
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)
    
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print()
    
    scaler = torch.amp.GradScaler(device)
    
    best_val_loss = np.inf
    best_val_exact_match = 0.0
    counter = 0
    logs = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4, verbose=True
    )
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_token_acc = train_one_epoch(model, train_loader, optimizer, device, scaler, vocab)
        val_loss, val_token_acc, val_exact_match = validate_one_epoch(model, val_loader, device, vocab)
        
        print(f"Train Loss: {train_loss:.4f} | Train Token Acc: {train_token_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Token Acc: {val_token_acc:.4f} | Val Exact Match: {val_exact_match:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']}")
        
        scheduler.step(val_exact_match)
        
        if val_exact_match > best_val_exact_match:
            best_val_exact_match = val_exact_match
            save_checkpoint(model, optimizer, epoch, vocab, CHECKPOINT_PATH)
            print("Checkpoint saved!")
            counter = 0
        else:
            counter += 1
            print(f"No improvement in exact match for {counter} epochs.")
        
        if epoch == 15 and not model.fine_tuning_mode:
            print("\n" + "="*50)
            print("STAGE 2: Unfreezing encoders for fine-tuning")
            print("="*50)
            model.unfreeze_clip_layers(num_layers=3)
            model.unfreeze_gpt2_layers(num_layers=3)
            
            clip_params = []
            gpt2_params = []
            other_params = []
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'clip_model' in name:
                        clip_params.append(param)
                    elif 'gpt2_model' in name:
                        gpt2_params.append(param)
                    else:
                        other_params.append(param)
            
            optimizer = torch.optim.AdamW([
                {'params': clip_params, 'lr': 1e-6},
                {'params': gpt2_params, 'lr': 1e-6},
                {'params': other_params, 'lr': 5e-5}
            ], weight_decay=1e-4)
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=4, verbose=True
            )
            print()
        
        if counter >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement")
        
        logs.append([epoch+1, train_loss, train_token_acc, val_loss, val_token_acc, val_exact_match, optimizer.param_groups[0]['lr']])
    
    log_df = pd.DataFrame(logs, columns=["epoch","train_loss","train_token_acc","val_loss","val_token_acc","val_exact_match","lr"])
    log_df.to_csv(LOG_CSV, index=False)
    
    plot_losses([x[1] for x in logs], [x[3] for x in logs], save_path=LOSS_GRAPH_PATH)
    
    print("Training complete!")
    print(f"Best exact match accuracy: {best_val_exact_match:.4f}")
if __name__ == "__main__":
    main()