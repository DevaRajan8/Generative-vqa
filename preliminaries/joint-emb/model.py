import torch
import torch.nn as nn
import os
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset

class EasyVQADataset(Dataset):
    def __init__(self, image_dir, data, vocab=None, ans_vocab=None, max_len=10, transform=None):
        self.image_dir = image_dir
        self.data = data
        self.transform = transform
        self.max_len = max_len
        
        if vocab is None:
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab
            
        if ans_vocab is None:
            self.ans_vocab = self.build_answer_vocab()
        else:
            self.ans_vocab = ans_vocab
    
    def build_vocab(self):
        words = []
        for item in self.data:
            words.extend(item['question'].lower().split())
        word_counts = Counter(words)
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in word_counts.most_common():
            vocab[word] = len(vocab)
        return vocab
    
    def build_answer_vocab(self):
        answers = set()
        for item in self.data:
            answers.add(item['answer'])
        ans_vocab = {ans: idx for idx, ans in enumerate(sorted(list(answers)))}
        return ans_vocab
    
    def tokenize(self, text):
        tokens = text.lower().split()
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in tokens]
        if len(indices) < self.max_len:
            indices += [self.vocab['<PAD>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_name = item['image_filename']
        question = item['question']
        answer = item['answer']
        
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        question_tokens = self.tokenize(question)
        answer_idx = self.ans_vocab[answer]
        
        return image, question_tokens, answer_idx

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 8 * 8, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=256, max_len=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 300)
        self.lstm = nn.LSTM(300, hidden_dim, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim * 2, embed_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        x = self.fc(hidden)
        return x

class AnswerEmbedding(nn.Module):
    def __init__(self, num_answers, embed_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(num_answers, embed_dim)
        
    def forward(self, x):
        return self.embedding(x)

class JointEmbeddingVQA(nn.Module):
    def __init__(self, vocab_size, num_answers, embed_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim)
        self.answer_embedding = AnswerEmbedding(num_answers, embed_dim)
        self.num_answers = num_answers
        
    def forward(self, images, questions):
        img_emb = self.image_encoder(images)
        q_emb = self.text_encoder(questions)
        combined = img_emb + q_emb
        combined = nn.functional.normalize(combined, p=2, dim=1)
        
        all_answers = torch.arange(self.num_answers, device=images.device)
        ans_emb = self.answer_embedding(all_answers)
        ans_emb = nn.functional.normalize(ans_emb, p=2, dim=1)
        
        logits = torch.matmul(combined, ans_emb.t())
        return logits

def contrastive_loss(logits, targets, temperature=0.3,class_weights=None):
    logits = logits / temperature
    if class_weights is not None:
        loss = nn.functional.cross_entropy(logits, targets, weight=class_weights)
    else:
        loss = nn.functional.cross_entropy(logits, targets)
    
    return loss