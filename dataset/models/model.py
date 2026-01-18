import torch
from torch import nn
import clip
from transformers import GPT2Model
class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.gru = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.ln_gru = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids, context, hidden):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(1)
        
        embeddings = self.embedding(input_ids).float()
        context_expanded = context.unsqueeze(1).expand(-1, embeddings.size(1), -1)
        
        combined = torch.cat([embeddings, context_expanded], dim=-1)
        attn_weights = torch.softmax(self.attention(combined), dim=1)
        attended_context = (context_expanded * attn_weights).sum(dim=1, keepdim=True)
        
        gru_input = torch.cat([embeddings, attended_context.expand(-1, embeddings.size(1), -1)], dim=-1)
        gru_output, hidden = self.gru(gru_input, hidden)
        gru_output = self.ln_gru(gru_output)
        
        return self.output(gru_output), hidden
class VQAModel(nn.Module):
    def __init__(
        self,
        vocab_size=3600,
        question_max_len=16,
        answer_max_len=10,
        hidden_size=512,
        num_layers=2,
        dropout=0.3,
        device='cuda',
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        unk_token_id=3
    ):
        super().__init__()
        self.device = device
        self.question_max_len = question_max_len
        self.answer_max_len = answer_max_len
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fine_tuning_mode = False
        
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id
        
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=device)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        
        self.gpt2_model = GPT2Model.from_pretrained("distilgpt2")
        self.gpt2_model.to(device)
        for p in self.gpt2_model.parameters():
            p.requires_grad = False
        
        self.img_proj = nn.Linear(512, hidden_size)
        self.q_proj = nn.Linear(768, hidden_size)
        
        self.gate_layer = nn.Linear(hidden_size*2, hidden_size)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size*3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.decoder = AttentionDecoder(hidden_size, vocab_size, num_layers, dropout)
    def unfreeze_clip_layers(self, num_layers=2):
        self.clip_model.train()
        self.clip_model.visual.float()
        
        total_blocks = len(self.clip_model.visual.transformer.resblocks)
        
        for i, block in enumerate(self.clip_model.visual.transformer.resblocks):
            if i >= total_blocks - num_layers:
                for p in block.parameters():
                    p.requires_grad = True
        
        if hasattr(self.clip_model.visual, "proj") and self.clip_model.visual.proj is not None:
            if isinstance(self.clip_model.visual.proj, torch.nn.Parameter):
                self.clip_model.visual.proj.requires_grad = True
            else:
                for p in self.clip_model.visual.proj.parameters():
                    p.requires_grad = True
        
        if hasattr(self.clip_model.visual, "ln_post"):
            for p in self.clip_model.visual.ln_post.parameters():
                p.requires_grad = True
        
        self.fine_tuning_mode = True
        print(f"Unfrozen last {num_layers} CLIP layers")
    def unfreeze_gpt2_layers(self, num_layers=1):
        self.gpt2_model.train()
        total_layers = len(self.gpt2_model.h)
        
        for i, layer in enumerate(self.gpt2_model.h):
            if i >= total_layers - num_layers:
                for p in layer.parameters():
                    p.requires_grad = True
                    p.data = p.data.float()
        
        for p in self.gpt2_model.ln_f.parameters():
            p.requires_grad = True
            p.data = p.data.float()
        
        self.fine_tuning_mode = True
        print(f"Unfrozen last {num_layers} GPT-2 layers")
    
    def encode_image(self, images):
        if self.fine_tuning_mode:
            images = images.float()
            img_features = self.clip_model.encode_image(images)
        else:
            with torch.no_grad():
                img_features = self.clip_model.encode_image(images)
        
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        return img_features.float()
    
    def encode_question(self, input_ids, attention_mask):
        if self.fine_tuning_mode:
            outputs = self.gpt2_model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                outputs = self.gpt2_model(input_ids=input_ids, attention_mask=attention_mask)
        
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        masked = last_hidden * mask
        sum_hidden = masked.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-6)
        text_features = sum_hidden / lengths
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.float()
    
    def fuse_features(self, img_features, q_features):
        x = torch.cat([img_features, q_features], dim=-1)
        gate = torch.sigmoid(self.gate_layer(x))
        fused = gate * img_features + (1-gate) * q_features
        
        fused = self.fusion(torch.cat([fused, x], dim=-1))
        return fused
    
    def forward(self, images, questions, answer_input_ids=None):
        img_features = self.encode_image(images)
        img_features = self.img_proj(img_features).float()
        
        q_features = self.encode_question(questions["input_ids"], questions["attention_mask"])
        q_features = self.q_proj(q_features).float()
        
        batch_size = img_features.size(0)
        context = self.fuse_features(img_features, q_features)
        
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, 
                           device=self.device, dtype=torch.float)
        
        if answer_input_ids is not None:
            logits, _ = self.decoder(answer_input_ids, context, hidden)
            return logits
        else:
            generated = torch.full((batch_size, self.answer_max_len), self.pad_token_id,
                                 dtype=torch.long, device=self.device)
            generated[:, 0] = self.bos_token_id
            
            for t in range(1, self.answer_max_len):
                current_input = generated[:, t-1]
                logits, hidden = self.decoder(current_input, context, hidden)
                next_tokens = logits.squeeze(1).argmax(dim=-1)
                generated[:, t] = next_tokens
                
                if (next_tokens == self.eos_token_id).all():
                    break
            
            return generated
    def generate_with_beam_search(self, images, questions, beam_width=5):
        batch_size = images.size(0)
        all_results = []
        
        for b in range(batch_size):
            img = images[b:b+1]
            q_ids = questions["input_ids"][b:b+1]
            q_mask = questions["attention_mask"][b:b+1]
            
            img_features = self.encode_image(img)
            img_features = self.img_proj(img_features).float()
            
            q_features = self.encode_question(q_ids, q_mask)
            q_features = self.q_proj(q_features).float()
            
            context = self.fuse_features(img_features, q_features)
            
            initial_hidden = torch.zeros(self.num_layers, 1, self.hidden_size, 
                                         device=self.device, dtype=torch.float)
            beams = [(
                torch.full((1, 1), self.bos_token_id, dtype=torch.long, device=self.device),
                0.0,
                initial_hidden
            )]
            
            completed_beams = []
            
            for t in range(1, self.answer_max_len):
                candidates = []
                
                for seq, score, hidden in beams:
                    if seq[0, -1].item() == self.eos_token_id:
                        completed_beams.append((seq, score))
                        continue
                    
                    current_input = seq[:, -1]
                    logits, new_hidden = self.decoder(current_input, context, hidden)
                    
                    log_probs = torch.log_softmax(logits.squeeze(1), dim=-1)
                    top_log_probs, top_indices = torch.topk(log_probs[0], beam_width)
                    
                    for i in range(beam_width):
                        next_token = top_indices[i].unsqueeze(0).unsqueeze(0)
                        new_seq = torch.cat([seq, next_token], dim=1)
                        new_score = score + top_log_probs[i].item()
                        candidates.append((new_seq, new_score, new_hidden))
                
                if len(candidates) == 0:
                    break
                
                beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            
            all_beams = completed_beams + [(seq, score) for seq, score, _ in beams]
            
            if len(all_beams) == 0:
                result = torch.full((1, self.answer_max_len), self.pad_token_id,
                                dtype=torch.long, device=self.device)
            else:
                best_beam = max(all_beams, key=lambda x: x[1] / (x[0].size(1) ** 0.7))
                result = torch.full((1, self.answer_max_len), self.pad_token_id,
                                   dtype=torch.long, device=self.device)
                seq_len = min(best_beam[0].size(1), self.answer_max_len)
                result[:, :seq_len] = best_beam[0][:, :seq_len]
            
            all_results.append(result)
        
        return torch.cat(all_results, dim=0)
if __name__ == "__main__":
    device = "cuda"
    model = VQAModel(device=device).to(device)
    model.eval()
    
    fake_image = torch.randn(1, 3, 224, 224).to(device)
    
    fake_question_ids = torch.tensor([[1, 10, 20, 30, 2, 0, 0]]).to(device)
    fake_question_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]]).to(device)
    question_batch = {
        "input_ids": fake_question_ids,
        "attention_mask": fake_question_mask
    }
    
    output = model(fake_image, question_batch)
    print(output)