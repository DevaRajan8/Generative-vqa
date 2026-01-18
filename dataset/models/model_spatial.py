import torch
from torch import nn
import clip
from transformers import GPT2Model
import math
class SpatialAdapter(nn.Module):
    """
    Spatial Adapter with Multi-Head Cross-Attention for spatial reasoning.
    Processes CLIP patch features (14x14 grid) with question guidance.
    """
    def __init__(self, patch_dim=512, question_dim=512, hidden_dim=512, num_heads=8, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # 2D Positional encoding for spatial awareness (14x14 grid)
        self.register_buffer('pos_encoding_2d', self._create_2d_positional_encoding(14, 14, patch_dim))
        
        # Projection layers for patches and questions
        self.patch_proj = nn.Linear(patch_dim, hidden_dim)
        self.question_proj = nn.Linear(question_dim, hidden_dim)
        
        # Multi-head cross-attention: patches attend to question
        self.cross_attn_query = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn_key = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn_value = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn_out = nn.Linear(hidden_dim, hidden_dim)
        
        # Self-attention for spatial relationships between patches
        self.self_attn_query = nn.Linear(hidden_dim, hidden_dim)
        self.self_attn_key = nn.Linear(hidden_dim, hidden_dim)
        self.self_attn_value = nn.Linear(hidden_dim, hidden_dim)
        self.self_attn_out = nn.Linear(hidden_dim, hidden_dim)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def _create_2d_positional_encoding(self, height, width, dim):
        """Create 2D positional encoding for spatial grid"""
        pos_h = torch.arange(height).unsqueeze(1).repeat(1, width).flatten()
        pos_w = torch.arange(width).unsqueeze(0).repeat(height, 1).flatten()
        
        pe = torch.zeros(height * width, dim)
        
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        # Encode height (first half of dimensions)
        pe[:, 0:dim//2:2] = torch.sin(pos_h.unsqueeze(1) * div_term[:dim//4])
        pe[:, 1:dim//2:2] = torch.cos(pos_h.unsqueeze(1) * div_term[:dim//4])
        
        # Encode width (second half of dimensions)
        pe[:, dim//2::2] = torch.sin(pos_w.unsqueeze(1) * div_term[:dim//4])
        pe[:, dim//2+1::2] = torch.cos(pos_w.unsqueeze(1) * div_term[:dim//4])
        
        return pe.unsqueeze(0)  # [1, H*W, dim]
    
    def _multi_head_attention(self, query, key, value, num_heads):
        """Generic multi-head attention implementation"""
        batch_size = query.size(0)
        
        # Linear projections in batch from hidden_dim => num_heads x head_dim
        Q = query.view(batch_size, -1, num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        K = key.view(batch_size, -1, num_heads, self.head_dim).transpose(1, 2)
        V = value.view(batch_size, -1, num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # [B, H, N, D]
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        return context, attn_weights
    
    def forward(self, patch_features, question_features):
        """
        Args:
            patch_features: [batch_size, num_patches, patch_dim] - CLIP patch features
            question_features: [batch_size, question_dim] - Question encoding
        Returns:
            spatial_context: [batch_size, hidden_dim] - Spatially-aware context
        """
        batch_size, num_patches, _ = patch_features.shape
        
        # Add 2D positional encoding to patches
        patch_features = patch_features + self.pos_encoding_2d[:, :num_patches, :].to(patch_features.device)
        
        # Project patches and question to hidden_dim
        patches = self.patch_proj(patch_features)  # [B, N, H]
        question = self.question_proj(question_features.unsqueeze(1))  # [B, 1, H]
        
        # 1. Cross-attention: Patches attend to question (question-guided spatial focus)
        Q_cross = self.cross_attn_query(patches)
        K_cross = self.cross_attn_key(question)
        V_cross = self.cross_attn_value(question)
        
        cross_context, _ = self._multi_head_attention(Q_cross, K_cross, V_cross, self.num_heads)
        cross_out = self.cross_attn_out(cross_context)
        patches = self.ln1(patches + self.dropout(cross_out))
        
        # 2. Self-attention: Patches attend to each other (spatial relationships)
        Q_self = self.self_attn_query(patches)
        K_self = self.self_attn_key(patches)
        V_self = self.self_attn_value(patches)
        
        self_context, _ = self._multi_head_attention(Q_self, K_self, V_self, self.num_heads)
        self_out = self.self_attn_out(self_context)
        patches = self.ln2(patches + self.dropout(self_out))
        
        # 3. Feedforward network
        ffn_out = self.ffn(patches)
        patches = self.ln3(patches + ffn_out)
        
        # 4. Pool patches to single spatial context vector (weighted average)
        # Use question to compute attention weights over patches
        attn_scores = torch.matmul(patches, question.transpose(1, 2))  # [B, N, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)
        spatial_context = (patches * attn_weights).sum(dim=1)  # [B, H]
        
        return spatial_context
class VQAModelWithSpatialAdapter(nn.Module):
    """
    Enhanced VQA Model with Spatial Adapter for spatial reasoning.
    Uses patch-based CLIP features instead of global encoding.
    """
    def __init__(
        self,
        base_model,
        hidden_size=512,
        num_heads=8,
        dropout=0.3
    ):
        super().__init__()
        
        # Copy attributes from base model
        self.device = base_model.device
        self.question_max_len = base_model.question_max_len
        self.answer_max_len = base_model.answer_max_len
        self.vocab_size = base_model.vocab_size
        self.hidden_size = hidden_size
        self.num_layers = base_model.num_layers
        self.fine_tuning_mode = base_model.fine_tuning_mode
        
        self.pad_token_id = base_model.pad_token_id
        self.bos_token_id = base_model.bos_token_id
        self.eos_token_id = base_model.eos_token_id
        self.unk_token_id = base_model.unk_token_id
        
        # Use base model's encoders (will be frozen during fine-tuning)
        self.clip_model = base_model.clip_model
        self.clip_preprocess = base_model.clip_preprocess
        self.gpt2_model = base_model.gpt2_model
        
        # Keep base model's decoder
        self.decoder = base_model.decoder
        
        # NEW: Spatial adapter for patch-based processing
        self.spatial_adapter = SpatialAdapter(
            patch_dim=512,  # CLIP ViT-B/16 patch dimension
            question_dim=768,  # GPT-2 dimension
            hidden_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Enhanced projection layers
        self.spatial_context_proj = nn.Linear(hidden_size, hidden_size)
        self.q_proj = nn.Linear(768, hidden_size)
        
        # Enhanced fusion layer for spatial context
        self.spatial_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
    def extract_clip_patch_features(self, images):
        """
        Extract patch features from CLIP instead of global features.
        Returns: [batch_size, num_patches, patch_dim]
        """
        # Convert images to CLIP's dtype (handles float16/float32 mismatch)
        clip_dtype = self.clip_model.visual.conv1.weight.dtype
        images = images.to(clip_dtype)
        
        if self.fine_tuning_mode:
            # Get patch features from CLIP vision transformer
            x = self.clip_model.visual.conv1(images)  # [B, 768, 14, 14] for ViT-B/16
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, 768, 196]
            x = x.permute(0, 2, 1)  # [B, 196, 768]
            
            # Add class token and positional encoding
            class_token = self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            )
            x = torch.cat([class_token, x], dim=1)  # [B, 197, 768]
            x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
            
            # Pass through transformer blocks
            x = self.clip_model.visual.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_model.visual.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            
            # Remove class token, keep only patches
            patch_features = x[:, 1:, :]  # [B, 196, 768]
            
            # Project to 512 dimensions
            if hasattr(self.clip_model.visual, 'proj') and self.clip_model.visual.proj is not None:
                if isinstance(self.clip_model.visual.proj, torch.nn.Parameter):
                    patch_features = patch_features @ self.clip_model.visual.proj
                else:
                    patch_features = self.clip_model.visual.proj(patch_features)
        else:
            with torch.no_grad():
                x = self.clip_model.visual.conv1(images)
                x = x.reshape(x.shape[0], x.shape[1], -1)
                x = x.permute(0, 2, 1)
                
                class_token = self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                )
                x = torch.cat([class_token, x], dim=1)
                x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
                
                x = self.clip_model.visual.ln_pre(x)
                x = x.permute(1, 0, 2)
                x = self.clip_model.visual.transformer(x)
                x = x.permute(1, 0, 2)
                
                patch_features = x[:, 1:, :]
                
                if hasattr(self.clip_model.visual, 'proj') and self.clip_model.visual.proj is not None:
                    if isinstance(self.clip_model.visual.proj, torch.nn.Parameter):
                        patch_features = patch_features @ self.clip_model.visual.proj
                    else:
                        patch_features = self.clip_model.visual.proj(patch_features)
        
        return patch_features.float()
    
    def encode_question(self, input_ids, attention_mask):
        """Encode question using GPT-2 (same as base model)"""
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
    
    def forward(self, images, questions, answer_input_ids=None):
        """
        Forward pass with spatial adapter.
        """
        # Extract patch features instead of global features
        patch_features = self.extract_clip_patch_features(images)  # [B, 196, 512]
        
        # Encode question
        q_features = self.encode_question(questions["input_ids"], questions["attention_mask"])  # [B, 768]
        
        # Apply spatial adapter: patches + question -> spatial context
        spatial_context = self.spatial_adapter(patch_features, q_features)  # [B, 512]
        spatial_context = self.spatial_context_proj(spatial_context)
        
        # Project question features
        q_projected = self.q_proj(q_features)  # [B, 512]
        
        # Fuse spatial context with question features
        fused = self.spatial_fusion(torch.cat([spatial_context, q_projected], dim=-1))  # [B, 512]
        
        batch_size = images.size(0)
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, 
                           device=self.device, dtype=torch.float)
        
        if answer_input_ids is not None:
            # Training mode: use teacher forcing
            logits, _ = self.decoder(answer_input_ids, fused, hidden)
            return logits
        else:
            # Inference mode: autoregressive generation
            generated = torch.full((batch_size, self.answer_max_len), self.pad_token_id,
                                 dtype=torch.long, device=self.device)
            generated[:, 0] = self.bos_token_id
            
            for t in range(1, self.answer_max_len):
                current_input = generated[:, t-1]
                logits, hidden = self.decoder(current_input, fused, hidden)
                next_tokens = logits.squeeze(1).argmax(dim=-1)
                generated[:, t] = next_tokens
                
                if (next_tokens == self.eos_token_id).all():
                    break
            
            return generated
    
    def generate_with_beam_search(self, images, questions, beam_width=5):
        """Beam search generation (same as base model but with spatial features)"""
        batch_size = images.size(0)
        all_results = []
        
        for b in range(batch_size):
            img = images[b:b+1]
            q_ids = questions["input_ids"][b:b+1]
            q_mask = questions["attention_mask"][b:b+1]
            
            # Extract spatial features
            patch_features = self.extract_clip_patch_features(img)
            q_features = self.encode_question(q_ids, q_mask)
            spatial_context = self.spatial_adapter(patch_features, q_features)
            spatial_context = self.spatial_context_proj(spatial_context)
            q_projected = self.q_proj(q_features)
            context = self.spatial_fusion(torch.cat([spatial_context, q_projected], dim=-1))
            
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
    # Test the spatial adapter architecture
    print("Testing Spatial Adapter Architecture...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create dummy base model components
    from model import VQAModel
    
    base_model = VQAModel(device=device).to(device)
    
    # Create spatial adapter model
    spatial_model = VQAModelWithSpatialAdapter(base_model).to(device)
    spatial_model.eval()
    
    # Test with dummy data
    fake_image = torch.randn(2, 3, 224, 224).to(device)
    fake_question_ids = torch.tensor([[1, 10, 20, 30, 2, 0, 0], [1, 15, 25, 35, 2, 0, 0]]).to(device)
    fake_question_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0]]).to(device)
    question_batch = {
        "input_ids": fake_question_ids,
        "attention_mask": fake_question_mask
    }
    
    print(f"\nInput shapes:")
    print(f"  Images: {fake_image.shape}")
    print(f"  Questions: {fake_question_ids.shape}")
    
    # Test patch extraction
    with torch.no_grad():
        patch_features = spatial_model.extract_clip_patch_features(fake_image)
        print(f"\nPatch features shape: {patch_features.shape}")
        print(f"  Expected: [2, 196, 512] (batch_size, num_patches, patch_dim)")
        
        # Test forward pass
        output = spatial_model(fake_image, question_batch)
        print(f"\nGenerated output shape: {output.shape}")
        print(f"  Expected: [2, {spatial_model.answer_max_len}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in spatial_model.parameters())
    spatial_adapter_params = sum(p.numel() for p in spatial_model.spatial_adapter.parameters())
    trainable_params = sum(p.numel() for p in spatial_model.parameters() if p.requires_grad)
    
    print(f"\nParameter counts:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Spatial adapter parameters: {spatial_adapter_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    print("\n✓ Spatial adapter architecture test passed!")