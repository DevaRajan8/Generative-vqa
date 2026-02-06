import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from model import VQAModel
from model_spatial import VQAModelWithSpatialAdapter
from train import AugmentedVQADataset, Vocab, save_checkpoint, plot_losses
import math
def filter_spatial_questions(df):
    """
    Filter dataset for spatial/directional questions.
    Returns both spatial subset and general subset for mixed training.
    """
    spatial_keywords = [
        # Directional
        'right', 'left', 'above', 'below', 'top', 'bottom',
        # Positional
        'front', 'behind', 'next to', 'beside', 'near', 'between',
        'in front', 'in back', 'across from', 'opposite',
        # Relational
        'closest', 'farthest', 'nearest', 'furthest',
        # Spatial queries
        'where is', 'which side', 'what side', 'what direction',
        'on the left', 'on the right', 'at the top', 'at the bottom'
    ]
    
    # Create regex pattern for case-insensitive matching
    pattern = '|'.join(spatial_keywords)
    spatial_mask = df['question'].str.lower().str.contains(pattern, na=False, regex=True)
    
    spatial_df = df[spatial_mask].copy()
    general_df = df[~spatial_mask].copy()
    
    print(f"\n📊 Dataset Filtering Results:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Spatial samples: {len(spatial_df):,} ({len(spatial_df)/len(df)*100:.1f}%)")
    print(f"  General samples: {len(general_df):,} ({len(general_df)/len(df)*100:.1f}%)")
    
    # Show sample spatial questions
    if len(spatial_df) > 0:
        print(f"\n📝 Sample Spatial Questions:")
        for i, row in spatial_df.sample(min(5, len(spatial_df))).iterrows():
            print(f"  Q: {row['question']}")
            print(f"  A: {row['answer']}\n")
    
    return spatial_df, general_df
def create_mixed_dataset(spatial_df, general_df, spatial_ratio=0.85, min_spatial_samples=1000):
    """
    Create mixed dataset with specified ratio of spatial to general questions.
    Increased default to 85% spatial for better spatial learning.
    """
    if len(spatial_df) < min_spatial_samples:
        print(f"\n⚠️  WARNING: Only {len(spatial_df)} spatial samples found!")
        print(f"  Recommended minimum: {min_spatial_samples}")
        print(f"  Mixing with general data to prevent catastrophic forgetting...")
        
        # Use all spatial samples
        num_spatial = len(spatial_df)
        # Calculate general samples to maintain ratio
        num_general = int(num_spatial * (1 - spatial_ratio) / spatial_ratio)
        num_general = min(num_general, len(general_df))
    else:
        # Calculate samples based on ratio
        num_spatial = len(spatial_df)
        num_general = int(num_spatial * (1 - spatial_ratio) / spatial_ratio)
        num_general = min(num_general, len(general_df))
    
    # Sample general data
    general_sample = general_df.sample(n=num_general, random_state=42)
    
    # Combine and shuffle
    mixed_df = pd.concat([spatial_df, general_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n🔀 Mixed Dataset Created:")
    print(f"  Spatial: {num_spatial:,} ({num_spatial/len(mixed_df)*100:.1f}%)")
    print(f"  General: {num_general:,} ({num_general/len(mixed_df)*100:.1f}%)")
    print(f"  Total: {len(mixed_df):,}")
    
    return mixed_df
def unfreeze_clip_layers(model, num_layers=4):
    """
    Unfreeze last N layers of CLIP for spatial feature learning.
    """
    total_blocks = len(model.clip_model.visual.transformer.resblocks)
    
    for i, block in enumerate(model.clip_model.visual.transformer.resblocks):
        if i >= total_blocks - num_layers:
            for p in block.parameters():
                p.requires_grad = True
    
    # Unfreeze projection and layer norm
    if hasattr(model.clip_model.visual, "proj") and model.clip_model.visual.proj is not None:
        if isinstance(model.clip_model.visual.proj, torch.nn.Parameter):
            model.clip_model.visual.proj.requires_grad = True
        else:
            for p in model.clip_model.visual.proj.parameters():
                p.requires_grad = True
    
    if hasattr(model.clip_model.visual, "ln_post"):
        for p in model.clip_model.visual.ln_post.parameters():
            p.requires_grad = True
    
    print(f"  ✓ Unfroze last {num_layers} CLIP layers")
def freeze_base_model(model, unfreeze_clip_layers_count=4):
    """
    Freeze most of the model, unfreeze spatial adapter and last CLIP layers.
    """
    # Freeze all CLIP first
    for param in model.clip_model.parameters():
        param.requires_grad = False
    
    # Unfreeze last N CLIP layers for spatial learning
    unfreeze_clip_layers(model, num_layers=unfreeze_clip_layers_count)
    
    # Freeze GPT-2
    for param in model.gpt2_model.parameters():
        param.requires_grad = False
    
    # Freeze base decoder
    for param in model.decoder.parameters():
        param.requires_grad = False
    
    # Unfreeze spatial adapter
    for param in model.spatial_adapter.parameters():
        param.requires_grad = True
    
    # Unfreeze spatial projection and fusion layers
    for param in model.spatial_context_proj.parameters():
        param.requires_grad = True
    for param in model.q_proj.parameters():
        param.requires_grad = True
    for param in model.spatial_fusion.parameters():
        param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n🔒 Model Freezing Applied:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    
    return model
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-7):
    """
    Create learning rate scheduler with warmup and cosine decay.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
def create_optimizer_with_differential_lr(model, base_lr=5e-5):
    """
    Create optimizer with differential learning rates for different components.
    """
    clip_params = []
    spatial_adapter_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'clip_model' in name:
                clip_params.append(param)
            elif 'spatial_adapter' in name:
                spatial_adapter_params.append(param)
            else:
                other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': clip_params, 'lr': base_lr * 0.1},  # 10x lower for CLIP
        {'params': spatial_adapter_params, 'lr': base_lr},  # Full LR for adapter
        {'params': other_params, 'lr': base_lr * 0.5}  # 2x lower for projections
    ], weight_decay=1e-4)
    
    print(f"\n⚙️  Optimizer Configuration:")
    print(f"  CLIP params: {len(clip_params):,} (LR: {base_lr * 0.1:.2e})")
    print(f"  Spatial adapter params: {len(spatial_adapter_params):,} (LR: {base_lr:.2e})")
    print(f"  Other params: {len(other_params):,} (LR: {base_lr * 0.5:.2e})")
    
    return optimizer
def train_one_epoch(model, dataloader, optimizer, device, vocab, scaler):
    """Training loop for one epoch"""
    model.train()
    total_loss = 0.0
    total_token_acc = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_token_id, label_smoothing=0.1)
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        images = batch['image'].to(device)
        questions = {
            'input_ids': batch['question_ids'].to(device),
            'attention_mask': batch['question_mask'].to(device)
        }
        answers = batch['answer_ids'].to(device)
        
        with torch.amp.autocast(device):
            logits = model(images, questions, answer_input_ids=answers)
            
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_answers = answers[:, 1:].contiguous()
            
            loss = criterion(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_answers.view(-1)
            )
            
            # Calculate token accuracy
            predicted_tokens = shifted_logits.argmax(dim=-1)
            correct = (predicted_tokens == shifted_answers).float()
            mask = (shifted_answers != vocab.pad_token_id).float()
            token_acc = (correct * mask).sum() / mask.sum()
            total_token_acc += token_acc.item()
        
        if torch.isnan(loss):
            print("⚠️  NaN loss detected, skipping batch.")
            continue
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_token_acc = total_token_acc / len(dataloader)
    return avg_loss, avg_token_acc
def validate_one_epoch(model, dataloader, device, vocab):
    """Validation loop for one epoch"""
    model.eval()
    total_loss = 0.0
    total_token_acc = 0.0
    exact_matches = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_token_id)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch['image'].to(device)
            questions = {
                'input_ids': batch['question_ids'].to(device),
                'attention_mask': batch['question_mask'].to(device)
            }
            answers = batch['answer_ids'].to(device)
            
            with torch.amp.autocast(device):
                logits = model(images, questions, answer_input_ids=answers)
                
                shifted_logits = logits[:, :-1, :].contiguous()
                shifted_answers = answers[:, 1:].contiguous()
                
                loss = criterion(
                    shifted_logits.view(-1, shifted_logits.size(-1)),
                    shifted_answers.view(-1)
                )
                
                # Token accuracy
                predicted_tokens = shifted_logits.argmax(dim=-1)
                correct = (predicted_tokens == shifted_answers).float()
                mask = (shifted_answers != vocab.pad_token_id).float()
                token_acc = (correct * mask).sum() / mask.sum()
                total_token_acc += token_acc.item()
            
            total_loss += loss.item()
            
            # Generate answers for exact match
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
    print("=" * 80)
    print("🚀 VQA SPATIAL ADAPTER FINE-TUNING V2 (ENHANCED)")
    print("=" * 80)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Paths
    DATA_DIR = r"./gen_vqa_v2"
    CSV_PATH = os.path.join(DATA_DIR, "metadata.csv")
    PRETRAINED_CHECKPOINT = "./output2/continued_training/vqa_checkpoint.pt"
    OUTPUT_DIR = "./output2/spatial_adapter_v2_2"
    FINE_TUNED_CHECKPOINT = os.path.join(OUTPUT_DIR, "vqa_spatial_checkpoint.pt")
    LOG_CSV = os.path.join(OUTPUT_DIR, "train_log.csv")
    LOSS_GRAPH_PATH = os.path.join(OUTPUT_DIR, "loss_plot.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Hyperparameters (ENHANCED)
    batch_size = 64
    base_learning_rate = 5e-5  # Increased from 3e-5
    num_epochs = 100  # Increased from 20
    patience = 15  # Increased from 6
    warmup_epochs = 3  # NEW: Warmup period
    spatial_ratio = 0.85  # Increased from 0.70
    clip_layers_to_unfreeze = 6  # NEW: Unfreeze CLIP layers
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n⚙️  Enhanced Configuration:")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Base learning rate: {base_learning_rate:.2e}")
    print(f"  Max epochs: {num_epochs} (increased from 20)")
    print(f"  Warmup epochs: {warmup_epochs}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Spatial ratio: {spatial_ratio:.0%} (increased from 70%)")
    print(f"  CLIP layers to unfreeze: {clip_layers_to_unfreeze}")
    
    # Load dataset
    print(f"\n📂 Loading dataset from: {CSV_PATH}")
    metadata = pd.read_csv(CSV_PATH)
    
    # Filter spatial questions
    spatial_df, general_df = filter_spatial_questions(metadata)
    
    # Create mixed dataset with higher spatial ratio
    mixed_data = create_mixed_dataset(spatial_df, general_df, spatial_ratio=spatial_ratio)
    
    # Load pretrained checkpoint
    print(f"\n📥 Loading pretrained model from: {PRETRAINED_CHECKPOINT}")
    checkpoint = torch.load(PRETRAINED_CHECKPOINT, map_location=device)
    
    # Load vocabulary
    vocab = Vocab()
    vocab.vocab = checkpoint['vocab']
    vocab.vocab_size = len(checkpoint['vocab'])
    vocab.word2idx = checkpoint['word2idx']
    vocab.idx2word = checkpoint['idx2word']
    vocab.pad_token_id = checkpoint['pad_token_id']
    vocab.bos_token_id = checkpoint['bos_token_id']
    vocab.eos_token_id = checkpoint['eos_token_id']
    vocab.unk_token_id = checkpoint['unk_token_id']
    
    print(f"  Vocabulary size: {len(vocab.vocab):,}")
    
    # Setup tokenizer FIRST (before creating model)
    question_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    if question_tokenizer.pad_token is None:
        question_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    # Create base model
    base_model = VQAModel(
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
    
    # Resize GPT-2 embeddings to match tokenizer (before loading weights)
    base_model.gpt2_model.resize_token_embeddings(len(question_tokenizer))
    
    # Load pretrained weights
    base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("  ✓ Pretrained weights loaded")
    
    # Create spatial adapter model
    print(f"\n🔧 Creating VQA model with spatial adapter...")
    model = VQAModelWithSpatialAdapter(
        base_model=base_model,
        hidden_size=512,
        num_heads=8,
        dropout=0.3
    ).to(device)
    
    # Freeze base model, unfreeze spatial adapter AND last CLIP layers
    model = freeze_base_model(model, unfreeze_clip_layers_count=clip_layers_to_unfreeze)
    
    # Split data
    train_df, test_df = train_test_split(mixed_data, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
    
    print(f"\n📊 Data Split:")
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Validation: {len(val_df):,} samples")
    print(f"  Test: {len(test_df):,} samples")
    
    # Create datasets (with safe augmentation)
    from torchvision import transforms
    
    # Safe augmentation that preserves spatial relationships
    safe_augmentation = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(5),  # Small rotation only
    ])
    
    train_dataset = AugmentedVQADataset(
        train_df, DATA_DIR, question_tokenizer, vocab,
        clip_processor=model.clip_preprocess,
        augment=False,  # Keep disabled for now
        question_max_len=20,
        answer_max_len=12
    )
    val_dataset = AugmentedVQADataset(
        val_df, DATA_DIR, question_tokenizer, vocab,
        clip_processor=model.clip_preprocess,
        augment=False,
        question_max_len=20,
        answer_max_len=12
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Setup optimizer with differential learning rates
    optimizer = create_optimizer_with_differential_lr(model, base_lr=base_learning_rate)
    
    # Setup cosine scheduler with warmup
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = len(train_loader) * warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    print(f"\n📈 Learning Rate Schedule:")
    print(f"  Warmup steps: {num_warmup_steps:,} ({warmup_epochs} epochs)")
    print(f"  Total steps: {num_training_steps:,}")
    print(f"  Schedule: Linear warmup → Cosine decay")
    
    scaler = torch.amp.GradScaler(device)
    
    # Training loop
    print("\n" + "=" * 80)
    print("🎯 STARTING ENHANCED SPATIAL ADAPTER FINE-TUNING")
    print("=" * 80)
    
    best_val_exact_match = 0.0
    best_val_loss = np.inf
    counter = 0
    logs = []
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_token_acc = train_one_epoch(model, train_loader, optimizer, device, vocab, scaler)
        
        # Validate
        val_loss, val_token_acc, val_exact_match = validate_one_epoch(model, val_loader, device, vocab)
        
        # Step scheduler (per epoch)
        current_lr = optimizer.param_groups[1]['lr']  # Spatial adapter LR
        
        # Print metrics
        print(f"\n📈 Metrics:")
        print(f"  Train Loss: {train_loss:.4f} | Train Token Acc: {train_token_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Token Acc: {val_token_acc:.4f}")
        print(f"  Val Exact Match: {val_exact_match:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if val_exact_match > best_val_exact_match:
            best_val_exact_match = val_exact_match
            save_checkpoint(model, optimizer, epoch, vocab, FINE_TUNED_CHECKPOINT)
            print(f"  ✅ New best model saved! (Exact Match: {val_exact_match:.4f})")
            counter = 0
        else:
            counter += 1
            print(f"  ⏳ No improvement for {counter} epoch(s)")
        
        # Early stopping
        if counter >= patience:
            print(f"\n⏹️  Early stopping triggered after {patience} epochs without improvement")
            break
        
        # Log metrics
        logs.append([
            epoch + 1,
            train_loss,
            train_token_acc,
            val_loss,
            val_token_acc,
            val_exact_match,
            current_lr
        ])
        
        # Step scheduler per batch for next epoch
        for _ in range(len(train_loader)):
            scheduler.step()
    
    # Save logs and plots
    log_df = pd.DataFrame(
        logs,
        columns=["epoch", "train_loss", "train_token_acc", "val_loss", "val_token_acc", "val_exact_match", "lr"]
    )
    log_df.to_csv(LOG_CSV, index=False)
    plot_losses([x[1] for x in logs], [x[3] for x in logs], save_path=LOSS_GRAPH_PATH)
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ ENHANCED FINE-TUNING COMPLETE")
    print("=" * 80)
    print(f"\n📊 Final Results:")
    print(f"  Best Exact Match: {best_val_exact_match:.4f}")
    print(f"  Total Epochs: {len(logs)}")
    print(f"  Improvement from v1: {best_val_exact_match - 0.2037:.4f} ({(best_val_exact_match - 0.2037) / 0.2037 * 100:+.1f}%)")
    print(f"\n💾 Outputs:")
    print(f"  Model: {FINE_TUNED_CHECKPOINT}")
    print(f"  Logs: {LOG_CSV}")
    print(f"  Plot: {LOSS_GRAPH_PATH}")
    print("\n🎉 Ready to test on spatial questions!")
if __name__ == "__main__":
    main()