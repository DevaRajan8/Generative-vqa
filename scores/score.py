import os
import torch
import pandas as pd
from PIL import Image
from transformers import GPT2Tokenizer
from model import VQAModel
from model_spatial import VQAModelWithSpatialAdapter
from train import Vocab
from tqdm import tqdm
import numpy as np
# Install python-Levenshtein if not already installed
try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    print("Installing python-Levenshtein...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'python-Levenshtein'])
    from Levenshtein import distance as levenshtein_distance
# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_TYPE = "feature"  # "spatial" or "feature"
SPATIAL_CHECKPOINT = "./output2/spatial_adapter_v2_2/vqa_spatial_checkpoint.pt"
FEATURE_CHECKPOINT = "./output2/feature_extraction/vqa_checkpoint.pt"
CSV_PATH = "./gen_vqa_v2/metadata.csv"
IMG_DIR = "./gen_vqa_v2"
MAX_SAMPLES = None  # None for all samples
# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================
def load_spatial_model(checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    vocab = Vocab()
    vocab.vocab = checkpoint['vocab']
    vocab.vocab_size = len(checkpoint['vocab'])
    vocab.word2idx = checkpoint['word2idx']
    vocab.idx2word = checkpoint['idx2word']
    vocab.pad_token_id = checkpoint['pad_token_id']
    vocab.bos_token_id = checkpoint['bos_token_id']
    vocab.eos_token_id = checkpoint['eos_token_id']
    vocab.unk_token_id = checkpoint['unk_token_id']
    
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
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
    
    base_model.gpt2_model.resize_token_embeddings(len(tokenizer))
    
    model = VQAModelWithSpatialAdapter(
        base_model=base_model,
        hidden_size=512,
        num_heads=8,
        dropout=0.3
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    return model, vocab, tokenizer
def load_feature_model(checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    vocab = Vocab()
    vocab.vocab = checkpoint['vocab']
    vocab.vocab_size = len(checkpoint['vocab'])
    vocab.word2idx = checkpoint['word2idx']
    vocab.idx2word = checkpoint['idx2word']
    vocab.pad_token_id = checkpoint['pad_token_id']
    vocab.bos_token_id = checkpoint['bos_token_id']
    vocab.eos_token_id = checkpoint['eos_token_id']
    vocab.unk_token_id = checkpoint['unk_token_id']
    
    model = VQAModel(
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
    
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.gpt2_model.resize_token_embeddings(len(tokenizer))
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    return model, vocab, tokenizer
def generate_answer(model, vocab, tokenizer, image_path, question, device='cuda'):
    image = Image.open(image_path).convert('RGB')
    image = model.clip_preprocess(image).unsqueeze(0).to(device)
    
    question_tokens = tokenizer(
        question,
        padding='max_length',
        truncation=True,
        max_length=model.question_max_len,
        return_tensors='pt'
    )
    
    questions = {
        'input_ids': question_tokens['input_ids'].to(device),
        'attention_mask': question_tokens['attention_mask'].to(device)
    }
    
    with torch.no_grad():
        if hasattr(model, 'generate_with_beam_search'):
            generated = model.generate_with_beam_search(image, questions, beam_width=5)
        else:
            logits = model(image, questions)
            generated = logits.argmax(dim=-1)
    
    return vocab.decoder(generated[0].cpu().numpy())
# ============================================================================
# EVALUATION METRICS
# ============================================================================
def exact_match_accuracy(predictions, ground_truths):
    """
    Calculate exact match accuracy (case-insensitive, stripped).
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
    
    Returns:
        accuracy: Percentage of exact matches
    """
    matches = sum(1 for pred, gt in zip(predictions, ground_truths) 
                  if pred.strip().lower() == gt.strip().lower())
    accuracy = (matches / len(predictions)) * 100 if predictions else 0
    return accuracy, matches
def vqa_accuracy(predictions, ground_truths_list):
    """
    VQA official metric: min(#humans_said_answer / 3, 1)
    
    Note: This assumes ground_truths_list is a list of lists,
    where each inner list contains multiple human annotations.
    If you only have one annotation per question, this reduces to exact match.
    
    Args:
        predictions: List of predicted answers
        ground_truths_list: List of lists of ground truth answers
    
    Returns:
        vqa_score: VQA accuracy score (0-100)
    """
    if not isinstance(ground_truths_list[0], list):
        # Convert single annotations to list format
        ground_truths_list = [[gt] for gt in ground_truths_list]
    
    scores = []
    for pred, gt_list in zip(predictions, ground_truths_list):
        pred_clean = pred.strip().lower()
        matches = sum(1 for gt in gt_list if pred_clean == gt.strip().lower())
        score = min(matches / 3.0, 1.0)  # VQA official formula
        scores.append(score)
    
    vqa_score = (sum(scores) / len(scores)) * 100 if scores else 0
    return vqa_score
def calculate_anls(prediction, ground_truth, threshold=0.5):
    """
    Calculate ANLS (Average Normalized Levenshtein Similarity) for a single pair.
    
    Args:
        prediction: Predicted answer string
        ground_truth: Ground truth answer string
        threshold: Minimum similarity threshold (default: 0.5)
    
    Returns:
        anls_score: ANLS score (0-1)
    """
    pred_clean = prediction.strip().lower()
    gt_clean = ground_truth.strip().lower()
    
    if len(gt_clean) == 0:
        return 1.0 if len(pred_clean) == 0 else 0.0
    
    # Calculate Levenshtein distance
    dist = levenshtein_distance(pred_clean, gt_clean)
    
    # Normalize by length of ground truth
    max_len = max(len(pred_clean), len(gt_clean))
    
    if max_len == 0:
        return 1.0
    
    # Calculate normalized similarity
    similarity = 1 - (dist / max_len)
    
    # Apply threshold
    anls = similarity if similarity >= threshold else 0.0
    
    return anls
def average_anls(predictions, ground_truths, threshold=0.5):
    """
    Calculate average ANLS across all predictions.
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        threshold: Minimum similarity threshold
    
    Returns:
        avg_anls: Average ANLS score (0-100)
    """
    anls_scores = []
    for pred, gt in zip(predictions, ground_truths):
        score = calculate_anls(pred, gt, threshold)
        anls_scores.append(score)
    
    avg_anls = (sum(anls_scores) / len(anls_scores)) * 100 if anls_scores else 0
    return avg_anls, anls_scores
# ============================================================================
# MAIN EVALUATION
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("VQA EVALUATION: ACCURACY + ANLS")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"Model: {MODEL_TYPE.upper()}\n")
    
    # Load model
    if MODEL_TYPE == "spatial":
        model, vocab, tokenizer = load_spatial_model(SPATIAL_CHECKPOINT, device)
    else:
        model, vocab, tokenizer = load_feature_model(FEATURE_CHECKPOINT, device)
    
    print("✓ Model loaded!\n")
    
    # Load dataset
    df = pd.read_csv(CSV_PATH)
    if MAX_SAMPLES:
        df = df.head(MAX_SAMPLES)
    
    print(f"Evaluating {len(df)} samples\n")
    
    # Generate predictions
    print("Generating predictions...")
    predictions = []
    ground_truths = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(IMG_DIR, row['image_path'])
        
        if not os.path.exists(image_path):
            continue
        
        try:
            prediction = generate_answer(model, vocab, tokenizer, 
                                        image_path, row['question'], device)
            ground_truth = row['answer']
            
            predictions.append(prediction)
            ground_truths.append(ground_truth)
            
        except Exception as e:
            continue
    
    print(f"\n✓ Generated {len(predictions)} predictions\n")
    
    # Calculate metrics
    print("Calculating metrics...\n")
    
    # 1. Exact Match Accuracy
    exact_acc, exact_matches = exact_match_accuracy(predictions, ground_truths)
    
    # 2. VQA Accuracy (with single annotation, same as exact match)
    vqa_acc = vqa_accuracy(predictions, ground_truths)
    
    # 3. ANLS (Average Normalized Levenshtein Similarity)
    anls_score, anls_scores = average_anls(predictions, ground_truths, threshold=0.5)
    
    # Results
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    print(f"\n📊 Accuracy Metrics:")
    print(f"  Exact Match Accuracy:  {exact_acc:.2f}% ({exact_matches}/{len(predictions)})")
    print(f"  VQA Accuracy:          {vqa_acc:.2f}%")
    
    print(f"\n📊 ANLS Metrics:")
    print(f"  Average ANLS (τ=0.5):  {anls_score:.2f}%")
    print(f"  ANLS Std Dev:          {np.std(anls_scores)*100:.2f}%")
    
    # Additional statistics
    print(f"\n📊 Additional Statistics:")
    print(f"  Total samples:         {len(predictions)}")
    print(f"  Avg prediction length: {np.mean([len(p.split()) for p in predictions]):.2f} words")
    print(f"  Avg GT length:         {np.mean([len(gt.split()) for gt in ground_truths]):.2f} words")
    
    # Show some examples
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)
    
    # Sort by ANLS score
    sorted_indices = np.argsort(anls_scores)
    
    print("\n🏆 Best Predictions (Highest ANLS):")
    print("-" * 80)
    for i in sorted_indices[-5:][::-1]:
        print(f"\nGround Truth: {ground_truths[i]}")
        print(f"Prediction:   {predictions[i]}")
        print(f"ANLS:         {anls_scores[i]:.4f}")
        print(f"Exact Match:  {'✓' if predictions[i].strip().lower() == ground_truths[i].strip().lower() else '✗'}")
    
    print("\n" + "=" * 80)
    print("⚠️  Worst Predictions (Lowest ANLS):")
    print("-" * 80)
    for i in sorted_indices[:5]:
        print(f"\nGround Truth: {ground_truths[i]}")
        print(f"Prediction:   {predictions[i]}")
        print(f"ANLS:         {anls_scores[i]:.4f}")
        print(f"Exact Match:  {'✓' if predictions[i].strip().lower() == ground_truths[i].strip().lower() else '✗'}")
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)

    with open(f"{MODEL_TYPE}.txt", "w", encoding="utf-8") as f:

        f.write("=" * 80 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n")
    
        f.write("\n📊 Accuracy Metrics:\n")
        f.write(f"  Exact Match Accuracy:  {exact_acc:.2f}% ({exact_matches}/{len(predictions)})\n")
        f.write(f"  VQA Accuracy:          {vqa_acc:.2f}%\n")
    
        f.write("\n📊 ANLS Metrics:\n")
        f.write(f"  Average ANLS (τ=0.5):  {anls_score:.2f}%\n")
        f.write(f"  ANLS Std Dev:          {np.std(anls_scores)*100:.2f}%\n")
    
        f.write("\n📊 Additional Statistics:\n")
        f.write(f"  Total samples:         {len(predictions)}\n")
        f.write(f"  Avg prediction length: {np.mean([len(p.split()) for p in predictions]):.2f} words\n")
        f.write(f"  Avg GT length:         {np.mean([len(gt.split()) for gt in ground_truths]):.2f} words\n")
    
        f.write("\n" + "=" * 80 + "\n")
        f.write("SAMPLE PREDICTIONS\n")
        f.write("=" * 80 + "\n")
    
        sorted_indices = np.argsort(anls_scores)
    
        f.write("\n🏆 Best Predictions (Highest ANLS):\n")
        f.write("-" * 80 + "\n")
        for i in sorted_indices[-5:][::-1]:
            f.write(f"\nGround Truth: {ground_truths[i]}\n")
            f.write(f"Prediction:   {predictions[i]}\n")
            f.write(f"ANLS:         {anls_scores[i]:.4f}\n")
            f.write(
                f"Exact Match:  {'✓' if predictions[i].strip().lower() == ground_truths[i].strip().lower() else '✗'}\n"
            )
    
        f.write("\n" + "=" * 80 + "\n")
        f.write("⚠️  Worst Predictions (Lowest ANLS):\n")
        f.write("-" * 80 + "\n")
        for i in sorted_indices[:5]:
            f.write(f"\nGround Truth: {ground_truths[i]}\n")
            f.write(f"Prediction:   {predictions[i]}\n")
            f.write(f"ANLS:         {anls_scores[i]:.4f}\n")
            f.write(
                f"Exact Match:  {'✓' if predictions[i].strip().lower() == ground_truths[i].strip().lower() else '✗'}\n"
            )
    

    
    # Save results to CSV
    results_df = pd.DataFrame({
        'prediction': predictions,
        'ground_truth': ground_truths,
        'anls_score': anls_scores,
        'exact_match': [pred.strip().lower() == gt.strip().lower() 
                       for pred, gt in zip(predictions, ground_truths)]
    })
    
    output_file = f"vqa_evaluation_{MODEL_TYPE}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")