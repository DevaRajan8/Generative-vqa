import torch
from collections import defaultdict
def compute_per_class_accuracy(model, dataloader, dataset, device):

    model.eval()

    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for images, questions, answers in dataloader:
            images = images.to(device)
            questions = questions.to(device)
            answers = answers.to(device)
            
            logits = model(images, questions)
            _, predicted = logits.max(1)
            
            for pred, true in zip(predicted, answers):
                true_idx = true.item()
                class_total[true_idx] += 1
                if pred == true:
                    class_correct[true_idx] += 1
    

    idx_to_ans = {v: k for k, v in dataset.ans_vocab.items()}
    
    print("\nPer-Class Accuracy:")
    print("-" * 50)
    
    for ans_idx in sorted(class_total.keys()):
        answer_name = idx_to_ans[ans_idx]
        correct = class_correct[ans_idx]
        total = class_total[ans_idx]
        accuracy = 100.0 * correct / total if total > 0 else 0
        
        # Visual bar
        bar_length = int(accuracy / 5)  
        bar = '' * bar_length + '' * (20 - bar_length)
        
        print(f"{answer_name:12s} {bar} {accuracy:5.1f}% ({correct}/{total})")
    
    print("\n")
    
    return class_correct, class_total