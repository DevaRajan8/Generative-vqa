import torch
from model import JointEmbeddingVQA, EasyVQADataset
from main import process_data
from evaluate import evaluate_on_test_set
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    model_path = r'C:\Users\rdeva\Downloads\sem6\SynerVQA\preliminaries\joint-emb\best_vqa_model.pth'
    train_json = r'C:\Users\rdeva\Downloads\sem6\SynerVQA\easy_vqa_data\questions_answers.json'
    test_json = r'C:\Users\rdeva\Downloads\sem6\SynerVQA\preliminaries\joint-emb\testfiles\tquestions_answers.json'
    test_images = r'C:\Users\rdeva\Downloads\sem6\SynerVQA\preliminaries\joint-emb\testfiles\timages'
    
    
    print("Loading vocabulary...")
    train_data, _ = process_data(train_json)
    dummy_dataset = EasyVQADataset('images', train_data)
    vocab = dummy_dataset.vocab
    ans_vocab = dummy_dataset.ans_vocab
    
    
    print("Loading model...")
    model = JointEmbeddingVQA(len(vocab), len(ans_vocab))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    
    results, accuracy = evaluate_on_test_set(
        model, test_images, test_json, vocab, ans_vocab, device
    )
    
    print(f"\nTesting complete! Accuracy: {accuracy:.2f}%")
if __name__ == '__main__':
    main()