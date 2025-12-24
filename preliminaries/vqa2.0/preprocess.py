import json
import os
from collections import Counter
def preprocess_vqa2(annotations_file, questions_file, output_file, images_dir):

    print("Loading VQA v2 data...")
    with open(annotations_file, 'r') as f:
        vqa_data = json.load(f)
    
    annotations = vqa_data['annotations']
    questions_data = vqa_data.get('questions', [])
    
    
    question_lookup = {}
    if questions_data:
        for q in questions_data:
            question_lookup[q['question_id']] = q['question']
    
    print(f"Found {len(annotations)} annotations")
    
    
    image_qa_map = {}
    
    for ann in annotations:
        image_id = ann['image_id']
        question_id = ann['question_id']
        
        
        question = question_lookup.get(question_id, "")
        if not question:
            print(f"Warning: No question found for question_id {question_id}")
            continue
        
        
        answer = ann['multiple_choice_answer']
        
        
        
        image_filename = f"COCO_train2014_{image_id:012d}.jpg"
        
        
        image_path = os.path.join(images_dir, image_filename)
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_filename}")
            continue
        
        
        if image_id not in image_qa_map:
            image_qa_map[image_id] = {
                'image_id': image_id,
                'image_filename': image_filename,
                'qa_pairs': []
            }
        
        image_qa_map[image_id]['qa_pairs'].append({
            'question': question,
            'answer': answer,
            'question_type': ann.get('question_type', ''),
            'answer_type': ann.get('answer_type', '')
        })
    
    
    processed_data = list(image_qa_map.values())
    
    
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    
    total_images = len(processed_data)
    total_qa = sum(len(item['qa_pairs']) for item in processed_data)
    
    
    all_answers = []
    for item in processed_data:
        for qa in item['qa_pairs']:
            all_answers.append(qa['answer'])
    
    unique_answers = len(set(all_answers))
    answer_counts = Counter(all_answers)
    
    print(f"\nPreprocessing complete!")
    print(f"Statistics:")
    print(f"   Total images: {total_images}")
    print(f"   Total Q&A pairs: {total_qa}")
    print(f"   Unique answers: {unique_answers}")
    print(f"   Avg Q&A per image: {total_qa/total_images:.1f}")
    
    print(f"\nTop 10 most common answers:")
    for answer, count in answer_counts.most_common(10):
        print(f"   {answer}: {count} ({100*count/total_qa:.1f}%)")
    
    print(f"\nSaved to: {output_file}")
    
    return processed_data
if __name__ == "__main__":
    

    print("Processing TRAINING data")

    train_data = preprocess_vqa2(
        annotations_file=r'train_data.json',
        questions_file=r'train_data.json',  
        output_file=r'train_processed.json',
        images_dir=r'train_images\train2014'
    )
    

    print("Processing VALIDATION data")

    val_data = preprocess_vqa2(
        annotations_file=r'val_data.json',
        questions_file=r'val_data.json',
        output_file=r'val_processed.json',
        images_dir=r'val_images\val2014'
    )
    
    print("ALL PREPROCESSING COMPLETE!")
    print(f"\nYou can now use:")
    print(f"  - train_processed.json for training")
    print(f"  - val_processed.json for validation")