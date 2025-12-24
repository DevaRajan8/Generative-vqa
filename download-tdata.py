import os
import json
import shutil
from pathlib import Path

def download_specific_images(image_ids, output_dir="easy_vqa_data"):

    
    # try:
    #     import easy_vqa
    # except ImportError:
    #     print("Installing easy-vqa package...")
    #     os.system("pip install easy-vqa")
    #     import easy_vqa
    
    from easy_vqa import get_train_questions, get_train_image_paths
    

    if isinstance(image_ids, int):
        image_ids = [image_ids]
    

    output_path = Path(output_dir)
    images_dir = output_path / "timages"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {len(image_ids)} specific image(s)...")
    

    train_questions, train_answers, train_image_ids = get_train_questions()
    train_image_paths = get_train_image_paths()
    
    dataset = []
    
    for img_id in image_ids:
        # Check if image ID exists
        if img_id not in train_image_paths:
            print(f"Image ID {img_id} not found in dataset")
            continue
            
        # Copy image
        src_path = train_image_paths[img_id]
        dst_path = images_dir / f"{img_id}.png"
        
        try:
            shutil.copy2(src_path, dst_path)
            print(f"Copied image {img_id}.png")
            
            # Get all questions for this image
            image_qa_pairs = []
            for idx, question_img_id in enumerate(train_image_ids):
                if question_img_id == img_id:
                    image_qa_pairs.append({
                        "question": train_questions[idx],
                        "answer": train_answers[idx]
                    })
            
            dataset.append({
                "image_id": img_id,
                "image_filename": f"{img_id}.png",
                "qa_pairs": image_qa_pairs
            })
            
            print(f"   Found {len(image_qa_pairs)} Q&A pairs for this image")
            
        except Exception as e:
            print(f" Error copying image {img_id}: {e}")
    

    if dataset:
        qa_file = output_path / "tquestions_answers.json"
        with open(qa_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\nQ&A pairs saved to: {qa_file}")
        
        print("\n--- Downloaded Data ---")
        for item in dataset:
            print(f"\nImage: {item['image_filename']} (ID: {item['image_id']})")
            print(f"Q&A pairs:")
            for qa in item['qa_pairs']:
                print(f"  Q: {qa['question']}")
                print(f"  A: {qa['answer']}")
    
    return dataset

if __name__ == "__main__":
    dataset = download_specific_images(202)
