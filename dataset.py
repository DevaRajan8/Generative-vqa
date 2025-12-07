"""
Script to download 200 images from easy-VQA dataset with their questions and answers
"""

import os
import json
import shutil
from pathlib import Path

def download_easy_vqa_data(num_images=200, output_dir="easy_vqa_data"):

    

    try:
        import easy_vqa
    except ImportError:
        print("Installing easy-vqa package...")
        os.system("pip install easy-vqa")
        import easy_vqa
    
    from easy_vqa import get_train_questions, get_train_image_paths
    
    # Create output directory structure
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {num_images} images and their Q&A pairs...")
    
    # Get questions and answers
    train_questions, train_answers, train_image_ids = get_train_questions()
    train_image_paths = get_train_image_paths()
    
    # Get unique image IDs (since multiple questions can reference same image)
    unique_image_ids = list(set(train_image_ids))[:num_images]
    
    # Prepare data structure
    dataset = []
    images_copied = 0
    
    for img_id in unique_image_ids:
        # Copy image
        src_path = train_image_paths[img_id]
        dst_path = images_dir / f"{img_id}.png"
        
        try:
            shutil.copy2(src_path, dst_path)
            images_copied += 1
            
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
            
            if images_copied % 50 == 0:
                print(f"Progress: {images_copied}/{num_images} images downloaded")
                
        except Exception as e:
            print(f"Error copying image {img_id}: {e}")
    
    # Save Q&A data to JSON
    qa_file = output_path / "questions_answers.json"
    with open(qa_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Create a summary file
    summary = {
        "total_images": len(dataset),
        "total_qa_pairs": sum(len(item["qa_pairs"]) for item in dataset),
        "dataset_info": {
            "name": "easy-VQA",
            "source": "https://github.com/vzhou842/easy-VQA",
            "image_size": "64x64",
            "image_format": "PNG"
        }
    }
    
    summary_file = output_path / "dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Download complete!")
    print(f"📁 Images saved to: {images_dir}")
    print(f"📄 Q&A pairs saved to: {qa_file}")
    print(f"📊 Summary saved to: {summary_file}")
    print(f"\nTotal images: {len(dataset)}")
    print(f"Total Q&A pairs: {summary['total_qa_pairs']}")
    
    # Display sample data
    print("\n--- Sample Data ---")
    sample = dataset[0]
    print(f"Image: {sample['image_filename']}")
    print(f"Questions for this image:")
    for qa in sample['qa_pairs'][:3]:  # Show first 3 Q&A pairs
        print(f"  Q: {qa['question']}")
        print(f"  A: {qa['answer']}")
    
    return dataset

if __name__ == "__main__":
    # Download 200 images by default
    dataset = download_easy_vqa_data(num_images=200)