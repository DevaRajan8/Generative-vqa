import os
import json
import random
import shutil
import pandas as pd
from tqdm import tqdm
from collections import Counter
IMAGES_DIR = r"../train2014"
QUESTIONS_PATH = r"../v2_OpenEnded_mscoco_train2014_questions.json"
# ANNOTATIONS_PATH = r"../v2_mscoco_train2014_annotations.json"
OUTPUT_DIR = "./gen_vqa_v2"
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
print("Loading VQA v2 data...")
with open(QUESTIONS_PATH, "r") as f:
    questions = json.load(f)["questions"]
with open(ANNOTATIONS_PATH, "r") as f:
    annotations = json.load(f)["annotations"]
qid_to_ann = {ann["question_id"]: ann for ann in annotations}
print("Merging questions and answers...")
merged_data = []
answer_counter = Counter()
EXCLUDED_ANSWERS = ['yes', 'no', 'unknown', 'none', 'n/a', 'cant tell', 'not sure']
AMBIGUOUS_QUESTIONS = ['what is in the image', 'what is this', 'what is that', 'what do you see']
for q in tqdm(questions, total=len(questions)):
    ann = qid_to_ann.get(q["question_id"])
    if not ann:
        continue
    
    answers = [a["answer"] for a in ann["answers"] if a["answer"].strip()]
    if not answers:
        continue
    
    main_answer = max(set(answers), key=answers.count)
    main_answer = main_answer.lower().strip()
    question_text = q["question"].lower().strip()
    
    if main_answer in EXCLUDED_ANSWERS:
        continue
    
    if any(ambig in question_text for ambig in AMBIGUOUS_QUESTIONS):
        continue
    
    if len(main_answer.split()) <= 5 and len(main_answer) <= 30:
        merged_data.append({
            "image_id": q["image_id"],
            "question_id": q["question_id"],
            "question": q["question"],
            "answer": main_answer
        })
        answer_counter[main_answer] += 1
print(f"Total valid Q-A pairs (after filtering): {len(merged_data)}")
MIN_ANSWER_FREQ = 20
frequent_answers = {ans for ans, count in answer_counter.items() if count >= MIN_ANSWER_FREQ}
filtered_data = [item for item in merged_data if item["answer"] in frequent_answers]
print(f"After frequency filtering (min_freq={MIN_ANSWER_FREQ}): {len(filtered_data)} pairs")
MAX_SAMPLES_PER_ANSWER = 600
answer_samples = {}
for item in filtered_data:
    ans = item["answer"]
    if ans not in answer_samples:
        answer_samples[ans] = []
    if len(answer_samples[ans]) < MAX_SAMPLES_PER_ANSWER:
        answer_samples[ans].append(item)
balanced_data = []
for samples in answer_samples.values():
    balanced_data.extend(samples)
random.shuffle(balanced_data)
print(f"After balancing: {len(balanced_data)} pairs with {len(answer_samples)} unique answers")
print("Copying selected images and saving data...")
final_data = []
for item in tqdm(balanced_data):
    img_name = f"COCO_train2014_{item['image_id']:012d}.jpg"
    src_path = os.path.join(IMAGES_DIR, img_name)
    dst_path = os.path.join(OUTPUT_DIR, "images", img_name)
    
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        item["image_path"] = f"images/{img_name}"
        final_data.append(item)
print(f"Final dataset: {len(final_data)} pairs")
with open(os.path.join(OUTPUT_DIR, "qa_pairs.json"), "w") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)
pd.DataFrame(final_data).to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)
print("Data preparation complete.")