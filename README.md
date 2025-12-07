# SynerVQA

A Visual Question Answering (VQA) system using joint embedding architecture trained on the easy-VQA dataset.

## Overview

SynerVQA implements a deep learning model that answers natural language questions about images by learning joint embeddings of visual and textual features.

## Features

- **Joint Embedding Architecture**: Combines CNN-based image features with LSTM-based question encoding
- **Easy-VQA Dataset**: Automated download and preprocessing of 64x64 training images
- **Training Pipeline**: Includes data augmentation, early stopping, and learning rate scheduling
- **Inference**: Pre-trained model for question answering on new images

## Project Structure

```
├── dataset.py           # Dataset downloader (200 images)
├── download-tdata.py    # Specific image downloader
├── preliminaries/
│   └── joint-emb/       # Main VQA implementation
│       ├── model.py     # JointEmbeddingVQA architecture
│       ├── train.py     # Training loop & early stopping
│       ├── test.py      # Validation utilities
│       ├── predict.py   # Inference script
│       └── main.py      # Training entry point
└── easy_vqa_data/       # Downloaded dataset (images + Q&A pairs)
```

## Quick Start

**1. Download Dataset**
```bash
python dataset.py
```

**2. Train Model**
```bash
cd preliminaries/joint-emb
python main.py
```

**3. Run Inference**
```bash
python predict.py
```

## Model Architecture

- **Vision**: CNN (Conv2D → BatchNorm → ReLU → MaxPool)
- **Language**: LSTM with word embeddings
- **Fusion**: Concatenated joint embedding → MLP classifier

## Requirements

- PyTorch
- torchvision
- easy-vqa
- PIL

## License

MIT License - Copyright (c) 2025 Devarajan S