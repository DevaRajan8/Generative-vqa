# GitHub Copilot Instructions for SynerVQA Project

## Project Context
This is a research project implementing a dual-stream Visual Question Answering (VQA) system for robotics applications, combining:
- **Stream A**: Attention-based VQA model (interpretable)
- **Stream B**: Embedding-based VQA model (generalizable)
- **RAG**: Retrieval-Augmented Generation for knowledge integration
- **Fusion**: Combining both streams for robust predictions

## Code Style Guidelines

### General Python
- Follow PEP 8 conventions
- Use type hints for all function signatures
- Maximum line length: 120 characters
- Use Black for formatting
- Use descriptive variable names (no single letters except loop counters)

### PyTorch Specifics
- Always move tensors to device explicitly: `tensor.to(device)`
- Use `torch.nn.Module` for all model components
- Implement `forward()` method for all models
- Add docstrings explaining input/output tensor shapes

Example:
```python
def forward(self, images: torch.Tensor, questions: torch.Tensor) -> torch.Tensor:
    """
    Args:
        images: (B, 3, 224, 224) - Batch of RGB images
        questions: (B, max_len) - Tokenized questions
    Returns:
        logits: (B, num_classes) - Answer predictions
    """
```

### Model Architecture
- Keep encoders, attention, and classifiers in separate classes
- Use config files (YAML) for hyperparameters
- Log all hyperparameters at training start
- Save checkpoints regularly

### Data Pipeline
- Use `torch.utils.data.Dataset` for custom datasets
- Implement proper `__len__` and `__getitem__`
- Handle data preprocessing in dataset class
- Use data augmentation for training only

### Training Code
- Use `tqdm` for progress bars
- Log to TensorBoard
- Save best model based on validation accuracy
- Implement early stopping
- Use learning rate schedulers

### Testing
- Write unit tests for all utility functions
- Use pytest fixtures for common test data
- Mock external dependencies (datasets, models)
- Aim for >80% code coverage

## Project-Specific Patterns

### Attention Visualization
When generating attention maps:
```python
attention_weights = attention_module(features, question)  # (B, H, W)
visualize_attention(image, attention_weights, save_path)
```

### RAG Integration
When adding retrieval:
```python
knowledge = rag_module.retrieve(question, top_k=5)
enhanced_features = concat([image_features, text_features, knowledge])
```

### Fusion Logic
When combining streams:
```python
stream_a_output = stream_a(image, question)  # Attention-based
stream_b_output = stream_b(image, question)  # Embedding-based
final_output = fusion_module(stream_a_output, stream_b_output)
```

## Common Pitfalls to Avoid
- ❌ Don't hardcode batch size
- ❌ Don't forget to call `model.eval()` during evaluation
- ❌ Don't forget `torch.no_grad()` during inference
- ❌ Don't mix CPU and GPU tensors
- ❌ Don't forget to normalize images (ImageNet stats)
- ❌ Don't use mutable default arguments

## Preferred Libraries
- **Deep Learning**: PyTorch (not TensorFlow)
- **Computer Vision**: torchvision, opencv-python
- **NLP**: transformers, tokenizers
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn, tensorboard
- **Config**: YAML (via PyYAML)
- **Testing**: pytest, pytest-cov

## Documentation Standards
- All modules should have module-level docstrings
- All classes should document attributes
- All functions should have Google-style docstrings
- Include usage examples for complex functions

## File Naming Conventions
- Models: `{stream_name}_model.py` (e.g., `stream_a_model.py`)
- Training scripts: `train_{model}.py`
- Configs: `{experiment_name}.yaml`
- Tests: `test_{module_name}.py`

## When Suggesting Code
1. Always include error handling
2. Add logging statements for debugging
3. Include shape comments for tensors
4. Suggest performance optimizations when relevant
5. Mention potential edge cases

## Research-Specific
- When implementing papers, cite the source
- Compare implementations with paper specifications
- Log all experimental settings
- Save results in structured JSON format
- Generate comparison tables automatically
