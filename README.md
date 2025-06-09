# GPT-2 Implementation from Scratch

A complete implementation of GPT-2 (Generative Pre-trained Transformer 2) from scratch using PyTorch, trained on the complete works of Charles Dickens.

## 🚀 Features

- **Complete GPT-2 Architecture**: Multi-head attention, transformer blocks, positional encoding
- **Character-level Tokenization**: Simple and effective for this corpus
- **Flexible Model Sizes**: From small (6 layers) to large configurations
- **Advanced Training**: AdamW optimizer, learning rate scheduling, gradient clipping
- **Interactive Generation**: Real-time text generation with various parameters
- **Comprehensive Evaluation**: Perplexity calculation, attention analysis, diversity metrics
- **Visualization Tools**: Training curves, attention patterns, text statistics

## 📁 Project Structure

```
gpt-2/
├── requirements.txt              # Python dependencies
├── charwise-gpt/
│   ├── model.py                 # GPT-2 model architecture
│   ├── tokenizer.py             # Character-level tokenizer
│   ├── train.py                 # Training script
│   ├── generate.py              # Interactive text generation
│   ├── evaluate.py              # Model evaluation and analysis
│   ├── data_utils.py            # Data loading and preprocessing
│   ├── utils.py                 # Utility functions and configurations
│   └── dickens/
│       └── combined.txt         # Charles Dickens corpus
```

## 🛠️ Installation

1. **Clone and navigate to the project:**
```bash
cd /home/joey/projects/gpt-2
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

### 1. Train the Model

```bash
cd charwise-gpt
python train.py
```

This will:
- Load and preprocess the Charles Dickens corpus
- Build a character-level tokenizer
- Train a GPT-2 model with the following default configuration:
  - 6 transformer layers
  - 6 attention heads
  - 384 dimensional embeddings
  - 512 maximum sequence length

### 2. Generate Text Interactively

```bash
python generate.py
```

This opens an interactive mode where you can:
- Enter prompts and generate text
- Adjust generation parameters (temperature, length, etc.)
- Generate multiple samples

### 3. Evaluate the Model

```bash
python evaluate.py
```

This performs comprehensive evaluation including:
- Perplexity calculation
- Text diversity analysis
- Character and word frequency analysis
- Temperature comparison

## 🏗️ Architecture Details

### GPT-2 Model Components

1. **Token Embedding**: Converts characters to dense vectors
2. **Positional Encoding**: Learnable position embeddings
3. **Transformer Blocks**: Multi-head attention + feed-forward networks
4. **Layer Normalization**: Pre-norm architecture (GPT-2 style)
5. **Language Modeling Head**: Final projection to vocabulary

### Key Features

- **Multi-Head Attention**: Scaled dot-product attention with causal masking
- **GELU Activation**: Gaussian Error Linear Units (as in original GPT-2)
- **Dropout**: Configurable dropout for regularization
- **Gradient Clipping**: Prevents exploding gradients
- **Weight Initialization**: Following GPT-2 paper specifications

## 🎛️ Configuration Options

### Training Parameters

- **Learning Rate**: 3e-4 (with cosine annealing)
- **Batch Size**: 32 (adjustable based on GPU memory)
- **Weight Decay**: 0.1
- **Gradient Clipping**: Max norm of 1.0
- **Optimizer**: AdamW with β₁=0.9, β₂=0.95

## 📊 Usage Examples

### Basic Text Generation

```python
from generate import TextGenerator

generator = TextGenerator('best_model.pt', 'tokenizer.json')
text = generator.generate(
    prompt="It was the best of times",
    max_length=200,
    temperature=0.8
)
print(text)
```

### Batch Evaluation

```python
from evaluate import ModelEvaluator

evaluator = ModelEvaluator(model, tokenizer)
results = evaluator.run_comprehensive_evaluation(test_text)
print(f"Perplexity: {results['perplexity']:.2f}")
```

### Custom Training

```python
from train import GPT2Trainer

trainer = GPT2Trainer(model, tokenizer, train_loader, val_loader)
trainer.train(num_epochs=20)
```

## 🎮 Interactive Commands

When using the interactive generator (`python generate.py`):

- `/temp 0.8` - Set temperature (creativity level)
- `/length 300` - Set maximum generation length  
- `/samples 3` - Generate multiple samples
- `/settings` - Show current parameters
- `/help` - Show all commands
- `/quit` - Exit

## 📈 Training Monitoring

The training script provides:

- **Real-time Loss Tracking**: Training and validation loss
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Sample Generation**: Periodic text samples during training
- **Checkpointing**: Automatic model saving
- **Visualization**: Training curves and statistics

## 🔍 Evaluation Metrics

The evaluation script calculates:

1. **Perplexity**: Language modeling performance
2. **Text Diversity**: Unique n-gram ratios
3. **Character/Word Frequencies**: Distribution analysis
4. **Temperature Effects**: Generation quality vs creativity
5. **Repetition Analysis**: Repetitive pattern detection


## 🎭 Sample Outputs

Here are some example generations from the trained model:

**Prompt**: "It was the best of times"
**Generated**: "It was the best of times, and the worst of times, when the old gentleman had been so much surprised by the appearance of the strange old man who had been so much troubled by the sight of the poor old woman who had been so much distressed..."

**Prompt**: "Christmas morning"
**Generated**: "Christmas morning brought with it a strange feeling of peace and contentment that seemed to pervade the entire household. The old gentleman rose early, as was his custom, and made his way to the window..."

## 📚 Technical References

This implementation is based on:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Transformer architecture)
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2 paper)
- [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/) (Architecture explanation)


## 📄 License

This project is open source and available under the MIT License.
