# nanoGPT Java

A from-scratch Java reimplementation of [nanoGPT](https://github.com/karpathy/ng-video-lecture) — Andrej Karpathy's educational GPT implementation from the "Neural Networks: Zero To Hero" video lecture series.

## Overview

This is a pure Java implementation with **no external dependencies**. Everything is built from scratch:

- **Tensor**: N-dimensional array with automatic differentiation (autograd)
- **Neural network modules**: Linear, Embedding, LayerNorm, Dropout
- **Transformer architecture**: Self-attention heads, multi-head attention, feed-forward blocks
- **Two models**: Simple bigram baseline and full GPT transformer
- **AdamW optimizer** with weight decay
- **Character-level tokenizer** and data loader

## Project Structure

```
java/
├── pom.xml                              # Maven build configuration
└── src/main/java/nanogpt/
    ├── Tensor.java                      # Core tensor with autograd engine
    ├── Module.java                      # Base neural network module
    ├── Linear.java                      # Fully connected layer
    ├── Embedding.java                   # Embedding lookup table
    ├── LayerNorm.java                   # Layer normalization
    ├── Head.java                        # Single self-attention head
    ├── MultiHeadAttention.java          # Multi-head attention
    ├── FeedForward.java                 # Feed-forward network (Linear → ReLU → Linear)
    ├── Block.java                       # Transformer block (attention + FFN + residuals)
    ├── BigramLanguageModel.java         # Simple bigram language model
    ├── GPTLanguageModel.java            # Full GPT transformer model
    ├── AdamW.java                       # AdamW optimizer
    ├── Tokenizer.java                   # Character-level tokenizer
    ├── DataLoader.java                  # Mini-batch data loader
    └── Main.java                        # Entry point with training loops
```

## Building

Requires Java 17+.

```bash
# Using Maven
cd java
mvn compile

# Or compile directly
cd java
mkdir -p target/classes
javac -d target/classes src/main/java/nanogpt/*.java
```

## Running

### Bigram Model
```bash
# From the repository root (where input.txt is located)
java -cp java/target/classes nanogpt.Main bigram input.txt
```

### GPT Model
```bash
java -cp java/target/classes nanogpt.Main gpt input.txt
```

## Architecture

### Autograd Engine

The `Tensor` class implements reverse-mode automatic differentiation. Each tensor operation records its parents and a backward function. When `backward()` is called on the loss tensor, it performs a topological sort of the computation graph and propagates gradients in reverse order — the same approach used by PyTorch.

Supported operations: matmul (2D/3D batched), add (with broadcasting), scalar multiply, softmax, ReLU, masked fill, concatenation, transpose, view/reshape, embedding lookup, cross-entropy loss, layer normalization, and dropout.

### Model Correspondence

| Python (gpt.py)       | Java                        |
|------------------------|-----------------------------|
| `nn.Linear`            | `Linear.java`               |
| `nn.Embedding`         | `Embedding.java`            |
| `nn.LayerNorm`         | `LayerNorm.java`            |
| `nn.Dropout`           | `Tensor.dropout()`          |
| `Head`                 | `Head.java`                 |
| `MultiHeadAttention`   | `MultiHeadAttention.java`   |
| `FeedFoward`           | `FeedForward.java`          |
| `Block`                | `Block.java`                |
| `GPTLanguageModel`     | `GPTLanguageModel.java`     |
| `BigramLanguageModel`  | `BigramLanguageModel.java`  |
| `torch.optim.AdamW`    | `AdamW.java`                |

### Hyperparameters

**Bigram model** (matching bigram.py):
- batch_size=32, block_size=8, max_iters=3000, lr=1e-2

**GPT model** (matching gpt.py):
- batch_size=64, block_size=256, max_iters=5000, lr=3e-4
- n_embd=384, n_head=6, n_layer=6, dropout=0.2

## Notes

- This is a CPU-only implementation (no GPU/CUDA support)
- Training the full GPT model is significantly slower than the PyTorch version due to lack of GPU acceleration and optimized BLAS libraries
- The bigram model trains quickly and demonstrates the full training pipeline
- Weight initialization uses normal(0, 0.02) matching the original implementation
