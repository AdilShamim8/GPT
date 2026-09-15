# GPT-2 Rebuild nanoGPT

A beginner-friendly implementation of a GPT-2 style model, based on Andrej Karpathy's ["Neural Networks: Zero To Hero"](https://karpathy.ai/zero-to-hero.html) lecture series. This repository follows the nanoGPT implementation demonstrated in [this YouTube lecture](https://youtu.be/kCc8FmEb1nY?feature=shared).

## Overview

This project aims to rebuild the nanoGPT architecture from scratch, providing a clear, educational implementation of the key concepts behind transformer-based language models. Following Karpathy's approach, we start simple and incrementally build toward a functioning GPT-2 style model.

## Repository Contents

- `bigram.py`: Simple bigram language model implementation
- `gpt.py`: Full GPT model implementation with transformer architecture
- `gpt_dev.ipynb`: Interactive notebook for development and experimentation
- `input.txt`: Sample text for training the model
- `more.txt`: Additional text data for experimentation

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.0+
- NumPy

### Installation

```bash
# Clone the repository
git clone https://github.com/AdilShamim8/GPT-2-Rebuild-nanoGPT.git
cd GPT-2-Rebuild-nanoGPT

# Install dependencies
pip install torch numpy matplotlib
```

### Usage

#### Training a Bigram Model

```bash
python bigram.py
```

#### Training the GPT Model

```bash
python gpt.py
```

#### Interactive Development

For an interactive experience, open and run the Jupyter notebook:

```bash
jupyter notebook gpt_dev.ipynb
```

## Project Structure

The project follows a progressive approach to building a GPT-2 style model:

1. **Bigram Language Model**: A simple statistical model (`bigram.py`)
2. **Full GPT Implementation**: Transformer-based model with self-attention (`gpt.py`)
3. **Development Notebook**: Interactive experimentation with model components (`gpt_dev.ipynb`)

## Model Architecture

The GPT model implemented here includes:

- Token embedding layer
- Positional encoding
- Multi-head self-attention
- Feed-forward neural networks
- Layer normalization
- Dropout for regularization

## Implementation Notes

As noted in the original repository, model initialization is crucial for good performance. While the current implementation will train and function, convergence may be slower than optimal without proper initialization techniques.

## Customization

You can experiment with different hyperparameters in both `bigram.py` and `gpt.py`:

- Model size (n_embd)
- Number of layers (n_layer)
- Number of attention heads (n_head)
- Batch size
- Learning rate
- Context length

## Learning Resources

- [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) - The lecture series this project is based on
- [The original nanoGPT lecture](https://youtu.be/kCc8FmEb1nY?feature=shared) - YouTube video
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual guide to transformers
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Original research paper
- [Attention is All You Need paper](https://arxiv.org/abs/1706.03762) - Original research paper
- - [OpenAI ChatGPT blog post](https://openai.com/index/chatgpt/)

## Acknowledgments

This repository is inspired by and builds upon [Andrej Karpathy's nanoGPT lecture code](https://github.com/karpathy/ng-video-lecture). The implementation follows his educational approach to building transformer-based language models from scratch.

##License

This project is licensed under the MIT License - see the LICENSE file for details.
---
| Last updated: 2025-09-16
