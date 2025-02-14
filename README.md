# Building a Transformer from Scratch

This repository provides a step-by-step implementation of a transformer architecture, designed for educational purposes. Each step is self-contained and builds upon the previous ones, allowing learners to understand how transformers work from the ground up.

## Project Structure

### 📚 Tutorials
Step-by-step guides explaining each component:
1. [Setup and Prerequisites](tutorials/00_setup.md)
2. [Basic Attention Mechanism](tutorials/01_attention_basic.md)
3. [Multi-Head Attention](tutorials/02_attention_multi.md)
4. [Position Encodings](tutorials/03_position_encodings.md)
5. [Full Transformer](tutorials/04_full_transformer.md)

### 💻 Implementation Versions
Each version represents a different stage of development:
- v1: Basic attention mechanism
- v2: Multi-head attention
- v3: Complete transformer

### 🔬 Examples
- Attention visualization
- Practical use cases
- Interactive experiments

## Getting Started

### Prerequisites
```bash
python 3.10+
poetry
```

### Installation
```bash
# Clone the repository
git clone https://github.com/Paulisure/transformer-implementation.git
cd transformer-implementation

# Install dependencies
poetry install
```

### Running the Examples
```bash
# Activate the poetry environment
poetry shell

# Run specific version
python -m src.transformer.v1_basic_attention.main
```

## Learning Path

### 1. Basic Attention (Current Stage)
- Understanding scaled dot-product attention
- Implementing basic attention mechanism
- Visualizing attention patterns

### 2. Multi-Head Attention (Next)
- Multiple attention heads
- Parallel attention computation
- Information from different representation subspaces

### 3. Full Transformer (Future)
- Position encodings
- Encoder-decoder architecture
- Training and optimization

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.