import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from src.transformer.v1_basic_attention.attention import MultiHeadAttention

def get_output_dir():
    """Get the output directory for the current tutorial."""
    # Assuming this script is run from the project root
    output_dir = Path('tutorials/tutorial_1_basic_attention/outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def plot_attention_weights(attention_weights, title="Attention Weights", token_labels=None):
    """Enhanced plotting function with custom token labels"""
    output_dir = get_output_dir()
    
    weights = attention_weights[0, 0].detach().numpy()
    plt.figure(figsize=(10, 8))
    
    if token_labels is None:
        token_labels = [f'Token {i+1}' for i in range(weights.shape[0])]
    
    sns.heatmap(weights, 
                annot=True, 
                fmt='.2f', 
                cmap='Blues',
                xticklabels=token_labels,
                yticklabels=token_labels)
    plt.title(title)
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.tight_layout()
    
    output_path = output_dir / f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

def demonstrate_fibonacci_attention():
    """Shows attention patterns in a Fibonacci sequence"""
    d_model = 8
    num_heads = 1
    seq_length = 6
    batch_size = 1
    model = MultiHeadAttention(d_model, num_heads)
    
    # Create Fibonacci sequence
    sequence = [1, 1]
    for i in range(4):
        sequence.append(sequence[-1] + sequence[-2])
    
    # Create input tensors
    query = torch.zeros(batch_size, seq_length, d_model)
    for i, num in enumerate(sequence):
        query[0, i, 0] = num / max(sequence)  # Normalize values
    
    output, weights = model(query, query, query)
    plot_attention_weights(
        weights,
        "Fibonacci Sequence Attention",
        [f'Fib({i+1})={num}' for i, num in enumerate(sequence)]
    )
    return weights

def demonstrate_pattern_attention():
    """Shows attention between repeating patterns"""
    d_model = 8
    num_heads = 1
    seq_length = 6
    batch_size = 1
    model = MultiHeadAttention(d_model, num_heads)
    
    # Create patterns
    pattern_a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    pattern_b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    pattern_c = torch.tensor([0.0, 0.0, 1.0, 0.0])
    
    # Create sequence with repeating patterns
    query = torch.zeros(batch_size, seq_length, d_model)
    patterns = [pattern_a, pattern_b, pattern_c, pattern_a, pattern_b, pattern_c]
    for i, pattern in enumerate(patterns):
        query[0, i, :4] = pattern
    
    output, weights = model(query, query, query)
    plot_attention_weights(
        weights,
        "Pattern Recognition Attention",
        ['A1', 'B1', 'C1', 'A2', 'B2', 'C2']
    )
    return weights

def demonstrate_dependency_attention():
    """Shows attention with sequential dependencies"""
    d_model = 8
    num_heads = 1
    seq_length = 4
    batch_size = 1
    model = MultiHeadAttention(d_model, num_heads)
    
    # Create sequence where each position depends on previous ones
    query = torch.zeros(batch_size, seq_length, d_model)
    for i in range(seq_length):
        query[0, i, :i+1] = 1.0  # Each position attends to all previous positions
    
    output, weights = model(query, query, query)
    plot_attention_weights(
        weights,
        "Sequential Dependency Attention",
        ['Start', 'Dep(1)', 'Dep(1,2)', 'Dep(1,2,3)']
    )
    return weights

if __name__ == "__main__":
    print("Generating attention visualizations for Tutorial 1...")
    
    print("\nCase 1: Fibonacci Sequence Attention")
    print("Expected: Each number should attend strongly to its two predecessors")
    demonstrate_fibonacci_attention()
    
    print("\nCase 2: Pattern Recognition")
    print("Expected: Similar patterns (A1-A2, B1-B2, C1-C2) should attend to each other")
    demonstrate_pattern_attention()
    
    print("\nCase 3: Sequential Dependencies")
    print("Expected: Later positions should attend more to earlier positions")
    demonstrate_dependency_attention()