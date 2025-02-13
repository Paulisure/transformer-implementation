import torch
import matplotlib.pyplot as plt
import seaborn as sns
from ..attention import MultiHeadAttention

def test_attention_shapes():
    """Test if the attention mechanism produces correct output shapes."""
    batch_size = 2
    seq_length = 4
    d_model = 8
    num_heads = 2
    
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    query = torch.randn(batch_size, seq_length, d_model)
    key = torch.randn(batch_size, seq_length, d_model)
    value = torch.randn(batch_size, seq_length, d_model)
    
    output, attention_weights = mha(query, key, value)
    
    assert output.shape == (batch_size, seq_length, d_model), \
        f"Expected output shape {(batch_size, seq_length, d_model)}, got {output.shape}"
    
    assert attention_weights.shape == (batch_size, num_heads, seq_length, seq_length), \
        f"Expected attention weights shape {(batch_size, num_heads, seq_length, seq_length)}, got {attention_weights.shape}"
    
    return True

def test_attention_mask():
    """Test if the attention mechanism correctly applies masks."""
    batch_size = 1
    seq_length = 4
    d_model = 8
    num_heads = 2
    
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    query = torch.randn(batch_size, seq_length, d_model)
    key = torch.randn(batch_size, seq_length, d_model)
    value = torch.randn(batch_size, seq_length, d_model)
    
    # Create a causal mask (lower triangular)
    mask = torch.tril(torch.ones(batch_size, num_heads, seq_length, seq_length))
    
    output, attention_weights = mha(query, key, value, mask)
    
    # Check if upper triangular part is zeroed out
    upper_triangular = attention_weights[:, :, :, :]
    upper_triangular = torch.triu(upper_triangular, diagonal=1)
    
    assert torch.all(upper_triangular == 0), "Mask was not correctly applied"
    
    return True

if __name__ == "__main__":
    test_attention_shapes()
    test_attention_mask()
    print("All tests passed!")