import torch
import torch.nn as nn
import math
class ScaledDotProductAttention(nn.Module):
    """
    Implements the scaled dot-product attention mechanism:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, query, key, value, mask=None):
        # query, key, value shapes: (batch_size, num_heads, seq_length, d_k)
        d_k = query.size(-1)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Compute output
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention allows the model to jointly attend to information
    from different representation subspaces at different positions.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention()

    def split_heads(self, x):
        """Split the last dimension into (num_heads, d_k)"""
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and split heads
        query = self.split_heads(self.W_q(query))
        key = self.split_heads(self.W_k(key))
        value = self.split_heads(self.W_v(value))
        
        # Apply attention
        output, attention_weights = self.attention(query, key, value, mask)
        
        # Concatenate heads and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output, attention_weights