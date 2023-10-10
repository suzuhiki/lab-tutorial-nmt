import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward

class DecoderBlock(nn.Module):
    def __init__(self, feature_dim, head_num, dropout, ff_hidden_dim) -> None:
        super().__init__()
        self.s_MHA = MultiHeadAttention(feature_dim, head_num, dropout)
        self.ed_MHA = MultiHeadAttention(feature_dim, head_num, dropout)
        self.layer_norm_1 = nn.LayerNorm([feature_dim])
        self.layer_norm_2 = nn.LayerNorm([feature_dim])
        self.layer_norm_3 = nn.LayerNorm([feature_dim])
        self.FF = FeedForward(dropout, feature_dim, ff_hidden_dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
    def forward(self, x, y, mask):
        Q = K = V = x
        x = self.s_MHA(Q, K, V, mask)
        x = self.dropout_1(x)
        x = x + Q
        x = self.layer_norm_1(x)
        Q = x
        K = V = y
        x = self.ed_MHA(Q, K, V)
        x = self.dropout_2(x)
        x = x + Q
        x = self.layer_norm_2(x)
        memory = x
        x = self.FF(x)
        x = self.dropout_3(x)
        x = x + memory
        x = self.layer_norm_3(x)
        return x
    