import torch
import torch.nn as nn

from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(self, feature_dim, head_num, dropout, ff_hidden_dim, device) -> None:
        super(EncoderBlock, self).__init__()
        
        self.MHA = MultiHeadAttention(feature_dim, head_num, dropout, device)
        self.layer_norm_1 = nn.LayerNorm([feature_dim])
        self.layer_norm_2 = nn.LayerNorm([feature_dim])
        self.FF = FeedForward(dropout, feature_dim, ff_hidden_dim=ff_hidden_dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, input, mask):
        Q = K = V = input
        
        x = self.MHA(Q, K ,V, mask)
        x = self.dropout_1(x)
        
        # 残差接続
        x = x + Q
        x = self.layer_norm_1(x)
        
        memory = x
        
        x = self.FF(x)
        x = self.dropout_2(x)
        x = x + memory
        x = self.layer_norm_2(x)
        
        return x