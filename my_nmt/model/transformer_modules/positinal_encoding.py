import torch
import torch.nn as nn
import math
import sys

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout: float, device, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.device = device

    # x (batch_size, word_num, embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        pos = torch.arange(0, x.size(1)).unsqueeze(0).repeat(x.size(0), 1).to(self.device)
        x = x + self.pos_embedding(pos)

        return x
