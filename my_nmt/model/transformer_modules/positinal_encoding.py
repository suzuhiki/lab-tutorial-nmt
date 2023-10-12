import torch
import torch.nn as nn
import math
import sys

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout: float, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        
        pe = torch.zeros(max_seq_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    # x (batch_size, word_num, embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x_t (word_num, batch_size, embed_dim)
        x_t = x.permute(1, 0, 2)

        # print("x_t: {}".format(x_t.size()))
        # print("pe: {}".format(self.pe[:x_t.size(0)].size()))
        # print(self.pe[:x.size(0)])

        x_t = x_t + self.pe[:x_t.size(0)]

        # (batch_size, word_num, embed_dim)
        x = x_t.permute(1, 0, 2)

        return self.dropout(x)
