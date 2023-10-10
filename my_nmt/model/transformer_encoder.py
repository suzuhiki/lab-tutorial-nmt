import torch
import torch.nn as nn
from .transformer_modules.positinal_encoding import PositionalEncoding
from .transformer_modules.encoder_block import EncoderBlock

class TransformerEncoder(nn.Module):
    def __init__(self, enc_vocab_dim, feature_dim, dropout, head_num, special_token, ff_hidden_dim = 2048, block_num = 6) -> None:
        super(TransformerEncoder, self).__init__()
        
        self.feature_dim = feature_dim
        self.embed = nn.Embedding(enc_vocab_dim, feature_dim, special_token["<pad>"])
        self.pos_enc = PositionalEncoding(feature_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(feature_dim, head_num, dropout, ff_hidden_dim) for _ in range(block_num)]) 
    
    def forward(self, x):
        x = self.embed(x)
        x = x*(self.feature_dim**0.5)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        return x