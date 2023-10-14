import torch
import torch.nn as nn
import sys

from .transformer_encoder import TransformerEncoder
from .transformer_decoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(self, vocab_size_src, vocab_size_tgt, dropout, head_num, feature_dim, special_token, device, block_num = 6, ff_hidden_size = 2048) -> None:
        super().__init__()
        
        self.encoder = TransformerEncoder(vocab_size_src, feature_dim, dropout, head_num, special_token, device, ff_hidden_size, block_num)
        self.decoder = TransformerDecoder(vocab_size_tgt, feature_dim, dropout, head_num, special_token, device, ff_hidden_size, block_num)
        self.special_token = special_token

    def forward(self, src, tgt):        
        gen_len = src.size(1) + 50
        
        # padの部分を1にする
        src_mask = torch.where(src == self.special_token["<pad>"], 1, 0).unsqueeze(1).unsqueeze(2)
        
        encoder_state = self.encoder(src, src_mask)

        output = self.decoder(encoder_state, tgt, src_mask, gen_len)
        return output