import torch
import torch.nn as nn

from .transformer_encoder import TransformerEncoder
from .transformer_decoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(self, vocab_size_src, vocab_size_tgt, dropout, head_num, feature_dim, special_token, block_num = 6, ff_hidden_size = 2048) -> None:
        super().__init__()
        
        self.encoder = TransformerEncoder(vocab_size_src, feature_dim, dropout, head_num, special_token, ff_hidden_size, block_num)
        self.decoder = TransformerDecoder(vocab_size_tgt, feature_dim, dropout, head_num, special_token, ff_hidden_size, block_num)

    def forward(self, src, tgt, mask):
        encoder_state = self.encoder(src)
        vocab_vec = self.decoder(encoder_state, tgt, mask)
        return vocab_vec