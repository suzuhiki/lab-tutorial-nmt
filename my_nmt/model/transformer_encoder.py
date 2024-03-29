import torch
import torch.nn as nn
import sys
from .transformer_modules.positinal_encoding import PositionalEncoding
from .transformer_modules.encoder_block import EncoderBlock


class TransformerEncoder(nn.Module):
    def __init__(self, enc_vocab_dim, feature_dim, dropout, head_num, special_token, max_len, device, ff_hidden_dim = 2048, block_num = 6, init = False) -> None:
        super(TransformerEncoder, self).__init__()
        
        self.device = device
        self.feature_dim = feature_dim
        self.embed = nn.Embedding(enc_vocab_dim, feature_dim, special_token["<pad>"])
        self.pos_enc = PositionalEncoding(feature_dim, dropout, device, max_len)
        self.dropout = nn.Dropout(dropout)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(feature_dim, head_num, dropout, ff_hidden_dim, device) for _ in range(block_num)]) 
        self.scale = torch.sqrt(torch.FloatTensor([feature_dim])).to(device)

        nn.init.constant_(self.embed.weight[special_token["<pad>"]], 0)
        if init:
            nn.init.normal_(self.embed.weight, 0, std=feature_dim ** -0.5)

    
    # x (batch_size, word_num)
    def forward(self, input, mask):
        embed = self.embed(input)

        encodeed = torch.mul(embed, self.scale)
        pos_encoded = self.pos_enc(encodeed)
        pos_encoded = self.dropout(pos_encoded)

        block_in = pos_encoded
        for encoder_block in self.encoder_blocks:
            block_out = encoder_block(block_in, mask)
            block_in = block_out
        output = block_out

        return output