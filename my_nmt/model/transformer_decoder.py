import torch
import torch.nn as nn
import sys
from .transformer_modules.positinal_encoding import PositionalEncoding
from .transformer_modules.decoder_block import DecoderBlock

class TransformerDecoder(nn.Module):
    def __init__(self, dec_vocab_dim, feature_dim, dropout, head_num, special_token, max_len, device, ff_hidden_dim = 2048, block_num = 6, init=False) -> None:
        super(TransformerDecoder, self).__init__()
        
        self.device = device
        self.special_token = special_token
        self.embed = nn.Embedding(dec_vocab_dim, feature_dim, special_token["<pad>"])
        self.pos_enc = PositionalEncoding(feature_dim, dropout, device, max_len)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(feature_dim, head_num, dropout, ff_hidden_dim, device) for _ in range(block_num)])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(feature_dim, dec_vocab_dim, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.scale = torch.sqrt(torch.FloatTensor([feature_dim])).to(device)

        nn.init.constant_(self.embed.weight[self.special_token["<pad>"]], 0)
        if init:
            nn.init.normal_(self.embed.weight, 0, std=feature_dim ** -0.5)
            nn.init.xavier_normal_(self.linear.weight)
        
    def forward(self, enc_state, decoder_input, src_mask):
        tgt_mask = self.make_target_mask(decoder_input)

        x = self.embed(decoder_input)

        x = torch.mul(x, self.scale)

        x = self.pos_enc(x)

        x = self.dropout(x)

        block_in = x
        for decoder_block in self.decoder_blocks:
            block_out = decoder_block(block_in, enc_state, tgt_mask, src_mask)
            block_in = block_out
        x = block_out

        x = self.linear(x)

        return x


    def make_target_mask(self, decoder_input):
        pad_mask = torch.where(decoder_input == self.special_token["<pad>"], 1, 0).unsqueeze(1).unsqueeze(2)

        tgt_mask = torch.ones((pad_mask.size(-1), pad_mask.size(-1))).triu(diagonal=1).to(self.device)

        tgt_mask = pad_mask + tgt_mask
        tgt_mask = torch.where(tgt_mask >= 1, 1, 0)
        
        return tgt_mask