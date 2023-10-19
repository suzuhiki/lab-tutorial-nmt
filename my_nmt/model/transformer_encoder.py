import torch
import torch.nn as nn
import sys
from .transformer_modules.positinal_encoding import PositionalEncoding
from .transformer_modules.encoder_block import EncoderBlock


class TransformerEncoder(nn.Module):
    def __init__(self, enc_vocab_dim, feature_dim, dropout, head_num, special_token, max_len, device, ff_hidden_dim = 2048, block_num = 6) -> None:
        super(TransformerEncoder, self).__init__()
        
        self.device = device
        self.special_token = special_token
        self.feature_dim = feature_dim
        self.embed = nn.Embedding(enc_vocab_dim, feature_dim, special_token["<pad>"])
        self.pos_enc = PositionalEncoding(feature_dim, dropout, device, max_len)
        self.dropout = nn.Dropout(dropout)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(feature_dim, head_num, dropout, ff_hidden_dim, device) for _ in range(block_num)]) 
        self.scale = torch.sqrt(torch.FloatTensor([feature_dim])).to(device)
    
    # x (batch_size, word_num)
    def forward(self, x, mask):

        # print(x[0])
        x = self.embed(x)*self.scale
        # print("scaled: {}".format(x[0][0]))
        x = self.pos_enc(x)
        # print("after_pos_enc: {}".format(x[0][0]))
        x = self.dropout(x)
        # print("after_dropout: {}".format(x[0][0]))

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask)
            # print(torch.argmax(x, dim=2)[0])
        
        return x