import torch
import torch.nn as nn
from .transformer_modules.positinal_encoding import PositionalEncoding
from .transformer_modules.decoder_block import DecoderBlock

class TransformerDecoder(nn.Module):
    def __init__(self, dec_vocab_dim, feature_dim, dropout, head_num, special_token, device, ff_hidden_dim = 2048, block_num = 6) -> None:
        super(TransformerDecoder, self).__init__()
        
        self.device = device
        self.special_token = special_token
        self.dec_vocab_dim = dec_vocab_dim
        self.feature_dim = feature_dim
        self.embed = nn.Embedding(dec_vocab_dim, feature_dim, special_token["<pad>"])
        self.pos_enc = PositionalEncoding(feature_dim, dropout)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(feature_dim, head_num, dropout, ff_hidden_dim, device) for _ in range(block_num)])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(feature_dim, dec_vocab_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, enc_state, decoder_in, max_len = 51):
        
        if self.training == True:
            mask = torch.where(decoder_in == self.special_token["<pad>"], 1, 0)
            
            s_mask = torch.ones_like(mask).triu(diagonal = 1)
            s_mask = s_mask + mask
            s_mask = torch.where(s_mask >= 1, 1, 0)
            
            x = self.embed(decoder_in)
            x = x*(self.feature_dim**0.5)
            x = self.pos_enc(x)
            x = self.dropout(x)
            for decoder_block in self.decoder_blocks:
                x = decoder_block(x, enc_state, mask, s_mask)
            x = self.linear(x)
            x = self.softmax(x)
            return x
        
        else:
            batch_size = enc_state.size(0)
            initial_data = torch.full((batch_size, 1), self.special_token["<bos>"]).to(self.device)
            input_data = initial_data
            
            for i in range(max_len):
                mask = torch.where(input_data == self.special_token["<pad>"], 1, 0)
                
                x = self.embed(input_data)
                x = x*(self.feature_dim**0.5)
                x = self.pos_enc(x)
                x = self.dropout(x)
                for decoder_block in self.decoder_blocks:
                    x = decoder_block(x, enc_state, mask)
                x = self.linear(x)
                
                # 単語ベクトルに変換 (batch_size, word_len)
                x = torch.argmax(x, dim = 2)
                input_data = torch.cat((initial_data, x), dim = 1)
            
            return input_data
