import torch
import torch.nn as nn
from .lstm_encoder import LSTM_Encoder
from .lstm_decoder import LSTM_Decoder

class LSTM(nn.Module):
    def __init__(self, hidden_size, vocab_size_src, vocab_size_tgt, padding_idx, embed_dim, device) -> None:
        super().__init__()
        
        self.vocab_size_tgt = vocab_size_tgt
        self.encoder = LSTM_Encoder(vocab_size_src, embed_dim, hidden_size, padding_idx, device)
        self.decoder = LSTM_Decoder(vocab_size_tgt, embed_dim, hidden_size, padding_idx, device)

    def forward(self, src, tgt):
        output = torch.zeros(tgt.size(0), tgt.size(1), self.vocab_size_tgt)
        # 出力系列長の最大値(EOSが出てこなかった場合)
        generate_size = src.size(1) + 50
        
        hidden_vec = self.encoder(src)
        vocab_vec = self.decoder(tgt, hidden_vec, generate_size)
        
        output = vocab_vec
        return output
    
    def train(self, mode:bool = True):
        self.encoder.train(mode)
        self.decoder.train(mode)