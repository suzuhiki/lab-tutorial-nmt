import torch
import torch.nn as nn
from lstm_encoder import LSTM_Encoder
from lstm_decoder import LSTM_Decoder

class Seq2Seq(nn.Module):
    def __init__(self, hidden_size, vocab_size_src, vocab_size_dst, padding_idx, embed_dim, device) -> None:
        super().__init__()
        
        self.vocab_size_dst = vocab_size_dst
        self.encoder = LSTM_Encoder(vocab_size_src, embed_dim, hidden_size, padding_idx, device)
        self.decoder = LSTM_Decoder(vocab_size_dst, embed_dim, hidden_size, padding_idx, device)

    def forward(self, src, dst):
        output = torch.zeros(dst.size(0), dst.size(1), self.vocab_size_dst)
        # 出力系列長の最大値
        generate_size = src.size(1) + 50
        
        hidden_vec = self.encoder(src)
        vocab_vec = self.decoder(dst, hidden_vec, generate_size)
        
        output = vocab_vec
        return output
    
    def train(self, mode:bool = True):
        self.encoder.train(mode)
        self.decoder.train(mode)