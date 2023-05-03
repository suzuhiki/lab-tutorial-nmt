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
        
        for i, (s, d) in enumerate(zip(src, dst)):
            hidden_vec = self.encoder(s)
            vocab_vec = self.decoder(d, hidden_vec)
            
            output[i] = vocab_vec
        return output
    
    def train(self, mode:bool = True):
        self.encoder.train(mode)
        self.decoder.train(mode)