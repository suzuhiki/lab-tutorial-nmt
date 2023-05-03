import torch
import torch.nn as nn
from lstm_encoder import LSTM_Encoder
from lstm_decoder import LSTM_Decoder

class Seq2Seq(nn.Module):
    def __init__(self, hidden_size, vocab_size_src, vocab_size_dst, padding_idx, embed_dim) -> None:
        super().__init__()

        self.encoder = LSTM_Encoder(vocab_size_src, embed_dim, hidden_size, padding_idx)
        self.decoder = LSTM_Decoder(vocab_size_dst, embed_dim, hidden_size, padding_idx)

    def forward(self, inputs):
        output = []
        
        for src, dst in inputs:
            print(src)
            print(dst)
            hidden_vec = self.encoder(src)
            vocab_vec = self.decoder(dst, hidden_vec)
            output.append(vocab_vec)
        return output
    
    def train(self, mode:bool = True):
        self.encoder.train(mode)
        self.decoder.train(mode)