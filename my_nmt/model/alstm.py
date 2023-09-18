import torch
import torch.nn as nn
from .alstm_encoder import ALSTM_Encoder
from .alstm_decoder import ALSTM_Decoder

class ALSTM(nn.Module):
    def __init__(self, hidden_size, vocab_size_src, vocab_size_tgt, padding_idx, embed_dim, device, dropout) -> None:
        super().__init__()
        
        self.vocab_size_tgt = vocab_size_tgt
        self.encoder = ALSTM_Encoder(vocab_size_src, embed_dim, hidden_size, padding_idx, device, dropout)
        self.decoder = ALSTM_Decoder(vocab_size_tgt, embed_dim, hidden_size, padding_idx, device, dropout)

    def forward(self, src, tgt):
        output = torch.zeros(tgt.size(0), tgt.size(1), self.vocab_size_tgt)
        # 出力系列長の最大値(EOSが出てこなかった場合)
        generate_size = src.size(1) + 50
        
        encoder_state = self.encoder(src) # (hiddens[timestep, batch size, hidden size], hidden, cell)
        
        vocab_vec = self.decoder(tgt, encoder_state, generate_size)
        
        output = vocab_vec
        return output
    
    def train(self, mode:bool = True):
        self.encoder.train(mode)
        self.decoder.train(mode)