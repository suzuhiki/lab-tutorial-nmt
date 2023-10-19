import torch
import torch.nn as nn
import sys
from .transformer_modules.positinal_encoding import PositionalEncoding
from .transformer_modules.decoder_block import DecoderBlock

class TransformerDecoder(nn.Module):
    def __init__(self, dec_vocab_dim, feature_dim, dropout, head_num, special_token, max_len, device, ff_hidden_dim = 2048, block_num = 6) -> None:
        super(TransformerDecoder, self).__init__()
        
        self.device = device
        self.special_token = special_token
        self.dec_vocab_dim = dec_vocab_dim
        self.feature_dim = feature_dim
        self.embed = nn.Embedding(dec_vocab_dim, feature_dim, special_token["<pad>"])
        self.pos_enc = PositionalEncoding(feature_dim, dropout, device, max_len)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(feature_dim, head_num, dropout, ff_hidden_dim, device) for _ in range(block_num)])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(feature_dim, dec_vocab_dim)
        self.softmax = nn.Softmax(dim=2)
        self.scale = torch.sqrt(torch.FloatTensor([feature_dim])).to(device)
        
    def forward(self, enc_state, decoder_input, src_mask, max_len, inference_mode = False):
        
        if self.training == True:
            tgt_mask = self.make_target_mask(decoder_input)
            return self.decoder_process(decoder_input, enc_state, tgt_mask, src_mask)
        
        # 推論時 一文ずつ
        else:
            
            if inference_mode == True:
                batch_size = enc_state.size(0)
                pred_seqs = []

                for i in range(batch_size):
                    pred_seq = [self.special_token["<bos>"]]

                    for _ in range(max_len):
                        tgt_in = torch.LongTensor(pred_seq).unsqueeze(0).to(self.device)
                        tgt_mask = self.make_target_mask(tgt_in)
                        
                        with torch.no_grad():
                            output = self.decoder_process(tgt_in, enc_state[i].unsqueeze(0), tgt_mask, src_mask[i])

                        pred_token_id = output.argmax(dim = 2)[:,-1].item()
                        pred_seq.append(pred_token_id)

                        if pred_token_id == self.special_token["<eos>"]:
                            break
                    pred_seqs.append(pred_seq)
                return pred_seqs

            else:
                tgt_mask = self.make_target_mask(decoder_input)
                return self.decoder_process(decoder_input, enc_state, tgt_mask, src_mask)


    def make_target_mask(self, decoder_input):
        pad_mask = torch.where(decoder_input == self.special_token["<pad>"], 1, 0).unsqueeze(1).unsqueeze(2)
        # print("pad_mask: {}".format(pad_mask[0]))
        tgt_mask = torch.ones((pad_mask.size(-1), pad_mask.size(-1))).triu(diagonal=1).to(self.device)
        # print("tgt_mask: {}".format(tgt_mask[0]))
        tgt_mask = pad_mask + tgt_mask
        tgt_mask = torch.where(tgt_mask >= 1, 1, 0)
        # print("mask: {}".format(tgt_mask[0]))
        
        return tgt_mask
    
    def decoder_process(self, input, enc_state, tgt_mask, src_mask):

        x = self.embed(input)

        x = x*self.scale

        x = self.pos_enc(x)

        x = self.dropout(x)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, enc_state, tgt_mask, src_mask)
        x = self.linear(x)
        
        return x