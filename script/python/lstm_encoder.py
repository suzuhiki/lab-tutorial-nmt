import torch.nn as nn
import torch

class LSTM_Encoder(nn.Module):
  def __init__(self, vocab_size, embed_dim, hidden_size, padding_idx) -> None:
    super(LSTM_Encoder, self).__init__()
    
    self.padding_idx = padding_idx
    self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx)
    self.lstm_cell = nn.LSTMCell(embed_dim, hidden_size) # LSTMのセル単位で処理を行う
  
  def forward(self, inputs): # inputs:入力文(1文) 並列化は考慮しない mask:マスク行列(PADのワンホット)
    s_mask = torch.where(inputs != self.padding_idx, 1, 0)
    embedded_vector = self.embedding(inputs.unsqueeze(0))
    hiddens = []
    
    for token, w_mask in zip(embedded_vector, s_mask): # token:1単語の埋め込み行列
      if w_mask == 1:
        continue
        
      hidden, cell = self.lstm_cell(token, hidden, cell)
      hiddens.append(hidden)
    
    return hiddens[-1]