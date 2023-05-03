import torch.nn as nn
import torch

class LSTM_Decoder(nn.Module):
  def __init__(self, vocab_size, embed_dim, hidden_size, padding_idx) -> None:
    super(LSTM_Decoder, self).__init__()
    
    self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx)
    self.lstm_cell = nn.LSTMCell(embed_dim, hidden_size) # LSTMのセル単位で処理を行う
    self.fc = nn.Linear(hidden_size, vocab_size)
  
  def forward(self, inputs, hidden): # inputs:入力文(1文) 並列化は考慮しない hidden:隠れ層
    s_mask = torch.where(inputs != self.padding_idx, 1, 0)
    
    if self.training == True:
      embedded_vector = self.embedding(inputs.unsqueeze(0))
      output = []
      for token, w_mask in zip(embedded_vector, s_mask): # token:1単語の埋め込み行列
        hidden_tmp, cell_tmp = self.lstm_cell(token, hidden, cell)
        
        if w_mask == 0:
          hidden, cell = hidden_tmp, cell_tmp
          
        output.append(self.fc(hidden_tmp))  
      return output