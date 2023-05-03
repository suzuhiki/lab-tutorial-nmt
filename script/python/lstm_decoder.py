import torch.nn as nn
import torch

class LSTM_Decoder(nn.Module):
  def __init__(self, vocab_size, embed_dim, hidden_size, padding_idx) -> None:
    super(LSTM_Decoder, self).__init__()
    
    self.vocab_size = vocab_size
    self.padding_idx = padding_idx
    self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx)
    self.lstm_cell = nn.LSTMCell(embed_dim, hidden_size) # LSTMのセル単位で処理を行う
    self.fc = nn.Linear(hidden_size, vocab_size)
  
  def forward(self, inputs, encoder_state): # inputs:入力文(1文) 並列化は考慮しない encoder_state:(hidden,cell)
    s_mask = torch.where(inputs == self.padding_idx, 1, 0)
    hidden, cell = encoder_state
    output = torch.zeros(inputs.size(0), self.vocab_size)
    
    if self.training == True:
      embedded_vector = self.embedding(inputs)
      
      for i, (token, w_mask) in enumerate(zip(embedded_vector, s_mask)): # token:1単語の埋め込み行列
        hidden_tmp, cell_tmp = self.lstm_cell(token.unsqueeze(0), (hidden, cell))
        
        output[i] = self.fc(hidden_tmp)  
        
        if w_mask == 0:
          hidden, cell = hidden_tmp, cell_tmp
    return output