import torch.nn as nn
import torch

class LSTM_Encoder(nn.Module):
  def __init__(self, vocab_size, embed_dim, hidden_size, padding_idx, device) -> None:
    super(LSTM_Encoder, self).__init__()
    
    self.device = device
    self.hidden_size = hidden_size
    self.padding_idx = padding_idx
    self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx)
    self.lstm_cell = nn.LSTMCell(embed_dim, hidden_size) # LSTMのセル単位で処理を行う
  
  def forward(self, inputs): #inputs : (batch, timestep)
    s_mask = torch.where(inputs == self.padding_idx, 1, 0)
    s_mask = torch.permute(s_mask, (1, 0))
    
    h_c_mask = torch.zeros(inputs.size(1), inputs.size(0), self.hidden_size, device=self.device)
    
    for i, timestep in enumerate(s_mask):
      for j, batch in enumerate(timestep):
        if batch == 1:
          h_c_mask[i][j] = torch.zeros(self.hidden_size, device=self.device)
    
    embedded_vector = self.embedding(inputs) # (batch, timestep, vocab)

    hidden = torch.zeros(inputs.size(0),self.hidden_size, device=self.device)
    cell = torch.zeros(inputs.size(0), self.hidden_size, device=self.device)

    permuted_vec = torch.permute(embedded_vector, (1, 0, 2))
    
    for i in range(permuted_vec.size(0)):
      tmp_hidden, tmp_cell = self.lstm_cell(permuted_vec[i], (hidden, cell))
      hidden = torch.where(h_c_mask[i] == 0, tmp_hidden, hidden)
      cell = torch.where(h_c_mask[i] == 0, tmp_cell, cell)
    
    return (hidden, cell)