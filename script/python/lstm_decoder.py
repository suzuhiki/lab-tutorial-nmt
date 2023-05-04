import torch.nn as nn
import torch

class LSTM_Decoder(nn.Module):
  def __init__(self, vocab_size, embed_dim, hidden_size, padding_idx, device) -> None:
    super(LSTM_Decoder, self).__init__()
    
    self.special_token = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
    self.vocab_size = vocab_size
    self.padding_idx = padding_idx
    self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx)
    self.lstm_cell = nn.LSTMCell(embed_dim, hidden_size) # LSTMのセル単位で処理を行う
    self.fc = nn.Linear(hidden_size, vocab_size)
  
  def forward(self, inputs, encoder_state, generate_len):
    hidden, cell = encoder_state
    
    if self.training == True:
      output = torch.zeros(inputs.size(1) ,inputs.size(0), self.vocab_size)
      embedded_vector = self.embedding(inputs)
      permuted_vec = torch.permute(embedded_vector, (1, 0, 2))
      
      for i in range(permuted_vec.size(0)):
        hidden, cell = self.lstm_cell(permuted_vec[i], (hidden, cell))
        output[i] = self.fc(hidden)
      
      return torch.permute(output, (1, 0, 2)) 
    
    else:
      output = torch.zeros(generate_len, self.vocab_size)
      output_tmp = torch.tensor([self.special_token["<BOS>"]])
      output_tmp = self.embedding(output_tmp)
      
      embed_eos = self.embedding(self.special_token["<EOS>"])
      
      for i in range(generate_len):
        hidden, cell = self.lstm_cell(output_tmp.unsqueeze(0), (hidden, cell))
        output_tmp = self.fc(hidden)
        
        output[i] = self.fc(output_tmp)
        
        if output_tmp == embed_eos:
          break
      return output