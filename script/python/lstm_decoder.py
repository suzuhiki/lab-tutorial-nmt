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
  
  def forward(self, inputs, encoder_state, generate_len): # inputs:入力文(1文) 並列化は考慮しない encoder_state:(hidden,cell)
    s_mask = torch.where(inputs == self.padding_idx, 1, 0)
    hidden, cell = encoder_state
    
    if self.training == True:
      output = torch.zeros(inputs.size(0), self.vocab_size)
      embedded_vector = self.embedding(inputs)
      
      for i, (token, w_mask) in enumerate(zip(embedded_vector, s_mask)): # token:1単語の埋め込み行列
        hidden_tmp, cell_tmp = self.lstm_cell(token.unsqueeze(0), (hidden, cell))
        
        output[i] = self.fc(hidden_tmp)  
        
        if w_mask == 0:
          hidden, cell = hidden_tmp, cell_tmp
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