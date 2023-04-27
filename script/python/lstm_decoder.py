import torch.nn as nn

class LSTM_Decoder(nn.Module):
  def __init__(self, vocab_size, embed_dim, hidden_size, padding_idx) -> None:
    super(LSTM_Decoder, self).__init__()
    
    self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx)
    self.lstm_cell = nn.LSTMCell(embed_dim, hidden_size) # LSTMのセル単位で処理を行う
    self.fc = nn.Linear(hidden_size, vocab_size)
  
  def forward(self, inputs, hidden): # inputs:入力文(1文) 並列化は考慮しない hidden:隠れ層
    embedded_vector = self.embedding(inputs.unsqueeze(0))
    output = []
    for token in embedded_vector: # token:1単語の埋め込み行列
      hidden, cell = self.lstm_cell(token, hidden, cell)
      output.append(self.fc(hidden))  
    return output