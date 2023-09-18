import torch.nn as nn
import torch

class ALSTM_Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, padding_id, device, dropout) -> None:
        super(ALSTM_Encoder, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.padding_id = padding_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_id)
        self.lstm_cell = nn.LSTMCell(embed_dim, hidden_size) # LSTMのセル単位で処理を行う
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs) -> (torch.Tensor, torch.Tensor): # inputs[batch][timestep]
        
        # maskが1のとき
        mask = torch.where(inputs == self.padding_id, 1, 0)

        # (timestep, batch)
        mask = torch.permute(mask, (1, 0))
        
        # forループにおいて、hiddenの各要素に対してマスクをかけるために新しいtensorを作成
        h_c_mask = torch.zeros(inputs.size(1), inputs.size(0), self.hidden_size, device=self.device)
    
        for i, timestep in enumerate(mask):
            for j, batch in enumerate(timestep):
                if batch == 1:
                    h_c_mask[i][j] = torch.ones(self.hidden_size, device=self.device)

        # inputs[batch][timestep]に格納されたword idに対してemmbeddingを実施
        embedded_vector = self.dropout(self.embedding(inputs)) # (batch, timestep, vocab)

        permuted_vec = torch.permute(embedded_vector, (1, 0, 2)) # (timestep, batch, vocab)

        # hiddenとcellの初期値
        hidden = torch.zeros(inputs.size(0), self.hidden_size, device=self.device)
        cell = torch.zeros(inputs.size(0), self.hidden_size, device=self.device)
        
        # attentionに利用するhiddenたち (timestep, batch size, hidden size)
        hiddens = torch.zeros(inputs.size(1), inputs.size(0), self.hidden_size, device=self.device)

        # timestepでループ
        for i in range(permuted_vec.size(0)):
            tmp_hidden, tmp_cell = self.lstm_cell(permuted_vec[i], (hidden, cell))
            
            # <pad>の場合はhiddenとcellを更新しない
            hidden = torch.where(h_c_mask[i] == 0, tmp_hidden, hidden)
            cell = torch.where(h_c_mask[i] == 0, tmp_cell, cell)
            
            # <pad>の場合は対応するhiddensをzerosのままにする
            hiddens[i] = torch.where(h_c_mask[i] == 0, hidden, torch.zeros(inputs.size(0), self.hidden_size, device=self.device))

        return (hiddens, hidden, cell)