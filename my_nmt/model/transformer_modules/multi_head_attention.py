import torch
import torch.nn as nn
import sys

class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, head_num, dropout, device) -> None:
        super(MultiHeadAttention, self).__init__()
        
        self.head_num = head_num
        self.feature_dim = feature_dim
        self.device = device
        
        if feature_dim % head_num != 0:
            print("[error] MultiHeadAttentionのhead数は埋め込み次元を割り切れる数にしてください")
            sys.exit()
        self.hidden_dim = int(feature_dim/head_num)
        
        self.linear_K = nn.Linear(feature_dim, feature_dim)
        self.linear_V = nn.Linear(feature_dim, feature_dim)
        self.linear_Q = nn.Linear(feature_dim, feature_dim)
        
        self.softmax = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(p=dropout)
        self.linear_out = nn.Linear(feature_dim, feature_dim)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.hidden_dim])).to(device)
    
    # query, key, value (batch_size, word_len, feature)
    def forward(self, query, key, value, mask):
        batch_size = query.size(0)
        
        Q = self.linear_Q(query)
        K = self.linear_K(key)
        V = self.linear_V(value)

        # ヘッド数に分割
        Q = Q.view(batch_size, -1, self.head_num, self.hidden_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.head_num, self.hidden_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.head_num, self.hidden_dim).permute(0, 2, 1, 3)
        # 自己注意の計算：attention(Q, K, V) = Softmax(QK/sqrt(d))V
        a_hat = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            a_hat = a_hat.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(a_hat, dim = -1)
        h = torch.matmul(self.dropout(attention), V)
        # マルチヘッドを統合：MultiHead(Q, K, V) = Concat(h1, h2, ...)Wo
        h = h.permute(0, 2, 1, 3).contiguous()
        h = h.view(batch_size, -1, self.feature_dim)
        return self.linear_out(h)
    
    # Q, K, V (batch_size, head_num, word_len, hidden_dim), mask (batch_size, word_len)
    def attention(self, Q, K, V, mask = None):
        
        # (batch_size, head_num, hidden_dim, word_num(K))
        K_t = torch.permute(K, (0, 1, 3, 2))
        
        # (batch_size, head_num, word_num(Q), word_num(K))
        QK = torch.matmul(Q, K_t) / (self.hidden_dim ** 0.5)
        
        if mask is not None:
            QK = QK.masked_fill(mask == 1, -float("inf"))
        
        # word_num(Q)の次元でsoftmax
        softmax_QK = self.softmax(QK)
        
        # KとVのword_numは同じなので
        # (batch_size, head_num, word_num(Q), hidden_dim)
        QKV = torch.matmul(self.dropout(softmax_QK), V)
        
        QKV = torch.tensor_split(QKV, QKV.size(1), dim=1)
        
        # (batch_size, word_num(Q), hidden_dim)
        result = torch.concat(QKV, dim=3).squeeze(1)

        return result