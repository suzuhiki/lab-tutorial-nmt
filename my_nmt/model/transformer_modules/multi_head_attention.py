import torch
import torch.nn as nn
import sys

class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, head_num, dropout) -> None:
        super(MultiHeadAttention, self).__init__()
        
        self.head_num = head_num
        self.feature_dim = feature_dim
        
        if feature_dim % head_num != 0:
            print("[error] MultiHeadAttentionのhead数は埋め込み次元で割り切れる数にしてください")
            sys.exit()
        self.hidden_dim = int(feature_dim/head_num)
        
        print("feature:" + str(feature_dim) + " hidden:" + str(self.hidden_dim))
        self.linear_Ks = nn.ModuleList([nn.Linear(in_features=feature_dim, out_features=self.hidden_dim, bias=False) for _ in range(head_num)])
        self.linear_Vs = nn.ModuleList([nn.Linear(feature_dim, self.hidden_dim, bias=False) for _ in range(head_num)])
        self.linear_Qs = nn.ModuleList([nn.Linear(feature_dim, self.hidden_dim, bias=False) for _ in range(head_num)])
        
        self.softmax = nn.Softmax(dim = 2)
        self.dropout = nn.Dropout(p=dropout)
        self.linear_out = nn.Linear(feature_dim, feature_dim, bias= False)
    
    # Q,K,V (batch_size, word_len, feature)
    def forward(self, Q, K, V, mask = None):
        
        split_QKVs = self.split_head(Q,K,V)
        
        result = self.attention(split_QKVs)
        
        return result
    
    # x (batch_size, word_len, feature)
    def split_head(self, Q, K, V):
        
        result = []
        # (QKV, head_num, batch_size, word_len, hidden_dim)
        result[0] = torch.zeros(self.head_num, Q.size(0), Q.size(1), self.hidden_dim)
        result[1] = torch.zeros(self.head_num, K.size(0), K.size(1), self.hidden_dim)
        result[2] = torch.zeros(self.head_num, V.size(0), V.size(1), self.hidden_dim)
        
        for i in range(self.head_num):
            result[0][i] = self.linear_Qs[i](Q)
            result[1][i] = self.linear_Ks[i](K)
            result[2][i] = self.linear_Vs[i](V)
        
        return result
    
    # Qs (head_num, batch_size, word_num, hidden_dim) 
    def concat_head(self, Qs):
        result = torch.cat(Qs, 3)
        result = self.linear_out(result)
        
        return result
    
    # QKVs (QKV, head_num, batch_size, word_num, hidden_dim)
    def attention(self, QKVs, mask = None):
        
        # (head_num, batch_size, word_num, hidden_dim)
        Q = QKVs[0]
        K = QKVs[1]
        V = QKVs[2]
        
        # (head_num, batch_size, hidden_dim, word_num)
        K_t = torch.permute(K, (0, 1, 3, 2))
        
        # (head_num, batch_size, word_num(Q), word_num(K))
        QK = torch.matmul(Q, K_t)
        # scaled dot attentionの重み
        QK = QK/(self.hidden_dim ** 0.5)
        
        if mask is not None:
            QK = QK * mask
        
        # word_num(Q)の次元でsoftmax
        softmax_QK = self.softmax(QK)
        softmax_QK = self.dropout(softmax_QK)
        
        # KとVのword_numは同じなので
        # (head_num, batch_size, word_num(Q), hidden_dim)
        QKV = torch.matmul(softmax_QK, V)
        
        # (batch_size, word_num(Q), hidden_dim)
        result = self.concat_head(QKV)
        return result
    
    