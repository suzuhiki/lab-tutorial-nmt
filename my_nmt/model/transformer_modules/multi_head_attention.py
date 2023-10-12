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
            print("[error] MultiHeadAttentionのhead数は埋め込み次元で割り切れる数にしてください")
            sys.exit()
        self.hidden_dim = int(feature_dim/head_num)
        
        self.linear_Ks = nn.ModuleList([nn.Linear(in_features=feature_dim, out_features=self.hidden_dim, bias=False) for _ in range(head_num)])
        self.linear_Vs = nn.ModuleList([nn.Linear(feature_dim, self.hidden_dim, bias=False) for _ in range(head_num)])
        self.linear_Qs = nn.ModuleList([nn.Linear(feature_dim, self.hidden_dim, bias=False) for _ in range(head_num)])
        
        self.softmax = nn.Softmax(dim = 2)
        self.dropout = nn.Dropout(p=dropout)
        self.linear_out = nn.Linear(feature_dim, feature_dim, bias= False)
    
    # Q,K,V (batch_size, word_len, feature)
    def forward(self, Q, K, V, mask):
        
        # print("Q: {}".format(Q.size()))
        # print(Q)

        # print("K: {}".format(K.size()))
        # print(K)

        # print("V: {}".format(V.size()))
        # print(V)



        split_QKVs = self.split_head(Q,K,V)
        # print("split_QKVs: {}".format(split_QKVs[0].size()))

        result = self.attention(split_QKVs, mask)
        
        return result
    
    # x (batch_size, word_len, feature)
    def split_head(self, Q, K, V):
        
        result = []
        # (QKV, head_num, batch_size, word_len, hidden_dim)
        result.append(torch.zeros(self.head_num, Q.size(0), Q.size(1), self.hidden_dim))
        result.append(torch.zeros(self.head_num, K.size(0), K.size(1), self.hidden_dim))
        result.append(torch.zeros(self.head_num, V.size(0), V.size(1), self.hidden_dim))
        
        for i in range(self.head_num):
            result[0][i] = self.linear_Qs[i](Q)
            result[1][i] = self.linear_Ks[i](K)
            result[2][i] = self.linear_Vs[i](V)
        
        return result
    
    # Qs (head_num, batch_size, word_num, hidden_dim) 
    def concat_head(self, Qs):
        result = torch.cat([Qs[i] for i in range(self.head_num)], dim=2).to(self.device)
        result = self.linear_out(result)
        
        return result
    
    # QKVs (QKV, head_num, batch_size, word_num, hidden_dim), mask (batch_size, word_len)
    def attention(self, QKVs, mask):
        
        # (head_num, batch_size, word_num, hidden_dim)
        Q = QKVs[0]
        K = QKVs[1]
        V = QKVs[2]
        
        # (head_num, batch_size, hidden_dim, word_num(K))
        K_t = torch.permute(K, (0, 1, 3, 2))
        
        # (head_num, batch_size, word_num(Q), word_num(K))
        QK = torch.matmul(Q, K_t)
        # scaled dot attentionの重み
        QK = (QK/(self.hidden_dim ** 0.5)).to(self.device)
        
        attntion_mask = torch.where(mask == 0, 0, -sys.maxsize).to(self.device)
        attntion_mask = attntion_mask.unsqueeze(-1)
        sized_mask = torch.zeros(QK[0].size()).to(self.device)
        sized_mask = (sized_mask + attntion_mask).to(self.device)

        QK = QK + attntion_mask
        
        # word_num(Q)の次元でsoftmax
        softmax_QK = self.softmax(QK)
        softmax_QK = self.dropout(softmax_QK).to(self.device)
        
        # KとVのword_numは同じなので
        # (head_num, batch_size, word_num(Q), hidden_dim)
        QKV = torch.matmul(softmax_QK, V.to(self.device))
        
        # (batch_size, word_num(Q), hidden_dim)
        result = self.concat_head(QKV)
        return result
    
    