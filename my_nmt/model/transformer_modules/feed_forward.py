import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dropout, feature_dim, ff_hidden_dim = 2048) -> None:
        super(FeedForward, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(feature_dim, ff_hidden_dim)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(ff_hidden_dim, feature_dim)
        
    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        
        return x