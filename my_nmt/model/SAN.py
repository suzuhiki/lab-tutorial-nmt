import torch
import torch.nn as nn


# Encoder

class Encoder(nn.Module):

    def __init__(self, n_words_src, n_embed, n_layers, n_heads, n_hidden, dropout, max_length, device):
        super(Encoder, self).__init__()
        self.tok_embedding = nn.Embedding(num_embeddings=n_words_src, embedding_dim=n_embed)
        self.pos_embedding = nn.Embedding(num_embeddings=max_length, embedding_dim=n_embed)
        self.layers = nn.ModuleList([EncoderLayer(n_embed, n_heads, n_hidden, dropout, device) for _ in range(n_layers)])
        # その他
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([n_embed])).to(device)

    def forward(self, src, src_mask):
        batch_size, src_len = src.shape
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = self.dropout((self.tok_embedding(src.to(self.device)) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class EncoderLayer(nn.Module):

    def __init__(self, n_embed, n_heads, n_hidden, dropout, device):
        super(EncoderLayer, self).__init__()
        self.self_attn_ln = nn.LayerNorm(n_embed)
        self.ff_ln = nn.LayerNorm(n_embed)
        self.self_attention = MultiHeadAttentionLayer(n_embed, n_heads, dropout, device)
        self.feedforward = FeedforwardLayer(n_embed, n_hidden, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        # self-attention
        _src = self.self_attention(src, src, src, src_mask)
        # add & norm
        src = self.self_attn_ln(src + self.dropout(_src))
        # feedforward
        _src = self.feedforward(src)
        # add & norm
        src = self.ff_ln(src + self.dropout(_src))
        return src
    
    
    # Main Block

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, n_embed, n_heads, dropout, device):
        super(MultiHeadAttentionLayer, self).__init__()
        # ベクトルがヘッド数で割り切れるか確認
        assert n_embed % n_heads == 0
        # ハイパーパラメータ
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        # 線形変換
        self.fc_q = nn.Linear(n_embed, n_embed)
        self.fc_k = nn.Linear(n_embed, n_embed)
        self.fc_v = nn.Linear(n_embed, n_embed)
        self.fc_o = nn.Linear(n_embed, n_embed)
        # その他
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        self.device = device

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        # クエリ・キュー・バリューの作成
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # ヘッド数に分割
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # 自己注意の計算：attention(Q, K, V) = Softmax(QK/sqrt(d))V
        a_hat = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            a_hat = a_hat.masked_fill(mask.to(self.device) == 0, -1e10)
        attention = torch.softmax(a_hat, dim = -1)
        h = torch.matmul(self.dropout(attention), V)
        # マルチヘッドを統合：MultiHead(Q, K, V) = Concat(h1, h2, ...)Wo
        h = h.permute(0, 2, 1, 3).contiguous()
        h = h.view(batch_size, -1, self.n_embed)
        return self.fc_o(h)


class FeedforwardLayer(nn.Module):

    def __init__(self, n_embed, n_hidden, dropout):
        super(FeedforwardLayer, self).__init__()
        self.fc_1 = nn.Linear(n_embed, n_hidden)
        self.fc_2 = nn.Linear(n_hidden, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc_2(self.dropout(torch.relu(self.fc_1(x))))
    
# Decoder

class Decoder(nn.Module):

    def __init__(self, n_words_tgt, n_embed, n_layers, n_heads, n_hidden, dropout, max_length, device):
        super(Decoder, self).__init__()
        self.tok_embedding = nn.Embedding(n_words_tgt, n_embed)
        self.pos_embedding = nn.Embedding(max_length, n_embed)
        self.layers = nn.ModuleList([DecoderLayer(n_embed, n_heads, n_hidden, dropout, device) for _ in range(n_layers)])
        self.fc_out = nn.Linear(n_embed, n_words_tgt)
        # その他
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([n_embed])).to(device)
        
    def forward(self, tgt, enc_src, tgt_mask, src_mask):
        batch_size, tgt_len = tgt.shape
        pos = torch.arange(0, tgt_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        tgt = self.dropout((self.tok_embedding(tgt.to(self.device)) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            tgt = layer(tgt, enc_src, tgt_mask, src_mask)
        return self.fc_out(tgt)


class DecoderLayer(nn.Module):

    def __init__(self, n_embed, n_heads, n_hidden, dropout, device):
        super(DecoderLayer, self).__init__()
        self.self_attn_ln = nn.LayerNorm(n_embed)
        self.enc_attn_ln = nn.LayerNorm(n_embed)
        self.ff_ln = nn.LayerNorm(n_embed)
        self.self_attention = MultiHeadAttentionLayer(n_embed, n_heads, dropout, device)
        self.enc_attention = MultiHeadAttentionLayer(n_embed, n_heads, dropout, device)
        self.feedforward = FeedforwardLayer(n_embed, n_hidden, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, enc_src, tgt_mask, src_mask):
        # self-attention
        _tgt = self.self_attention(tgt, tgt, tgt, tgt_mask)
        # add & norm
        tgt = self.self_attn_ln(tgt + self.dropout(_tgt))
        # enc-attention
        _tgt = self.enc_attention(tgt, enc_src, enc_src, src_mask)
        # add & norm
        tgt = self.enc_attn_ln(tgt + self.dropout(_tgt))
        # feedforward
        _tgt = self.feedforward(tgt)
        # add & norm
        tgt = self.ff_ln(tgt + self.dropout(_tgt))
        return tgt
    

class SAN_NMT(nn.Module):

    def __init__(self, encoder, decoder, src_pad_idx, tgt_pad_idx, device):
        super(SAN_NMT, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device
    
    # マスク：<pad>トークンでなければ1, <pad>トークンなら0 -> 実際の単語がない部分を計算時に無視する
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    # マスク：ソースと同様の<pad>に対するマスクに加え、未来の情報もマスクする
    def make_tgt_mask(self, tgt):
        tgt_len = tgt.shape[1]
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2).to(self.device)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device = self.device)).bool().to(self.device)
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask

    def forward(self, src, tgt):
        # マスク
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        # エンコード
        enc_src = self.encoder(src, src_mask)
        # デコード
        output = self.decoder(tgt, enc_src, tgt_mask, src_mask)
        return output