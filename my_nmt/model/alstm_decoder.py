import torch.nn as nn
import torch

class ALSTM_Decoder(nn.Module):
	def __init__(self, vocab_size, embed_dim, hidden_size, padding_idx, device, dropout) -> None:
		super(ALSTM_Decoder, self).__init__()
		
		self.device = device
		self.special_token = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
		self.vocab_size = vocab_size
		self.padding_idx = padding_idx
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx)
		self.lstm_cell = nn.LSTMCell(embed_dim, hidden_size) # LSTMのセル単位で処理を行う
		self.fc2 = nn.Linear(hidden_size, vocab_size)
		self.dropout = nn.Dropout(p=dropout)
		self.softmax = nn.Softmax(dim=1)
		self.softmax_0 = nn.Softmax(dim=0)
		self.fc1 = nn.Linear(hidden_size*2, hidden_size)
	
	def forward(self, inputs, encoder_state, generate_len): # inputs[batch][timestep]
		hiddens, hidden, cell = encoder_state

		# →hiddens(batchsize × hidden × timestep)
		hiddens_t = torch.permute(hiddens, (1, 2, 0))
		# →hiddens(batchsize × timestep × hidden)
		hiddens = torch.permute(hiddens, (1, 0, 2))
	

		if self.training == True:
			output = torch.zeros(inputs.size(1) ,inputs.size(0), self.vocab_size)
			
			# inputs[batch][timestep]に格納されたword idに対してemmbeddingを実施
			embedded_vector = self.dropout(self.embedding(inputs)) # (batch, timestep, vocab)
			
			permuted_vec = torch.permute(embedded_vector, (1, 0, 2))
			
			# timestepに対してループ
			for i in range(permuted_vec.size(0)):
				hidden, cell = self.lstm_cell(permuted_vec[i], (hidden, cell))

				# 内積計算のためにhiddenを2次元化
				hidden_2d = hidden.unsqueeze(dim=2)
    
				# ((batch) × timestep × hidden)dot((batch) × hidden × 1) → a((batch) × timestep × 1)
				a = torch.bmm(hiddens, hidden_2d)
				a = self.softmax(a)

				# ((batch) × hidden × timestep) dot ((batch) × timestep × 1) → c((batch) × hidden × 1)
				c = torch.bmm(hiddens_t, a)
				c = c.squeeze()
				
				# hとcを結合 → bar_h((batch) × hidden*2) ※一次元
				bar_h = torch.cat((hidden, c), dim=1)

				# bar_h(hidden*2) →fc1→ (hidden) → tanh(活性化) →fc2→ (vocab size)  
				output[i] = self.fc2(torch.tanh(self.dropout(self.fc1(bar_h))))
			
			return torch.permute(output, (1, 0, 2)) # (batch, timestep, vocab)
		
		else:
			# output = torch.zeros(generate_len, self.vocab_size, device=self.device)
			output = []
			
			# 1文章ずつ処理
			for i, (h, c, hs) in enumerate(zip(hidden, cell, hiddens)):
				sentence_out = []
				
				h_x = h.unsqueeze(dim=0)
				c_x = c.unsqueeze(dim=0)
				
				input_tmp = torch.tensor([self.special_token["<BOS>"]], device=self.device)
				input_tmp = self.embedding(input_tmp)
				# input_tmp[vocab]
				
				for j in range(generate_len):
					h_x ,c_x =self.lstm_cell(input_tmp, (h_x, c_x))

					# (timestep × hidden)dot(hidden × 1) → a(timestep × 1)
					a = torch.mm(hs, h_x.t())
					a = self.softmax_0(a)

					# (hidden × timestep) dot (timestep × 1) → c(hidden × 1)
					c = torch.mm(torch.t(hs), a)
					c = c.squeeze()

					# hとcを結合 → bar_h(hidden*2) ※一次元
					bar_h = torch.cat((h_x.squeeze(), c))

					# bar_h(hidden*2) →fc1→ (hidden) → tanh(活性化) →fc2→ (vocab size)
					output_tmp = self.fc2(torch.tanh(self.fc1(bar_h)))

					# 貪欲法で出力単語を決定
					output_tmp = torch.argmax(output_tmp)

					sentence_out.append(output_tmp.to("cpu").detach().numpy().copy().item())
					
					if output_tmp == self.special_token["<EOS>"]:
						break
					
					# 出力単語をembeddingして次の単語の予測に使う
					input_tmp = self.embedding(output_tmp).unsqueeze(dim=0)
				output.append(sentence_out)

			return output