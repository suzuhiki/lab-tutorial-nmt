from statistics import mean
import torch
from torch import nn as nn
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from ..util.first_debugger import FirstDebugger
from ..model.transformer import Transformer

def transformer_train(model, train_dataloader, dev_dataloader, optimizer, criterion, epoch_num, device,
               batch_size, tgt_id2w, model_save_span: int, model_save_path, writer: SummaryWriter, max_norm, hidden_dim, warmig_up_step, special_token, max_len, tgt_path):
    step_num = 1
    scaler = GradScaler(enabled=False)

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(1, epoch_num+1):
        model.train()
        train_loss = torch.tensor(0, dtype=torch.float).to(device)
        
        pbar = tqdm(train_dataloader, ascii=True)
        
        # 学習
        for i, (src, dst) in enumerate(pbar):
            optimizer.zero_grad()
            
            src_tensor = src.clone().detach().to(device)
            dst_tensor = dst.clone().detach().to(device)
            
            with autocast(enabled=False):
                pred = model(src_tensor, dst_tensor[:, :-1])
            
                # 平坦なベクトルに変換
                pred_edit = pred.contiguous().view(-1, pred.shape[-1]).to(device)
                dst_edit = dst[:, 1:].contiguous().view(-1).to(device)
            
                loss = criterion(pred_edit, dst_edit)

            train_loss += loss.item()
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            lrate = hidden_dim**(-0.5) * min(step_num**(-0.5), step_num * warmig_up_step**(-1.5))
            
            for p in optimizer.param_groups:
                p['lr'] = lrate
            
            step_num += 1
            
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_description("[epoch:%d] loss:%f" % (epoch, train_loss/(i+1)))
        train_loss = train_loss / len(train_dataloader)
        
        # バリデーション
        sentences = transformer_inference(model, dev_dataloader, special_token, device, max_len, tgt_id2w)

        with open(tgt_path, "r") as f:
            lines = f.readlines()
        tgt_dev_word_list = [ line.strip().split(" ") for line in lines ] 
        dev_word_data = [ [words] for words in tgt_dev_word_list ]

        c_bleu = corpus_bleu(dev_word_data, sentences, smoothing_function=SmoothingFunction().method1) * 100
        
        if epoch % model_save_span == 0:
            torch.save(model.state_dict(), f"{model_save_path}/{epoch}_{c_bleu}.pth")
        
        print(f"epoch {epoch} in {epoch_num} ---- train loss:{train_loss}, bleu score:{c_bleu}")
        
        if writer != None:
            writer.add_scalar("loss", train_loss, epoch)
            writer.add_scalar("bleu", c_bleu, epoch)


def transformer_inference(model: Transformer, inf_dataloader, special_token, device, max_len, id2w):

    encoder = model.get_encoder()
    decoder = model.get_decoder()

    encoder.eval()
    decoder.eval()

    pbar = tqdm(inf_dataloader, ascii=True)
    sentence_list = []


    with torch.no_grad():
        for batch in pbar:

            src, _ = batch
            src = src.to(device)
            src_mask = model.make_src_mask(src)
            src_mask = src_mask.to(device)

            batch_size = src.size(0)
            original_indexes = torch.arange(batch_size, device=device)

            generated_words = [[] for _ in range(batch_size)]

            dec_in = torch.full((batch_size, 1), special_token["<bos>"], dtype=torch.int64, device=device)
            eos = torch.full((batch_size, 1), special_token["<eos>"], dtype=torch.int64, device=device)
            enc_state = encoder(src, src_mask)

            for _ in range(max_len):


                output = decoder(enc_state, dec_in, src_mask)

                predicted_words = torch.argmax(output[:, -1:, :], dim=2)

                # EOS以外の単語を次の入力とする
                is_eos = predicted_words.eq(eos).squeeze(1)
                eos_indexes = is_eos.nonzero().squeeze(1).tolist()

                not_eos = predicted_words.ne(eos).squeeze(1)
                not_eos_indexes = not_eos.nonzero().squeeze(1)

                to_original_index = original_indexes.tolist()
                for i in eos_indexes:
                    dict_indexes = dec_in[i, 1:].tolist()
                    words = [id2w[dict_index] for dict_index in dict_indexes]
                    sent_index = to_original_index[i]
                    generated_words[sent_index] = words
                
                # encoderから再出力
                src_mask = src_mask.index_select(dim=0, index = not_eos_indexes)
                enc_state = enc_state.index_select(dim=0, index = not_eos_indexes)
                dec_in = dec_in.index_select(dim=0, index = not_eos_indexes)
                original_indexes = original_indexes.index_select(dim=0, index = not_eos_indexes)
                eos = eos.index_select(dim=0, index = not_eos_indexes)
                predicted_words = predicted_words.index_select(dim=0, index = not_eos_indexes)

                if dec_in.size(0) == 0:
                    break

                dec_in = torch.cat([dec_in, predicted_words], dim=1)
            
            # ループを終えてもEOSが出てこなかった場合
            sent_num = dec_in.size(0)
            if sent_num != 0:
                to_original_index = original_indexes.tolist()
                for i in range(sent_num):
                    dict_indexes = dec_in[i, 1:].tolist()
                    words = [id2w[dict_index] for dict_index in dict_indexes]
                    sent_index = to_original_index[i]
                    generated_words[sent_index] = words
            
            for sentence in generated_words:
                sentence_list.append(sentence)


    return sentence_list