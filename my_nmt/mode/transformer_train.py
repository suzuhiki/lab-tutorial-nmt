from statistics import mean
import torch
from torch import nn as nn
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from ..util.first_debugger import FirstDebugger

def transformer_train(model, train_dataloader, dev_dataloader, optimizer, criterion, epoch_num, device,
               batch_size, tgt_id2w, model_save_span: int, model_save_path, writer: SummaryWriter, max_norm, hidden_dim, warmig_up_step):
    step_num = 1
    scaler = GradScaler()

    for epoch in range(1, epoch_num+1):
        model.train()
        train_loss = torch.tensor(0, dtype=torch.float).to(device)
        bleu_list = []
        
        pbar = tqdm(train_dataloader, ascii=True)
        
        # 学習
        for i, (src, dst) in enumerate(pbar):
            optimizer.zero_grad()
            
            src_tensor = src.clone().detach().to(device)
            dst_tensor = dst.clone().detach().to(device)
            
            with autocast():
                pred = model(src_tensor, dst_tensor[:, :-1])
            
                # 平坦なベクトルに変換
                pred_edit = pred.contiguous().view(-1, pred.shape[-1]).to(device)
                dst_edit = dst[:, 1:].contiguous().view(-1).to(device)
            
                loss = criterion(pred_edit, dst_edit)

            train_loss += loss
            
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
        model.eval()
        loop_count = 0
        valid_loss = torch.tensor(0, dtype=torch.float).to(device)
        for src, dst in dev_dataloader:            
            with torch.no_grad():
                
                # 出力文確認  
                if True:
                    loop_count+=1
                    src_tensor = src.clone().detach().to(device)
                    dst_tensor = dst.clone().detach().to(device)

                    # pred (batch_size, word_len)
                    pred = model(src_tensor, dst_tensor, True)

                    # print(pred)
                    pred_text = []
                    id2w = np.vectorize(lambda id: tgt_id2w[id])
                    for sentence in pred:
                        pred_text.append(id2w(sentence))

                    pred_text_clean = []
                    for sentence in pred_text:
                        temp_list = []
                        for word in sentence:
                            if word != "<bos>" and word != "<pad>" and word != "<eos>":
                                temp_list.append(word)
                        pred_text_clean.append(temp_list)

                    dst_text = id2w(dst.to("cpu").detach().numpy().copy())
                    dst_text_clean = []

                    for sentence in dst_text:
                        tmp_list = []
                        for word in sentence:
                            if word != "<bos>" and word != "<pad>" and word != "<eos>":
                                tmp_list.append(word)
                        dst_text_clean.append(tmp_list)

                    bleu = 0
                    for pred_c, dst_c in zip(pred_text_clean, dst_text_clean):
                        bleu += sentence_bleu([dst_c], pred_c,  smoothing_function=SmoothingFunction().method1)
                        # print("dst:" + "".join(dst_c))
                        # print("pred:" + "".join(pred_c))
                    bleu = bleu / batch_size
                    bleu = bleu * 100
                    bleu_list.append(bleu)
                    print("bleu: {}".format(bleu))
                
                # loss確認
                optimizer.zero_grad()
                src_tensor = src.clone().detach().to(device)
                dst_tensor = dst.clone().detach().to(device)
                pred = model(src_tensor, dst_tensor[:, :-1])
                
                pred_edit = pred.contiguous().view(-1, pred.shape[-1]).to(device)
                dst_edit = dst[:, 1:].contiguous().view(-1).to(device)
        
                loss = criterion(pred_edit, dst_edit)
                valid_loss += loss
        
        valid_loss = valid_loss / len(dev_dataloader)
        
        if epoch % model_save_span == 0:
            torch.save(model.state_dict(), f"{model_save_path}/{epoch}_{mean(bleu_list)}.pth")
        
        print(f"epoch {epoch} in {epoch_num} ---- train loss:{train_loss}, valid loss:{valid_loss} bleu score:{mean(bleu_list)}")
        
        if writer != None:
            writer.add_scalar("loss", train_loss, epoch)
            writer.add_scalar("valid loss", valid_loss, epoch)
            writer.add_scalar("bleu", mean(bleu_list), epoch)