from statistics import mean
import torch
from torch import nn as nn
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from ..util.first_debugger import FirstDebugger

def transformer_train(model, train_dataloader, dev_dataloader, optimizer, criterion, epoch_num, device,
               batch_size, tgt_id2w, model_save_span: int, model_save_path, writer: SummaryWriter):
    for epoch in range(1, epoch_num+1):
        model.train()
        epoch_loss = torch.tensor(0, dtype=torch.float).to(device)
        bleu_list = []
        
        for src, dst in tqdm(train_dataloader):
            optimizer.zero_grad()
            
            src_tensor = src.clone().detach().to(device)
            dst_tensor = dst.clone().detach().to(device)

            pred = model(src_tensor, dst_tensor)
            
            loss = torch.tensor(0, dtype=torch.float).to(device)
            for s_pred, s_dst in zip(pred, dst):
                dst_edit = torch.cat((s_dst[1:], torch.zeros(1, dtype=torch.int32))).to(device)
                loss += criterion(s_pred, dst_edit)

            epoch_loss += loss
            loss.backward()
            optimizer.step()
                

        # バリデーション
        model.eval()
        for src, dst in dev_dataloader:            
            with torch.no_grad():
                src_tensor = src.clone().detach().to(device)
                dst_tensor = dst.clone().detach().to(device)
                
                # pred (batch_size, word_len)
                pred = model(src_tensor, dst_tensor)
                pred = pred.to("cpu").detach().numpy().copy()
                
                pred_text = []
                id2w = np.vectorize(lambda id: tgt_id2w[id])
                for sentence in pred:
                    pred_text.append(id2w(sentence)) 
                
                dst_text = id2w(dst.to("cpu").detach().numpy().copy())
                dst_text_clean = []
                
                for sentence in dst_text:
                    tmp_list = []
                    for word in sentence:
                        if word != "<bos>" and word != "<pad>":
                            tmp_list.append(word)
                    dst_text_clean.append(tmp_list)
                
                
                
                bleu = 0
                for pred_c, dst_c in zip(pred_text, dst_text_clean):
                    bleu += sentence_bleu([dst_c], pred_c,  smoothing_function=SmoothingFunction().method1)
                    # print("dst:" + "".join(dst_c))
                    # print("pred" + "".join(pred_c))
                bleu = bleu / batch_size
                bleu_list.append(bleu)
                print("bleu: {}".format(bleu))
                
        
        if epoch % model_save_span == 0:
            torch.save(model.state_dict(), f"{model_save_path}/{epoch}_{mean(bleu_list)}.pth")
        
        print(f"epoch {epoch} in {epoch_num} ---- epoch loss:{epoch_loss}, bleu score:{mean(bleu_list)}")
        
        if writer != None:
            writer.add_scalar("loss", epoch_loss, epoch)
            writer.add_scalar("bleu", mean(bleu_list), epoch)