import torch
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from statistics import mean

def SAN_train(model, max_epoch, dataloader_train, dataloader_val, optimizer, criterion, writer, device, model_save_span: int, model_save_path, special_token, max_len, tgt_id2w):
    best_valid_loss = float("inf")
    for epoch in range(1, max_epoch+1):
        train_loss = train(model, dataloader_train, optimizer, criterion, device)
        val_loss = valid(model, dataloader_val, criterion, special_token, device, max_len, tgt_id2w, writer, epoch)
        # モデルの保存
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
        # 損失の表示
        print("Epoch: %02d\tTrain Loss = %.3f\tValid Loss = %.3f" % (epoch+1, train_loss, val_loss))
    
        if writer != None:
            writer.add_scalar("loss", train_loss, epoch)
            writer.add_scalar("valid loss", val_loss, epoch)
            
        if epoch % model_save_span == 0:
            torch.save(model.state_dict(), f"{model_save_path}/{epoch}_{val_loss}.pth")

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for (src, tgt) in tqdm(dataloader):
        # 推論の準備
        optimizer.zero_grad()
        # 推論
        pred = model(src, tgt[:,:-1])
        # 損失計算の準備
        pred = pred.contiguous().view(-1, pred.shape[-1]).to(device)
        tgt = tgt[:,1:].contiguous().view(-1).to(device)
        
        # 損失計算
        loss = criterion(pred, tgt)
        epoch_loss += loss.item()
        # 逆伝播
        loss.backward()
        optimizer.step()
    return epoch_loss / len(dataloader)


def valid(model, dataloader, criterion, special_token, device, max_len, tgt_id2w, writer, epoch):
    model.eval()
    epoch_loss = 0
    loop_count = 0
    for (src, tgt) in tqdm(dataloader):
        bleu_list = []
        # 出力文作成
        if loop_count == 0:
            loop_count += 1
            pred = []
            for i in range(src.size(0)):
                src_mask = model.make_src_mask(src[i].unsqueeze(0))
                with torch.no_grad():
                    enc_state = model.encoder(src[i].unsqueeze(0), src_mask)
                
                tgt_indexes = [special_token["<bos>"]]
                for _ in range(max_len):
                    tgt_tensor = torch.LongTensor(tgt_indexes).unsqueeze(0).to(device)
                    tgt_mask = model.make_tgt_mask(tgt_tensor)
                    with torch.no_grad():
                        output = model.decoder(tgt_tensor, enc_state, tgt_mask, src_mask)
                    pred_token_id = output.argmax(dim=2)[:,-1].item()
                    tgt_indexes.append(pred_token_id)
                    if pred_token_id == special_token["<eos>"]:
                        break
                pred.append(tgt_indexes)
                
            id2w = np.vectorize(lambda id: tgt_id2w[id])
            
            pred_text = []
            for sentence in pred:
                pred_text.append(id2w(sentence))
            
            pred_text_clean = []
            for sentence in pred_text:
                temp_list = []
                for word in sentence:
                    if word != "<bos>" and word != "<pad>" and word != "<eos>":
                        temp_list.append(word)
                pred_text_clean.append(temp_list)
            
            dst_text = id2w(tgt.to("cpu").detach().numpy().copy())
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
                print("dst:" + "".join(dst_c))
                print("pred:" + "".join(pred_c))
            bleu = bleu / src.size(0)
            bleu = bleu * 100
            bleu_list.append(bleu)
            print("bleu: {}".format(bleu))
            
            if writer != None:
                writer.add_scalar("bleu", mean(bleu_list), epoch)
        
        # 推論の準備
        # 推論
        pred = model(src, tgt[:,:-1])
        # 損失計算の準備
        pred = pred.contiguous().view(-1, pred.shape[-1]).to(device)
        tgt = tgt[:,1:].contiguous().view(-1).to(device)
        # 損失計算
        loss = criterion(pred, tgt)
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

