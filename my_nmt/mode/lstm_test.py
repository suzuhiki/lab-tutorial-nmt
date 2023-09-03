import torch
import numpy as np


def lstm_test(model, test_dataloader, device, ouput_path, id2w):
    model.train(False)
    pred_text_clean = []
    
    with torch.no_grad():
        for src, dst in test_dataloader:
            src_tensor = src.clone().detach().to(device)
            dst_tensor = dst.clone().detach().to(device)
        
            pred = model(src_tensor, dst_tensor)
            
            pred_text = []
            en_id2w = np.vectorize(lambda id: id2w[id])
            for sentence in pred:
                pred_text.append(en_id2w(sentence))

            for sentence in pred_text:
                tmp_list = []
                for word in sentence:
                    if word != "<bos>" and word != "<pad>" and word != "<eos>":
                        tmp_list.append(word)
                pred_text_clean.append(" ".join(tmp_list))
    return pred_text_clean