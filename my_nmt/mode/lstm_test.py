import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def lstm_test(model, test_dataloader, batch_size, device, ouput_path, id2w):
    model.train(False)
    pred_text_clean = []
    bleu_list = []
    
    with torch.no_grad():
        for src, dst in test_dataloader:
            src_tensor = src.clone().detach().to(device)
            dst_tensor = dst.clone().detach().to(device)
        
            pred = model(src_tensor, dst_tensor)
            
            pred_text = []
            t_id2w = np.vectorize(lambda id: id2w[id])
            for sentence in pred:
                pred_text.append(t_id2w(sentence))

            dst_text = t_id2w(dst.to("cpu").detach().numpy().copy())
            dst_text_clean = []
            
            for sentence in dst_text:
                tmp_list = []
                for word in sentence:
                    if word != "<bos>" and word != "<pad>" and word != "<eos>":
                        tmp_list.append(word)
                dst_text_clean.append(tmp_list)

            for sentence in pred_text:
                tmp_list = []
                for word in sentence:
                    if word != "<bos>" and word != "<pad>" and word != "<eos>":
                        tmp_list.append(word)
                pred_text_clean.append(tmp_list)
                
            bleu = 0
            for pred_c, dst_c in zip(pred_text, dst_text_clean):
                bleu += sentence_bleu([dst_c], pred_c,  smoothing_function=SmoothingFunction().method1)
                # print("".join(dst_c))
                # print("".join(pred_c))
            bleu = bleu / batch_size
            bleu_list.append(bleu)
            print("bleu: {}".format(bleu))
    
    bleu_all = sum(bleu_list) / len(bleu_list)
    print("bleu all: {}".format(bleu_all))
    
    pred_for_out = []
    for text in pred_text_clean:
        pred_for_out.append("".join(text))
    
    with open("{}/{}".format(ouput_path, "test_out.txt"), "w") as f:
        f.write("\n".join(pred_for_out))