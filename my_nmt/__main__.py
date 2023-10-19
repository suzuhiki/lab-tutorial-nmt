import argparse
import json
import os
import torch
import random
import sys
import datetime

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from .other.my_dataset import MyDataset
from .model.lstm import LSTM
from .model.alstm import ALSTM
from .model.transformer import Transformer
from .model.SAN import SAN_NMT, Encoder, Decoder
from .mode.lstm_train import lstm_train
from .mode.lstm_test import lstm_test
from .mode.transformer_train import transformer_train
from .mode.SAN_train import SAN_train


def main():
    # for debug
    torch.set_printoptions(edgeitems=1000)
    torch.autograd.set_detect_anomaly(True)

    # コマンドライン引数読み取り
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--src_train_path", type=str, default=None)
    parser.add_argument("--src_dev_path", type=str, default=None)
    parser.add_argument("--src_test_path", type=str, default=None)
    
    parser.add_argument("--tgt_train_path", type=str, default=None)
    parser.add_argument("--tgt_dev_path", type=str, default=None)
    parser.add_argument("--tgt_test_path", type=str, default=None)
    
    parser.add_argument("--model", type=str, default="LSTM")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epoch_num", type=int, default=24)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--embed_size", type=int, default=256)
    
    parser.add_argument("--save_dir", type=str, default="./output")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model_file_path", type=str, default=None)
    
    parser.add_argument("--weight_decay", type=float, default=0) # AdamのL2正則化
    parser.add_argument("--dropout", type=float, default=0) # defaultではdropout無効
    
    parser.add_argument("--head_num", type=int, default=8)
    parser.add_argument("--ff_hidden_size", type=int, default=2048)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--block_num", type=int, default=6)
    parser.add_argument("--max_len", type=int, default=500)
    
    args = parser.parse_args()
    
    #初期設定
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))
    
    special_token = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
    batch_size = args.batch_size
    padding_id = special_token["<pad>"]
    
    model_names = ["LSTM", "ALSTM", "Transformer", "SAN"]
    mode_names = ["train", "test"]
    
    # コマンドライン引数確認
    if args.model not in model_names:
        print("(--model) 実装されているモデル名を指定してください")
        for m in model_names:
            print(m)
        sys.exit()
    
    if args.mode not in mode_names:
        print("(--mode) trainかtestを選択してください")
        sys.exit()

    
    ### 学習処理
    if args.mode == "train":
        # 出力先のフォルダを作成
        datatime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = "{}/{}_{}".format(args.save_dir, args.model, datatime_str)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # ログファイル設定
        writer = SummaryWriter(log_dir=save_dir)
        
        # config出力
        with open("{}/config.json".format(save_dir), mode="w") as f:
            json.dump(vars(args), f, separators=(",", ":"), indent=4)
            
        train_dataset = MyDataset(args.src_train_path, args.tgt_train_path, special_token)
        
        t_dicts = train_dataset.get_dicts()
        
        dev_dataset = MyDataset(args.src_dev_path, args.tgt_dev_path, special_token, *t_dicts)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_func)
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_func)
        
        src_vocab_size, tgt_vocab_size = train_dataset.get_vocab_size()
        print("語彙サイズ：src {}, tgt {}".format(src_vocab_size, tgt_vocab_size))
        
        if args.model == "Transformer":
            model = Transformer(src_vocab_size, tgt_vocab_size, args.dropout, args.head_num, args.feature_dim, special_token, args.max_len, device, args.block_num, args.ff_hidden_size).to(device)
            print(model)
            
            optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
            criterion = torch.nn.CrossEntropyLoss(ignore_index=padding_id)
            
            tgt_id2w = train_dataset.get_tgt_id2w()
            
            transformer_train(model, train_dataloader, dev_dataloader, optimizer, criterion, args.epoch_num, device, batch_size,
                              tgt_id2w, model_save_span=3, model_save_path=save_dir, writer=writer)
        
        elif args.model == "SAN":
            encoder = Encoder(src_vocab_size, args.feature_dim, args.block_num, args.head_num, args.ff_hidden_size, args.dropout, args.max_len, device).to(device)
            decoder = Decoder(tgt_vocab_size, args.feature_dim, args.block_num, args.head_num, args.ff_hidden_size, args.dropout, args.max_len, device).to(device)
            model = SAN_NMT(encoder, decoder, special_token["<pad>"], special_token["<pad>"], device).to(device)
            optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
            criterion = nn.CrossEntropyLoss(ignore_index=special_token["<pad>"])
            
            print(model)
            
            tgt_id2w = train_dataset.get_tgt_id2w()
            
            SAN_train(model, args.epoch_num, train_dataloader, dev_dataloader, optimizer, criterion, writer, device, 3, save_dir, special_token, args.max_len, tgt_id2w)
        
        else:
            if args.model == "LSTM":
                model = LSTM(args.hidden_size, src_vocab_size, tgt_vocab_size, padding_id, args.embed_size, device, args.dropout).to(device)
            elif args.model == "ALSTM":
                model = ALSTM(args.hidden_size, src_vocab_size, tgt_vocab_size, padding_id, args.embed_size, device, args.dropout).to(device)

            print(model)

            optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
            criterion = torch.nn.CrossEntropyLoss(ignore_index=padding_id)

            tgt_id2w = train_dataset.get_tgt_id2w()

            lstm_train(model, train_dataloader, dev_dataloader, optimizer, criterion, args.epoch_num, device, batch_size, 
                       tgt_id2w, model_save_span=3, model_save_path=save_dir, writer=writer)
    
    
    elif args.mode == "test":
        train_dataset = MyDataset(args.src_train_path, args.tgt_train_path, special_token)
        
        t_dicts = train_dataset.get_dicts()
        src_vocab_size, tgt_vocab_size = train_dataset.get_vocab_size()
        tgt_id2w = train_dataset.get_tgt_id2w()
        
        test_detaset = MyDataset(args.src_test_path, args.tgt_test_path, special_token, *t_dicts)
        test_dataloader = DataLoader(test_detaset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_func)
        
        if args.model == "LSTM":
            model = LSTM(args.hidden_size, src_vocab_size, tgt_vocab_size, padding_id, args.embed_size, device, args.dropout).to(device)
        elif args.model == "ALSTM":
            model = ALSTM(args.hidden_size, src_vocab_size, tgt_vocab_size, padding_id, args.embed_size, device, args.dropout).to(device)
            
        print(model)
        
        model.load_state_dict(torch.load(args.model_file_path))
        
        output_path = os.path.dirname(args.model_file_path)
        
        lstm_test(model, test_dataloader, batch_size, device, output_path, tgt_id2w)


# データローダーに使う関数
def collate_func(batch):
    src_t = []
    tgt_t = []
    
    for src, tgt in batch:
        src_t.append(torch.tensor(src))
        tgt_t.append(torch.tensor(tgt))
    
    return pad_sequence(src_t, batch_first=True), pad_sequence(tgt_t, batch_first=True)


if __name__ == "__main__":
    main()