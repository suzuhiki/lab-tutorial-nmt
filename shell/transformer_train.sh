#!/bin/bash

# 英語→日本語の翻訳
# プロジェクトのルートディレクトリで実行

python3 -m my_nmt \
--src_train_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/ogura/train-1_top100000.en.txt \
--src_dev_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/dev.128.en \
--tgt_train_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/ogura/train-1_top100000.ja.txt \
--tgt_dev_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/dev.128.ja \
--learning_rate 0.0005 \
--epoch_num 28 \
--batch_size 128 \
--dropout 0.1 \
--weight_decay 0.00001 \
--model Transformer \
--init_weight \
--model_save_span 30
