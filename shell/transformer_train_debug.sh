#!/bin/bash

# 英語→日本語の翻訳
# プロジェクトのルートディレクトリで実行

python3 -m my_nmt \
--src_train_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/train-1.short.en \
--src_dev_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/dev.128.en \
--tgt_train_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/train-1.short.ja \
--tgt_dev_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/dev.128.ja \
--learning_rate 0.0005 \
--epoch_num 20 \
--batch_size 128 \
--dropout 0.1 \
--weight_decay 0.00001 \
--model Transformer \

