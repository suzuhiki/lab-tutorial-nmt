#!/bin/bash

# 英語→日本語の翻訳
# プロジェクトのルートディレクトリで実行

python3 -m my_nmt \
--src_train_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/ogura/train-1_top100000.en.txt \
--src_dev_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/dev.en \
--tgt_train_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/ogura/train-1_top100000.ja.txt \
--tgt_dev_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/dev.ja \
--learning_rate 0.0001 \
--epoch_num 20 \
--batch_size 94 \
--dropout 0.1 \
--weight_decay 0.001 \
--model SAN \

