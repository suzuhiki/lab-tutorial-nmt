#!/bin/bash

# 英語→日本語の翻訳
# プロジェクトのルートディレクトリで実行

python3 -m my_nmt \
--src_train_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/debug_train.en \
--src_dev_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/debug_dev.en \
--tgt_train_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/debug_train.ja \
--tgt_dev_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/debug_dev.ja \
--learning_rate 0.001 \
--epoch_num 32 \
--dropout 0.1 \
--model Transformer \

