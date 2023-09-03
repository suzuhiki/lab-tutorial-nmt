#!/bin/bash

# 英語→日本語の翻訳
# プロジェクトのルートディレクトリで実行

python3 -m my_nmt \
--tgt_train_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/train-1.short.en \
--tgt_dev_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/dev.en \
--src_train_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/train-1.short.ja \
--src_dev_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/dev.ja \
--learning_rate 0.01 \
--epoch_num 28

