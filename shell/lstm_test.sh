#!/bin/bash

# 英語→日本語の翻訳
# プロジェクトのルートディレクトリで実行

python3 -m my_nmt \
--src_train_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/train-1.short.en \
--src_dev_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/dev.en \
--tgt_train_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/train-1.short.ja \
--tgt_dev_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/dev.ja \
--learning_rate 0.001 \
--epoch_num 28 \
--src_test_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/test.en \
--tgt_test_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/test.ja \
--mode test \
--model_file_path /home/morioka/workspace/git_projects/lab-tutorial-nmt/output/LSTM_20230903_221659_base/25_0.03873826608224089.pth \

