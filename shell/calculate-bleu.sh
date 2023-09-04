#!/bin/bash
# required : multi-bleu.perl included in moses
multi_bleu="/home/morioka/util/mosesdecoder/scripts/generic/multi-bleu.perl"
ref_path="./resource/tokenized/test.ja"
hyp_path="./output/LSTM_20230904_111500/test_0.04892972144348176.txt"

perl ${multi_bleu} -lc ${ref_path} < ${hyp_path} 