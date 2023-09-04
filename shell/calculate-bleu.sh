#!/bin/bash
# required : multi-bleu.perl included in moses
multi_bleu="/home/morioka/util/mosesdecoder/scripts/generic/multi-bleu.perl"
ref_path="./resource/tokenized/test.ja"
hyp_path="./output/LSTM_20230903_221659_base/test_out.txt"

perl ${multi_bleu} -lc ${ref_path} < ${hyp_path} 