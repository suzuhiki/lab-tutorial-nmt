#!/bin/bash
# required : multi-bleu.perl included in moses
multi_bleu="/home/morioka/util/mosesdecoder/scripts/generic/multi-bleu.perl"
ref_path="/home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/test.en"
hyp_path="/home/morioka/workspace/git_projects/lab-tutorial-nmt/data/test_output/lstm_s2s_2_0.02928408303238978.en"

perl ${multi_bleu} -lc ${ref_path} < ${hyp_path} 