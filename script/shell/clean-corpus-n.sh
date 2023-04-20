#!/bin/bash
# required : clean-corpus-n.perl included in moses
script_path=/home/morioka/util/mosesdecoder/scripts/training/clean-corpus-n.perl
corpus_path=/home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/train-1
output_path=/home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/train-1.cleaned

perl ${script_path} ${corpus_path} en ja ${output_path} 0 50
