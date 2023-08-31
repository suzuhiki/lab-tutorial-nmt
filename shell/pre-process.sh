#!/bin/bash

input_dir=/home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/src
output_dir=/home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized

TOKENIZER=/home/morioka/util/mosesdecoder/scripts/tokenizer/tokenizer.perl
KYTEA_MODEL=/home/morioka/util/kytea/models/jp-0.4.7-1.mod
Z2H=/home/morioka/util/script.converter.distribution/z2h-utf8.pl

echo "[Info] Tokenize English data..."
for file in train-1 dev devtest test; do
    cat ${input_dir}/${file}.en.txt | \
    perl -C $Z2H | \
    perl -C $TOKENIZER -threads 4 -l en -no-escape > ${output_dir}/${file}.en
done

echo "[Info] Tokenize Japanese data..."
for file in train-1 dev devtest test; do
    cat ${input_dir}/${file}.ja.txt | \
    perl -C -pe 'use utf8; s/　/ /g;' | \
    kytea -model $KYTEA_MODEL -out tok | \
    perl -C -pe 's/^ +//; s/ +$//; s/ +/ /g;' | \
    perl -C -pe 'use utf8; tr/\|[]/｜［］/; ' > ${output_dir}/${file}.ja
done