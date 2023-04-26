#!/bin/bash

input_path=/home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/train-1.cleaned
output_path=/home/morioka/workspace/git_projects/lab-tutorial-nmt/resource/tokenized/train-1.short

echo "[Info] Delete lines after 200000..."
for extention in en ja; do
    sed -n '1,20000p' ${input_path}.${extention} > ${output_path}.${extention}
done