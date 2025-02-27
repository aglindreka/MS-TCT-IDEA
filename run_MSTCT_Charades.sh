#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python train.py \
-dataset charades \
-mode rgb \
-model MS_TCT \
-train True \
-num_clips 128 \
-skip 0 \
-lr 0.00001 \
-comp_info False \
-epoch 80 \
-unisize True \
-alpha_l 10 \
-beta_l 1 \
-batch_size 1 \
-num_layer 12