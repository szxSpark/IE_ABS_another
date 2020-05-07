#!/bin/bash
set -x
SAVEPATH=../../../data/giga/models/seass
DATAHOME=../../../data/giga
#SAVEPATH=/home/aistudio/data/data22829/toutiao_word/models/seass
#DATAHOME=/home/aistudio/data/data22829/toutiao_word

python train.py \
       -save_path $SAVEPATH \
       -log_home $SAVEPATH \
       -online_process_data \
       -train_src ${DATAHOME}/train/train.article.txt -src_vocab ${DATAHOME}/train/source.vocab \
       -train_tgt ${DATAHOME}/train/train.title.txt -tgt_vocab ${DATAHOME}/train/target.vocab \
       -dev_input_src ${DATAHOME}/dev/valid.article.filter.txt.8k -dev_ref ${DATAHOME}/dev/valid.title.filter.txt.8k \
       -layers 1 -enc_rnn_size 512 -brnn -word_vec_size 300 -dropout 0.5 \
       -batch_size 64 -beam_size 1 \
       -epochs 20 \
       -optim adam -learning_rate 0.001 \
       -gpus 3 \
       -curriculum 0 -extra_shuffle \
       -start_eval_batch 50000 -eval_per_batch 5000 \
       -seed 12345 -cuda_seed 12345 \
       -max_sent_length 100 \
       -max_sent_length_source 100 \
       -subword False -is_save True \
       -pointer_gen False \
       -is_coverage False  -cov_loss_wt 1.0 \
       -share_embedding False \
       -english True \
       -halve_lr_bad_count 6


