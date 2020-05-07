#!/bin/bash
set -x
SAVEPATH=/home/zxsong/workspace/seass/data//toutiao_word/models/IE_ABS
DATAHOME=/home/zxsong/workspace/seass/data/toutiao_word
#SAVEPATH=/home/aistudio/work/toutiao_word/models/seass
#DATAHOME=/home/aistudio/work/toutiao_word
# dev_ref 不要用 subword

python train.py \
       -save_path $SAVEPATH \
       -log_home $SAVEPATH \
       -online_process_data \
       -train_src ${DATAHOME}/train/subword/train.article.txt -src_vocab ${DATAHOME}/train/subword/source.vocab \
       -train_svo ${DATAHOME}/train/subwordsvo.subword.txt \
       -train_tgt ${DATAHOME}/train/subword/train.title.txt -tgt_vocab ${DATAHOME}/train/subword/target.vocab \
       -dev_input_src ${DATAHOME}/dev/subword/valid.article.txt -dev_ref /home/aistudio/work/toutiao_char/dev/valid.title.txt \
       -layers 1 -enc_rnn_size 512 -brnn -word_vec_size 512 -dropout 0.5 \
       -batch_size 4 -beam_size 1 \
       -epochs 200 \
       -optim adam -learning_rate 0.001 \
       -gpus 2 \
       -curriculum 0 -extra_shuffle \
       -start_eval_batch 2000 -eval_per_batch 2000 \
       -log_interval 100 \
       -seed 12345 -cuda_seed 12345 \
       -max_sent_length 60 \
       -max_sent_length_source 1000 \
       -subword True -is_save True \
	     -pointer_gen False \
       -is_coverage False  -cov_loss_wt 1.0 \
       -share_embedding False \
       -max_grad_norm 5 \
       -halve_lr_bad_count 4
    #   -start_decay_at 20 -decay_interval 5\
    #   -min_lr 1e-6 \
    #   -lm_model_file /home/zxsong/workspace/seass/lm_pretrain_para_lstm_layer1_bptt30_tied.pkl \


#SAVEPATH=/home/aistudio/work/toutiao_word/models/seass
#DATAHOME=/home/aistudio/work/toutiao_word
# dev_ref 不要用 subword


#cat ../train.article.txt ../train.title.txt | subword-nmt learn-bpe -s 20000 -o codes
#subword-nmt apply-bpe -c codes < ../train.article.txt | subword-nmt get-vocab > voc.article
#subword-nmt apply-bpe -c codes < ../train.title.txt | subword-nmt get-vocab > voc.title
#subword-nmt apply-bpe -c codes --vocabulary voc.article --vocabulary-threshold 50 < ../train.article.txt > ./train.article.txt
#subword-nmt apply-bpe -c codes --vocabulary voc.title --vocabulary-threshold 50 < ../train.title.txt > ./train.title.txt


# CollectVocab
# 验证集
# subword-nmt apply-bpe -c ../../train/subword/codes --vocabulary ../../train/subword/voc.article  --vocabulary-threshold 50 < ../valid.article.txt > ./valid.article.txt
# subword-nmt apply-bpe -c ../../train/subword/codes --vocabulary ../../train/subword/voc.title --vocabulary-threshold 50 < ../valid.title.txt > valid.title.txt
# sed -r 's/(@@ )|(@@ ?$)//g' out.txt > out_detoken.txt

# 下边是应用10万条训练数据的用法
# subword-nmt apply-bpe -c ../../train_with_semi_supervised/subword/codes --vocabulary ../../train_with_semi_supervised/subword/voc.article --vocabulary-threshold 50 < ../valid.article.txt  > ./valid.article.semi.txt
# subword-nmt apply-bpe -c ../../train_with_semi_supervised/subword/codes --vocabulary ../../train_with_semi_supervised/subword/voc.title --vocabulary-threshold 50 < ../valid.title.txt  > ./valid.title.semi.txt
