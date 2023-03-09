#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 nohup python seq2seq_modelling.py -model_type treebert2bert -et_sep False -model_chkpt tbert2bert_new_init_none_train_newswire > ./logs/tbert2bert_new_init_none_train_newswire.log &
#CUDA_VISIBLE_DEVICES=1 nohup python seq2seq_modelling.py -model_type treebert2bert -et_sep True -model_chkpt tbert2bert_etsep_init_none_train_newswire > ./logs/tbert2bert_etsep_init_none_train_newswire.log &
#CUDA_VISIBLE_DEVICES=2 nohup python seq2seq_modelling.py -model_type bert2bert -et_sep False -model_chkpt bert2bert_init_none_train_newswire > ./logs/bert2bert_init_none_train_newswire.log & 
#CUDA_VISIBLE_DEVICES=3 nohup python seq2seq_modelling.py -model_type bert2bert -et_sep True -model_chkpt bert2bert_etsep_init_none_train_newswire > ./logs/bert2bert_etsep_init_none_train_newswire.log &

#CUDA_VISIBLE_DEVICES=0 nohup python tellmewhy_modelling.py -model_type treebert2bert -et_sep False -load_checkpoint_pth ./model_checkpoints/tbert2bert_init_none_train_newswire.pt -model_chkpt tbert2bert_init_newswire_train_tellwhy > ./logs/tbert2bert_init_newswire_train_tellwhy.log &
#wait
CUDA_VISIBLE_DEVICES=0 nohup python evalutation_script.py -eval_data_path ./Data/raw/seq2seq/new_test -load_checkpoint_pth ./model_checkpoints/tbert2bert_init_none_train_newswire.pt -model_type treebert2bert -et_sep False > ./logs/eval_newswire_tbert2bert_init_none_train_newswire_new.log &
#wait
#CUDA_VISIBLE_DEVICES=0 nohup python evalutation_script_v2.py -eval_data_path ./Data/raw/seq2seq/eval_tell_me_why -load_checkpoint_pth ./model_checkpoints/tbert2bert_init_none_train_newswire.pt -model_type treebert2bert -et_sep False > ./logs/eval_tellwhy2_tbert2bert_init_none_train_newswire.log &
#wait
#CUDA_VISIBLE_DEVICES=0 nohup python evalutation_script_v2.py -eval_data_path ./Data/raw/seq2seq/eval_tell_me_why -load_checkpoint_pth ./model_checkpoints/tbert2bert_init_newswire_train_tellwhy.pt -model_type treebert2bert -et_sep False > ./logs/eval_tellwhy2_tbert2bert_init_newswire_train_tellwhy.log &
