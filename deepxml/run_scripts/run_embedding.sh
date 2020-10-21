#!/bin/bash
dataset=$1
dir_version=$2
quantile=$3
use_post=$4
learning_rate=${5}
embedding_dims=$6
num_epochs=$7
dlr_factor=$8
dlr_step=${9}
batch_size=${10}
work_dir=${11}
MODEL_NAME="${12}"
temp_model_data="${13}"
topk=${14}
shortlist_method=${16}
seed=${17}
extra_params="${18}"
init='pretrained'
data_dir="${work_dir}/data"
current_working_dir=$(pwd)
docs=("trn" "tst")

trn_ft_file="trn_X_Xf.txt"
trn_lbl_file="trn_X_Y.txt"
tst_ft_file="tst_X_Xf.txt"
tst_lbl_file="tst_X_Y.txt"
lbl_ft_file="lbl_X_Xf.txt"

extra_params="${extra_params} --normalize --feature_type sparse"

embedding_file="fasttextB_embeddings_${embedding_dims}d.npy"

stats=`python3 -c "import sys, json; print(json.load(open('${temp_model_data}/aux_stats.json'))['${quantile}'])"` 
stats=($(echo $stats | tr ',' "\n"))
vocabulary_dims=$((${stats[0]}+1))
num_labels=${stats[2]}

DEFAULT_PARAMS="--dataset ${dataset} \
                --data_dir=${work_dir}/data \
                --num_labels ${num_labels} \
                --vocabulary_dims_document ${vocabulary_dims} \
                --vocabulary_dims_label ${vocabulary_dims} \
                --embeddings $embedding_file \
                --embedding_dims ${embedding_dims} \
                --num_epochs $num_epochs \
                --tr_feat_fname ${trn_ft_file} \
                --tr_label_fname ${trn_lbl_file} \
	        --val_feat_fname ${tst_ft_file} \
                --val_label_fname ${tst_lbl_file} \
                --ts_feat_fname ${tst_ft_file} \
                --ts_label_fname ${tst_lbl_file} \
                --lbl_feat_fname ${lbl_ft_file} \
                --top_k $topk \
                --seed ${seed} \
                --shortlist_method random \
                --share_weights \
                --model_fname ${MODEL_NAME} ${extra_params} \
                --get_only knn clf"

TRAIN_PARAMS="  --trans_method_document ${current_working_dir}/embedding.json \
                --trans_method_label ${current_working_dir}/embedding.json \
                --dropout 0.5 --optim Adam \
                --lr $learning_rate \
                --num_nbrs 1 \
                --margin 0.2 \
                --loss triplet_margin_onm \
                --model_method embedding \
                --num_workers 6 \
                --ann_threads 12 \
		--save_intermediate \
                --validate_after 30 \
                --validate \
                --init ${init} \
                --dlr_factor $dlr_factor \
                --dlr_step $dlr_step \
                --batch_size $batch_size \
                ${DEFAULT_PARAMS}"

PREDICT_PARAMS="--model_method full \
                --model_fname ${MODEL_NAME}\
                --pred_fname test_predictions \
                --out_fname predictions.txt \
                --batch_size 256 \
                ${DEFAULT_PARAMS}"

EXTRACT_PARAMS="--dataset ${dataset} \
                --data_dir ${work_dir}/data \
		--share_weights \
		--shortlist_method  random \
		--vocabulary_dims_document ${vocabulary_dims} \
		--vocabulary_dims_label ${vocabulary_dims} \
		--trans_method_document ${current_working_dir}/embedding.json \
		--trans_method_label ${current_working_dir}/embedding.json \
                --model_method embedding \
                --model_fname ${MODEL_NAME}\
                --batch_size 512 ${extra_params}"


./run_base.sh "train" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${TRAIN_PARAMS}"
./run_base.sh "predict" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${PREDICT_PARAMS}"

for doc in ${docs[*]} 
do 
    ./run_base.sh "extract" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${EXTRACT_PARAMS} --ts_feat_fname ${doc}_X_Xf.txt --out_fname export/${doc}_emb"
done
