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
data_dir="${work_dir}/data"
current_working_dir=$(pwd)
docs=("trn" "tst")

echo -e "\nUsing pre-trained embeddings."
embedding_file="fasttextB_embeddings_${embedding_dims}d.npy"

stats=`python3 -c "import sys, json; print(json.load(open('${temp_model_data}/aux_stats.json'))['${quantile}'])"` 
stats=($(echo $stats | tr ',' "\n"))
vocabulary_dims=${stats[0]}
num_labels=${stats[2]}

DEFAULT_PARAMS="--dataset ${dataset} \
                --data_dir=${work_dir}/data \
                --num_labels ${num_labels} \
                --vocabulary_dims ${vocabulary_dims} \
                --embeddings $embedding_file \
                --embedding_dims ${embedding_dims} \
                --num_epochs $num_epochs \
                --tr_feat_fname trn_X_Xf.txt \
                --tr_label_fname trn_X_Y.txt \
                --lbl_feat_fname lbl_X_Xf.txt \
		        --val_feat_fname tst_X_Xf.txt \
                --val_label_fname tst_X_Y.txt \
                --ts_feat_fname tst_X_Xf.txt \
                --ts_label_fname tst_X_Y.txt \
                --top_k $topk \
                --seed ${seed} \
                --shortlist_method random \
                --model_fname ${MODEL_NAME} ${extra_params} \
                --get_only knn clf"

TRAIN_PARAMS="  --trans_method ${current_working_dir}/full.json \
                --dropout 0.5 --optim Adam \
                --lr $learning_rate \
                --num_nbrs 1 \
                --model_method embedding \
                --dlr_factor $dlr_factor \
                --dlr_step $dlr_step \
                --batch_size $batch_size \
                --normalize \
                ${DEFAULT_PARAMS}"

PREDICT_PARAMS="--model_method full \
                --normalize \
                --model_fname ${MODEL_NAME}\
                --pred_fname test_predictions \
                --out_fname predictions.txt \
                --batch_size 256 \
                ${DEFAULT_PARAMS}"

EXTRACT_PARAMS="--dataset ${dataset} \
                --data_dir ${work_dir}/data \
                --normalize \
                --model_method full \
                --model_fname ${MODEL_NAME}\
                --batch_size 512 ${extra_params}"


./run_base.sh "train" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${TRAIN_PARAMS}"
./run_base.sh "extract" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${EXTRACT_PARAMS} --ts_feat_fname 0 --out_fname export/wrd_emb"

if [ "${quantile}" == "aux" ]
then
    echo -e "\nGenerating embeddings from auxiliary task."
    cp "${work_dir}/results/DeepXML/${dataset}/v_${dir_version}/aux/export/wrd_emb.npy" "${work_dir}/models/DeepXML/${dataset}/v_${dir_version}/aux_embeddings_${embedding_dims}d.npy"
    exit
fi

exit
./run_base.sh "predict" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${PREDICT_PARAMS}"

for doc in ${docs[*]} 
do 
    ./run_base.sh "extract" $dataset $work_dir $dir_version/$quantile $MODEL_NAME "${EXTRACT_PARAMS} --ts_feat_fname ${doc}_X_Xf.txt --ts_label_fname ${doc}_X_Y.txt --out_fname export/${doc}_emb"
done
