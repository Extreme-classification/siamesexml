#!/bin/bash
# Create an ensemble and create merged predictions
# $1 GPU DEIVCE ID
# $2 Model TYPE (DeepXML/DeepXML-fr etc.)
# $3 DATASET
# $4 VERSION
# eg. ./run_main.sh 0 DeepXML EURLex-4K 0
# eg. ./run_main.sh 0 DeepXML-fr EURLex-4K 0

gpu_id=$1
model_type=$2
dataset=$3
version=$4
seeds=(22 666 786)
beta=0.90 # Best beta value
index=0
fnames=""
save=1
source "../configs/${model_type}/${dataset}.sh"


work_dir="/data3/kd"
data_dir="${work_dir}/data/${dataset}"
results_dir="${work_dir}/results/${model_type}/${dataset}"
trn_lbl_file="${data_dir}/trn_X_Y.txt"
tst_lbl_file="${data_dir}/tst_X_Y.txt"
filter_labels_test="${data_dir}/filter_labels_test.txt"

# Run sequentially
for seed in ${seeds[@]}; do
    echo "Running learner: ${index}.."
    echo -e "${gpu_id} ${model_type} $dataset "${version}_${index}" ${seed}"
    #./run_main.sh ${gpu_id} ${model_type} $dataset "${version}_${index}" ${seed}
    #fnames="${fnames}${results_dir}/v_${version}_${index}/test_predictions_clf.npz,"
    fnames="${fnames}${results_dir}/v_${version}_${index}/score_beta_${beta}.npz,"
    ((index++))
done
echo -e "Evaluate ensemble!"
python3 ../tools/evaluate_ensemble.py ${trn_lbl_file} ${tst_lbl_file} ${fnames} ${filter_labels_test} $A $B $save
