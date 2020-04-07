dataset=$1

# train word embeddings with head labels only
./run_main.sh ${gpu_id} DeepXML-rlabeling ${dataset} $version $nc full $lr

# cluster labels

# train for different clusters