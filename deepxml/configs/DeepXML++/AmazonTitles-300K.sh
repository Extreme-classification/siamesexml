version=0
use_post=0
evaluation_type=1
num_splits=2
split_threshold=3
topk=600
embedding_dims=300
learning_rates=1
dlr_factor=0.5
num_labels=316053
A=0.6
B=2.6
ns_method=ensemble

lr_full=(0.02)
num_epochs_full=25
num_centroids_full=1
batch_size_full=128
dlr_step_full=14


lr_shortlist=(0.005)
num_epochs_shortlist=25
num_centroids_shortlist=1
batch_size_shortlist=255
dlr_step_shortlist=15


order=("shortlist" "full")
