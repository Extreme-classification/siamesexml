version=0
use_post=1
evaluation_type=1
num_splits=2
split_threshold=15
topk=50
embedding_dims=300
learning_rates=1
dlr_factor=0.5
num_labels=3993
A=0.55
B=1.5
ns_method=kcentroid

lr_full=(0.009)
num_epochs_full=25
num_centroids_full=1
batch_size_full=255
dlr_step_full=14


lr_shortlist=(0.007)
num_epochs_shortlist=20
num_centroids_shortlist=1
batch_size_shortlist=255
dlr_step_shortlist=10


order=("shortlist" "full")