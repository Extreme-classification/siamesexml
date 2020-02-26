from sklearn.preprocessing import normalize
import numpy as np
import libs.utils as utils


def get_and_update_shortlist(document_embeddings, shorty,
                             data_loader, _save_mem=True):
    # FIXME: Figure out a way to delete document embeddings
    if not hasattr(shorty, 'num_graphs'):
        _save_mem = False
    if _save_mem:  # Fetch one-by-one; save to disk and delete
        for idx in range(shorty.num_graphs):
            short, distances = shorty.query(document_embeddings, idx)
            data_loader.dataset.update_shortlist(short, distances, idx=idx)
    else:  # Fetch shortlist at once
        short, distances = shorty.query(document_embeddings)
        data_loader.dataset.update_shortlist(short, distances)


def get_partition_indices(num_graphs, data_loader):
    if num_graphs==1:
        return None
    else:
        partition_indices = []
        for idx in range(num_graphs):
            partition_indices.append(
                data_loader.dataset.shortlist.get_partition_indices(idx))
        return partition_indices

def update(data_loader, model, embedding_dim, shorty, flag=0,
           num_graphs=1, use_coarse=False):
    # 0: train and update, 1: train, 2: update
    doc_embeddings = model._document_embeddings(
        data_loader, return_coarse=use_coarse)
    # Do not normalize if kmeans clustering needs to be done!
    # doc_embeddings = normalize(doc_embeddings, copy=False)
    if flag == 0:
        partition_indices = get_partition_indices(num_graphs, data_loader)
        shorty.fit(doc_embeddings, data_loader.dataset.labels.data,
                   data_loader.dataset._ext_head, 
                   partition_indices=partition_indices)
        get_and_update_shortlist(doc_embeddings, shorty, data_loader)
    elif flag == 1:
        partition_indices = get_partition_indices(num_graphs, data_loader)
        # train and don't get shortlist
        shorty.fit(doc_embeddings, data_loader.dataset.labels.data,
                   data_loader.dataset._ext_head,
                   partition_indices=partition_indices)
    else:
        # get shortlist
        get_and_update_shortlist(doc_embeddings, shorty, data_loader)
    return None
