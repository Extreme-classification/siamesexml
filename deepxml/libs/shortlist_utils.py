from sklearn.preprocessing import normalize
import numpy as np
import libs.utils as utils


def get_and_update_shortlist(document_embeddings, shorty,
                             data_loader, _save_mem=True):
    # FIXME: Figure out a way to delete document embeddings
    short, distances = shorty.query(document_embeddings)
    data_loader.dataset.update_shortlist(short, distances)


def update(data_loader, model, embedding_dim, shorty, flag=0,
           use_coarse=False):
    # 0: train and update, 1: train, 2: update
    doc_embeddings = model._document_embeddings(
        data_loader, return_coarse=use_coarse)
    # Do not normalize if kmeans clustering needs to be done!
    # doc_embeddings = normalize(doc_embeddings, copy=False)
    if flag == 0:
        shorty.fit(
            label_features=data_loader.dataset.lbl_features.data if data_loader.dataset.lbl_features is not None else None,
            features=doc_embeddings,
            labels=data_loader.dataset.labels.data)
        get_and_update_shortlist(doc_embeddings, shorty, data_loader)
    elif flag == 1:
        # train and don't get shortlist
        shorty.fit(
            label_features=data_loader.dataset.lbl_features.data if data_loader.dataset.lbl_features is not None else None,
            features=doc_embeddings,
            labels=data_loader.dataset.labels.data)
    else:
        # get shortlist
        get_and_update_shortlist(doc_embeddings, shorty, data_loader)
    return None
