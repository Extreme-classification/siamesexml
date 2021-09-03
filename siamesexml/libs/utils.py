import numpy as np
import torch
import json
import os
from scipy.sparse import csr_matrix, save_npz
import numba as nb


def save_predictions(preds, result_dir, valid_labels, num_samples,
                     num_labels, get_fnames=['knn', 'clf', 'combined'],
                     prefix='predictions'):
    if isinstance(preds, dict):
        for _fname, _pred in preds.items():
            if _fname in get_fnames:
                if valid_labels is not None:
                    predicted_labels = map_to_original(
                        _pred, valid_labels, _shape=(num_samples, num_labels))
                else:
                    predicted_labels = _pred
                save_npz(os.path.join(
                    result_dir, '{}_{}.npz'.format(prefix, _fname)),
                    predicted_labels, compressed=False)
    else:
        if valid_labels is not None:
            predicted_labels = map_to_original(
                preds, valid_labels, _shape=(num_samples, num_labels))
        else:
            predicted_labels = preds
        save_npz(os.path.join(result_dir, '{}.npz'.format(prefix)),
                 predicted_labels, compressed=False)


def append_padding_classifier_one(classifier, num_labels,
                                  key_w='classifier.weight',
                                  key_b='classifier.bias'):
    _num_labels, dims = classifier[key_w].size()
    if _num_labels != num_labels:
        status = "Appended padding classifier."
        _device = classifier[key_w].device
        classifier[key_w] = torch.cat(
            [classifier[key_w], torch.zeros(1, dims).to(_device)], 0)
        classifier[key_b] = torch.cat(
            [classifier[key_b], -1e5*torch.ones(1, 1).to(_device)], 0)
    else:
        status = "Shapes are fine, Not padding again."
    return status


def append_padding_classifier(net, num_labels):
    if isinstance(num_labels, list):
        status = []
        for idx, item in enumerate(num_labels):
            status.append(append_padding_classifier_one(
                net, item, 'classifier.classifier.{}.weight'.format(
                    idx), 'classifier.classifier.{}.bias'.format(idx)))
        print("Padding not implemented for distributed classifier for now!")
    else:
        return append_padding_classifier_one(net, num_labels)


def append_padding_embedding(embeddings):
    """
        Append a row of zeros as embedding for <PAD>
        Args:
            embeddings: numpy.ndarray: embedding matrix
        Returns:
            embeddings: numpy.ndarray: transformed embedding matrix
    """
    embedding_dim = embeddings.shape[1]
    app = np.zeros((1, embedding_dim))
    return np.vstack([app, embeddings])


def get_header(fname):
    with open(fname, 'r') as fp:
        line = fp.readline()
    return list(map(int, line.split(" ")))


def get_data_stats(fname, key):
    def get(fname, key):
        with open(fname, 'r') as fp:
            val = json.load(fp)[key]
        return val
    if isinstance(key, tuple):
        out = []
        for _key in key:
            out.append(get(fname, _key))
        return tuple(out)
    else:
        return get(fname, key)


def map_to_original(mat, mapping, _shape, axis=1):
    mat = mat.tocsr()
    row_idx, col_idx = mat.nonzero()
    vals = np.array(mat[row_idx, col_idx]).squeeze()
    col_indices = list(map(lambda x: mapping[x], col_idx))
    return csr_matrix(
        (vals, (np.array(row_idx), np.array(col_indices))), shape=_shape)


def save_parameters(fname, params):
    json.dump({'num_labels': params.num_labels,
               'vocabulary_dims': params.vocabulary_dims,
               'use_shortlist': params.use_shortlist,
               'ann_method': params.ann_method,
               'num_nbrs': params.num_nbrs,
               'embedding_dims': params.embedding_dims,
               'label_padding_index': params.label_padding_index,
               'keep_invalid': params.keep_invalid},
              open(fname, 'w'),
              sort_keys=True,
              indent=4)


def load_parameters(fname, params):
    temp = json.load(open(fname, 'r'))
    params.num_labels = temp['num_labels']
    params.vocabulary_dims = temp['vocabulary_dims']
    params.num_nbrs = temp['num_nbrs']
    params.label_padding_index = temp['label_padding_index']
    params.ann_method = temp['ann_method']
    params.embedding_dims = temp['embedding_dims']
    params.keep_invalid = temp['keep_invalid']
