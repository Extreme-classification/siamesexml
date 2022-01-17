import numpy as np
import json
import os
from scipy.sparse import save_npz
from xclib.utils.sparse import _map_cols


def save_predictions(preds, result_dir, valid_labels, num_samples,
                     num_labels, get_fnames=['knn', 'clf', 'combined'],
                     prefix='predictions'):
    if isinstance(preds, dict):
        for _fname, _pred in preds.items():
            if _fname in get_fnames:
                if valid_labels is not None:
                    predicted_labels = _map_cols(
                        _pred, valid_labels, shape=(num_samples, num_labels))
                else:
                    predicted_labels = _pred
                save_npz(os.path.join(
                    result_dir, '{}_{}.npz'.format(prefix, _fname)),
                    predicted_labels, compressed=False)
    else:
        if valid_labels is not None:
            predicted_labels = _map_cols(
                preds, valid_labels, shape=(num_samples, num_labels))
        else:
            predicted_labels = preds
        save_npz(os.path.join(result_dir, '{}.npz'.format(prefix)),
                 predicted_labels, compressed=False)


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
