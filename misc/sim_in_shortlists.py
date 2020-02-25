"""
    Script to compute similarity between following shortlists:
    - over label centroids
    - over label representation from DeepXML
""" 

from xclib.data import data_utils
from xclib.utils import ann
from scipy.sparse import csr_matrix
from xclib.utils.sparse import topk, binarize, retain_topk
from xclib.evaluation.xc_metrics import Metrices
import numpy as np
import sys
from scipy.sparse import save_npz
from functools import reduce


def load_filter_mapping(fname):
    mapping = np.loadtxt(fname, dtype=np.int)
    return mapping


def remove_trivial_labels(pred, mapping):
    pred[mapping[:, 0], mapping[:, 1]] = 0
    pred.eliminate_zeros()
    return pred


def recall(predicted_labels, true_labels, k=300):
    """Compute recall@k
    Args:
    predicted_labels: csr_matrix
        predicted labels
    true_labels: csr_matrix
        true_labels
    k: int, default=5
        keep only top-k predictions
    """
    #predicted_labels = retain_topk(predicted_labels, k)
    denom = np.sum(true_labels, axis=1)
    rc = binarize(true_labels.multiply(predicted_labels))
    rc = np.sum(rc, axis=1)/(denom+1e-5)
    return np.mean(rc)*100


def jaccard_similarity(pred_0, pred_1, y=None): 
    """Jaccard similary b/w two different predictions matrices
    Args:
    pred_0: csr_matrix
        prediction for algorithm 0
    pred_1: csr_matrix
        prediction for algorithm 1
    y: csr_matrix or None
        true labels
    """
    def _correct_only(pred, y):
        pred = pred.multiply(y)
        pred.eliminate_zeros()
        return pred

    def _safe_divide(a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.true_divide(a, b)
            out[out == np.inf] = 0
            return np.nan_to_num(out)

    if y is not None:
        pred_0 = _correct_only(pred_0, y)
        pred_1 = _correct_only(pred_1, y)

    pred_0, pred_1 = binarize(pred_0), binarize(pred_1)
    intersection = np.array(pred_0.multiply(pred_1).sum(axis=1)).ravel()
    union = np.array(binarize(pred_0 + pred_1).sum(axis=1)).ravel()
    return np.mean(_safe_divide(intersection, union))


def get_valid_labels(trn_lb, lb_embed, trn_doc):
    """
        Remove invalid labels
    """
    ind_0 = np.where(np.array(trn_lb.sum(axis=0)).ravel())[0] #nnz
    ind_1 = np.where(np.square(lb_embed).sum(axis=1).ravel())[0] #nnz
    ind_2 = np.where(np.square(compute_centroid(trn_doc, trn_lb)).sum(axis=1).ravel())[0] #nnz
    return reduce(lambda x, y: np.intersect1d(x, y), [ind_0, ind_1, ind_2])


def convert_to_csr(ind, dist, original_mapping, _shape):
    """
        Convert dense data to a csr_matrix
    """
    rows = []
    data = []
    cols = []
    for idx, (_ind, _dist) in enumerate(zip(ind, dist)):
        cols.extend(_ind)
        data.extend(_dist)
        rows.extend([idx]*len(_ind))
    cols = np.fromiter(map(lambda x: original_mapping[x], cols), dtype=np.int)
    return csr_matrix((1-np.array(data), (rows, cols)), shape=_shape)


def compute_centroid(ft, lb):
    """
        compute label centroid
    """
    centroids = lb.transpose().dot(ft)
    return centroids


def compute_shortlist(tr_doc_embed, label_embed, num_labels, original_mapping, filter_mapping, ts_doc_embed=None):
    """
    Train an ANN graph over label embeddings and return shortlist
    
    Args:
    -----
    tr_doc_embed: np.ndarray, float32/float64
        document embeddings for training documents
    label_embed: np.ndarray, float32/float64
        label embeddings
    ts_doc_embed: optional; None or np.ndarray, float32/float64
        document embeddings for test documents
    
    Returns:
    --------
    tr_shortlist: csr_matrix
        predictions as per the training documents
    ts_shortlist: optional; None or csr_matrix
        predictions as per the test documents
    """
    # Train the graph
    graph = ann.HNSW(M=100, efC=300, efS=300, num_neighbours=300, space='cosine', num_threads=12, verbose=True)    
    graph.fit(label_embed)
    tr_shortlist = None
    #ind, dist = graph.predict(tr_doc_embed)
    #tr_shortlist = convert_to_csr(ind, dist, original_mapping, (tr_doc_embed.shape[0], num_labels))
    ts_shortlist = None
    if ts_doc_embed is not None:
        ind, dist = graph.predict(ts_doc_embed)
        ts_shortlist = convert_to_csr(ind, dist, original_mapping, (ts_doc_embed.shape[0], num_labels))
        ts_shortlist = remove_trivial_labels(ts_shortlist, filter_mapping)
    return tr_shortlist, ts_shortlist


def evaluate(_true, _pred):
    acc = Metrices(_true)
    prec = acc.eval(_pred)[0]*100
    rec = recall(_pred, _true)
    return ",".join(map(str, prec)), rec


def main():
    # load dense features
    tr_doc_embed = np.load(sys.argv[1]) # train document embeddings
    ts_doc_embed = np.load(sys.argv[2]) # test document embeddings
    label_embed = np.load(sys.argv[3]) # label embeddings
    num_labels = label_embed.shape[0]

    # load sparse
    tr_labels = data_utils.read_sparse_file(sys.argv[4]) # train labels
    ts_labels = data_utils.read_sparse_file(sys.argv[5]) # test labels
    filter_mapping = load_filter_mapping(sys.argv[6])
    valid_indices = get_valid_labels(tr_labels, label_embed, tr_doc_embed)

    _, ts_shortlist_centroid = compute_shortlist(
        tr_doc_embed, 
        compute_centroid(tr_doc_embed, tr_labels[:, valid_indices]),
        num_labels,
        valid_indices,
        filter_mapping,
        ts_doc_embed)

    prec, rec = evaluate(ts_labels, ts_shortlist_centroid)
    print("Precision: {} and recall: {}".format(prec, rec))

    _, ts_shortlist_embed = compute_shortlist(
        tr_doc_embed, 
        label_embed[valid_indices],
        num_labels,
        valid_indices,
        filter_mapping,
        ts_doc_embed)
    prec, rec = evaluate(ts_labels, ts_shortlist_embed)
    print("Precision: {} and recall: {}".format(prec, rec))

    prec, rec = evaluate(ts_labels, ts_shortlist_embed+ts_shortlist_centroid)
    print("Precision: {} and recall: {}".format(prec, rec))
    print("Jaccard similarity: {}".format(jaccard_similarity(ts_shortlist_centroid, ts_shortlist_embed)))
    print("Jaccard similarity (^true labels): {}".format(jaccard_similarity(ts_shortlist_centroid, ts_shortlist_embed, ts_labels)))

if __name__ == "__main__":
    main()