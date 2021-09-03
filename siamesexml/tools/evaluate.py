# Example to evaluate
import sys
import xclib.evaluation.xc_metrics as xc_metrics
import xclib.data.data_utils as data_utils
from scipy.sparse import load_npz, save_npz
from xclib.utils.sparse import sigmoid, normalize, retain_topk
import numpy as np
import os


def get_filter_map(fname):
    if fname is not None:
        return np.loadtxt(fname).astype(np.int)
    else:
        return None


def filter_predictions(pred, mapping):
    if mapping is not None and len(mapping) > 0:
        print("Filtering labels.")
        pred[mapping[:, 0], mapping[:, 1]] = 0
        pred.eliminate_zeros()
    return pred


def compute_probs(predictions, c=1, d=5.0):
    predictions.data[:] = c*np.exp(d*predictions.data[:])/np.exp(d)
    return predictions


def main(tst_label_fname, trn_label_fname, pred_fname,
         A, B, betas, save, filter_fname=None, top_k=200):
    ans = ""
    true_labels = data_utils.read_sparse_file(tst_label_fname)
    trn_labels = data_utils.read_sparse_file(trn_label_fname)
    inv_propen = xc_metrics.compute_inv_propesity(trn_labels, A, B)
    mapping = get_filter_map(filter_fname)
    acc = xc_metrics.Metrics(true_labels, inv_psp=inv_propen)
    root = os.path.dirname(pred_fname)
    if isinstance(betas, list):
        knn = filter_predictions(
            load_npz(pred_fname+'_knn.npz'), mapping)
        clf = filter_predictions(
            load_npz(pred_fname+'_clf.npz'), mapping)
        args = acc.eval(clf, 5)
        ans = f"classifier\n{xc_metrics.format(*args)}"
        args = acc.eval(knn, 5)
        ans = ans + f"\nshortlist\n{xc_metrics.format(*args)}"
        knn = sigmoid(knn)
        clf = compute_probs(clf.copy())

        for beta in betas:
            predicted_labels = beta*clf + (1-beta)*knn
            args = acc.eval(predicted_labels, 5)
            ans = ans + f"\nbeta: {beta:.2f}\n{xc_metrics.format(*args)}"
            if save:
                fname = os.path.join(root, f"score_{beta:.2f}.npz")
                save_npz(fname, retain_topk(predicted_labels, k=top_k),
                    compressed=False)
    return ans


if __name__ == '__main__':
    train_label_file = sys.argv[1]
    targets_file = sys.argv[2]  # Usually test data file
    predictions_file = sys.argv[4]  # In mat format
    filter_map = sys.argv[3]
    A = float(sys.argv[5])
    B = float(sys.argv[6])
    save = int(sys.argv[7])
    betas = list(map(float, sys.argv[8:]))
    main(targets_file, train_label_file,
         predictions_file, A, B, betas, save, filter_map)
