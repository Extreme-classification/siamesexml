# Example to evaluate
import sys
import xclib.evaluation.xc_metrics as xc_metrics
import xclib.data.data_utils as data_utils
from scipy.io import loadmat, savemat
from scipy.sparse import lil_matrix, load_npz, csr_matrix, save_npz
from xclib.utils.sparse import sigmoid, normalize, retain_topk
import scipy.sparse as sparse
import numpy as np
import os


def get_filter_map(fname):
    if fname is not None:
        return np.loadtxt(fname).astype(np.int)
    else:
        return None


def filter_predictions(pred, mapping):
    if mapping is not None:
        print("Filtering labels.")
        pred[mapping[:, 0], mapping[:, 1]] = 0
        pred.eliminate_zeros()
    return pred


def main(targets_label_file, train_label_file, predictions_file,
         A, B, betas, save, filter_fname=None):
    true_labels = data_utils.read_sparse_file(targets_label_file)
    trn_labels = data_utils.read_sparse_file(train_label_file)
    inv_propen = xc_metrics.compute_inv_propesity(trn_labels, A, B)
    mapping = get_filter_map(filter_fname)
    acc = xc_metrics.Metrics(true_labels, inv_psp=inv_propen)
    root = os.path.dirname(predictions_file)
    if betas[0] != -1:
        knn = filter_predictions(
            load_npz(predictions_file+'_knn.npz'), mapping)
        clf = filter_predictions(
            load_npz(predictions_file+'_clf.npz'), mapping)
        print("clf")
        args = acc.eval(clf, 5)
        print(xc_metrics.format(*args))
        print("knn")
        args = acc.eval(knn, 5)
        print(xc_metrics.format(*args))
        clf = sigmoid(clf)
        knn = sigmoid(knn)
        #clf = normalize(clf, norm='max')
        #knn = normalize(knn, norm='max')
        for beta in betas:
            predicted_labels = beta*clf + (1-beta)*knn
            args = acc.eval(predicted_labels, 5)
            print("beta: {0:.2f}".format(beta))
            print(xc_metrics.format(*args))
            if save:
                print("Saving predictions..")
                fname = os.path.join(
                        root, "score_beta_{0:.2f}.npz".format(beta))
                save_npz(fname, predicted_labels, compressed=False)
    else:
        predicted_labels = filter_predictions(
            load_npz(predictions_file+'.npz'), mapping)
        args = acc.eval(predicted_labels, 5)
        print(xc_metrics.format(*args))
        if save:
            print("Saving predictions..")
            fname = os.path.join(root, "score.npz")
            save_npz(fname, predicted_labels, compressed=False)


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
