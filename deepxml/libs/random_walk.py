import time
from numba import jit, njit, prange
from xclib.utils.sparse import normalize, retain_topk
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np


@njit(parallel=True, nogil=True)
def _random_walk(q_rng, q_lbl, l_rng, l_qry, walk_to=50, p_reset=0.2):
    n_nodes = l_rng.size - 1
    nbr_idx = np.empty((n_nodes, walk_to), dtype=np.int32)
    nbr_dat = np.empty((n_nodes, walk_to), dtype=np.float32)
    _row_idx = np.arange(n_nodes).reshape(1, -1)
    nbr_row = np.transpose(np.repeat(_row_idx, walk_to))
    for idx in prange(n_nodes):
        l_start, l_end = l_rng[idx], l_rng[idx+1]
        _s_query = l_qry[l_start: l_end]
        _qidx = np.random.choice(_s_query)
        q_start, q_end = q_rng[_qidx], q_rng[_qidx+1]
        p = np.divide(1, max((l_end - l_start)*(q_end - q_start), 1))
        nbr_idx[idx, 0] = np.random.choice(q_lbl[q_start: q_end])
        # nbr_dat[idx, 0] = p
        nbr_dat[idx, 0] = 1
        for walk in np.arange(1, walk_to):
            if np.random.random() < p_reset:
                l_start, l_end = l_rng[idx], l_rng[idx+1]
            else:
                _idx = nbr_idx[idx, walk-1]
                l_start, l_end = l_rng[_idx], l_rng[_idx+1]

            _s_query = l_qry[l_start: l_end]
            _qidx = np.random.choice(_s_query)
            q_start, q_end = q_rng[_qidx], q_rng[_qidx+1]
            p = np.divide(1, max((l_end - l_start)*(q_end - q_start), 1))
            nbr_idx[idx, walk] = np.random.choice(q_lbl[q_start: q_end])
            # nbr_dat[idx, walk] = p
            nbr_dat[idx, walk] = 1
    return nbr_row.flatten(), nbr_idx.flatten(), nbr_dat.flatten()


def random_walk(q_rng, q_lbl, l_rng, l_qry, walk_to=50, p_reset=0.2):
    return _random_walk(q_rng, q_lbl, l_rng, l_qry, walk_to, p_reset)


class RandomWalk(object):
    def __init__(self, walk_length=400, reset_prob=0.8, norm='l1', num_neighbours=5):
        self.walk_length = walk_length
        self.reset_prob = reset_prob
        self.num_neighbours = num_neighbours
        self.norm = norm

    def simulate(self, Y, seed=22):
        np.random.seed(seed)
        _, num_labels = Y.shape
        _Y = normalize(Y, norm=self.norm)

        temp = _Y
        q_lbl, q_rng = temp.indices, temp.indptr
        temp = _Y.transpose().tocsr()
        l_qry, l_rng = temp.indices, temp.indptr

        rows, cols, data = random_walk(
            q_rng, q_lbl, l_rng, l_qry,
            self.walk_length, self.reset_prob)

        graph = coo_matrix(
            (data, (rows, cols)), dtype=np.float32,
            shape=(num_labels, num_labels))
        graph.sum_duplicates()
        graph = graph.tocsr()
        graph.sort_indices()
        graph = retain_topk(graph, k=5)
        graph = normalize(graph, norm='l1')
        #graph.setdiag(1.0)
        return graph
