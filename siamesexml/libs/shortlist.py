import numpy as np
import pickle
from xclib.utils.sparse import topk, csr_from_arrays
import os
import numba as nb
import scipy.sparse as sp
from xclib.utils.graph import RandomWalk, normalize_graph
from xclib.utils.shortlist import Shortlist
from xclib.utils.shortlist import ShortlistCentroids
from xclib.utils.shortlist import ShortlistInstances


@nb.njit(parallel=True)
def map_dense(ind, mapping):
    out = np.full_like(ind, fill_value=0)
    nr, nc = ind.shape
    for i in nb.prange(nr):
        for j in range(nc):
            out[i, j] = mapping[ind[i, j]]
    return out


def normalize_graph(mat, axis=1):
    diags = mat.diagonal()
    print("Zero Diagonals:", np.sum(diags==0))
    col_nnz = np.sqrt(1/np.ravel(mat.sum(axis=0)))
    row_nnz = np.sqrt(1/np.ravel(mat.sum(axis=1)))
    c_diags = sp.diags(col_nnz)
    r_diags = sp.diags(row_nnz)
    mat = r_diags.dot(mat).dot(c_diags)
    mat.eliminate_zeros()
    return mat


class ShortlistMIPS(Shortlist):
    """Get nearest labels using their embeddings
    * brute or HNSW algorithm for search
    * option to process label representations with label correlation matrix

    Parameters
    ----------
    method: str, optional, default='hnsw'
        brute or hnsw
    num_neighbours: int
        number of neighbors (same as efS)
        * may be useful if the NN search retrieve less number of labels
        * typically doesn't happen with HNSW etc.
    M: int, optional, default=100
        HNSW M (Usually 100)
    efC: int, optional, default=300
        construction parameter (Usually 300)
    efS: int, optional, default=300
        search parameter (Usually 300)
    num_threads: int, optional, default=18
        use multiple threads to cluster
    space: str, optional, default='cosine'
        metric to use while quering
    verbose: boolean, optional, default=True
        print progress
    """
    def __init__(self, method='hnsw', num_neighbours=300, M=100, efC=300,
                 efS=300, space='cosine', verbose=True, num_threads=16):
        super().__init__(method, num_neighbours, M, efC, efS, num_threads)
        self.valid_indices = None

    def fit(self, X, Y=None, *args, **kwargs):
        ind = np.where(np.square(X).sum(axis=1) > 0)[0]
        if Y is not None:
            ind_1 = np.where(np.array(Y.sum(axis=0)).ravel() > 0)[0]
            ind = np.intersect1d(ind, ind_1)
            if len(ind) <= len(X):
                self.valid_indices = ind
                X = X[self.valid_indices]
                if Y is not None:
                    print("Doing random walk!")
                    cooc = RandomWalk(Y[:, self.valid_indices]).simulate(400, 0.8, 10)
                    cooc = normalize_graph(cooc)
                    X = cooc @ X
        else:
            print("Removing invalid.")
            self.valid_indices = ind
            X = X[self.valid_indices]
        super().fit(X)

    def query(self, X, *args, **kwargs):
        ind, sim = super().query(X)
        if self.valid_indices is not None:
            ind = map_dense(ind, self.valid_indices)
        return ind, sim

    def save(self, fname):
        metadata = {
            'valid_indices': self.valid_indices,
        }
        pickle.dump(metadata, open(fname+".metadata", 'wb'))
        super().save(fname+".index")

    def load(self, fname):
        self.index.load(fname+".index")
        obj = pickle.load(
            open(fname+".metadata", 'rb'))
        self.valid_indices = obj['valid_indices']

    def purge(self, fname):
        # purge files from disk
        if os.path.isfile(fname+".index"):
            os.remove(fname+".index")


class ShortlistEnsemble(object):
    """Get nearest labels using label embeddings and label centroids
    * Give less weight to KNN (typically 0.1 or 0.075)
    * brute or HNSW algorithm for search
    Parameters
    ----------
    method: str, optional, default='hnsw'
        brute or hnsw
    num_neighbours: int, optional, default=500
        number of labels to keep for each instance
        * will pad using pad_ind and pad_val in case labels
          are less than num_neighbours
    M: int, optional, default=100
        HNSW M (Usually 100)
    efC: dict, optional, default={'kcentroid': 300, 'knn': 50}
        construction parameter for kcentroid and knn
        * Usually 300 for kcentroid and 50 for knn
    efS: dict, optional, default={'kcentroid': 300, 'knn': 500}
        search parameter for kcentroid and knn
        * Usually 300 for kcentroid and 500 for knn
    num_threads: int, optional, default=24
        use multiple threads to cluster
    space: str, optional, default='cosine'
        metric to use while quering
    verbose: boolean, optional, default=True
        print progress
    num_clusters: int, optional, default=1
        cluster instances => multiple representatives for chosen labels
    pad_val: int, optional, default=-10000
        value for padding indices
        - Useful as documents may have different number of nearest labels
    gamma: float, optional, default=0.075
        weight for embedding shortlist.
        * final shortlist => gamma * kembed + (1-gamma) * kcentroid
    """
    def __init__(self, method='hnsw', num_neighbours={'ens': 500,
                 'kcentroid': 400, 'knn': 300, 'kembed': 300},
                 M={'kcentroid': 100, 'kembed': 100, 'knn': 50},
                 efC={'kcentroid': 300, 'kembed': 300, 'knn': 50},
                 efS={'kcentroid': 300, 'kembed': 300, 'knn': 100},
                 num_threads=24, space='cosine', verbose=True,
                 num_clusters=1, pad_val=-10000, gamma=0.075, 
                 use_knn=False):
        self.kcentroid = ShortlistCentroids(
            method=method, num_neighbours=efS['kcentroid'],
            M=M['kcentroid'], efC=efC['kcentroid'], efS=efS['kcentroid'],
            num_threads=num_threads, space=space, verbose=True)
        self.kembed = ShortlistMIPS(
            method=method, num_neighbours=efS['kembed'], M=M['kembed'],
            efC=efC['kembed'], efS=efS['kembed'], num_threads=num_threads,
            space=space, verbose=True)
        self.knn = None
        if use_knn:        
            self.knn = ShortlistInstances(
                method=method, num_neighbours=num_neighbours['knn'], M=M['knn'],
                efC=efC['knn'], efS=efS['knn'], num_threads=num_threads,
                space=space, verbose=True)

        self.num_labels = None
        self.num_neighbours = num_neighbours['ens']
        self.pad_val = pad_val
        self.pad_ind = -1
        self.gamma = gamma

    def fit(self, X, Y, Yf, *args, **kwargs):
        assert Y.shape[1] == Yf.shape[0], \
            "#label should match in kembed and kcentroid"
        self.pad_ind = Y.shape[1]
        self.num_labels = Y.shape[1]
        self.kcentroid.fit(X, Y)
        self.kembed.fit(Yf, Y)
        if self.knn is not None:
            self.knn.fit(X, Y)

    def merge(self, indices, similarities, num_instances):
        _shape = (num_instances, self.num_labels+1)
        short_kcentroid = csr_from_arrays(
            indices['kcentroid'], similarities['kcentroid'], _shape)
        short_kembed = csr_from_arrays(
            indices['kembed'], similarities['kembed'], _shape)
        # this could be further optimized, if required
        short = short_kcentroid + short_kembed
        if self.knn is not None:
            short_knn = csr_from_arrays(
                indices['knn'], similarities['knn'], _shape)
            indices, sim = topk(
                (self.gamma*short_knn + (1-self.gamma)*short),
                k=self.num_neighbours, pad_ind=self.pad_ind,
                pad_val=self.pad_val, return_values=True)
        else:
            indices, sim = topk(
                short, k=self.num_neighbours, pad_ind=self.pad_ind,
                pad_val=self.pad_val, return_values=True)
        return indices, sim

    def query(self, data):
        indices, sim = {}, {}
        indices['kcentroid'], sim['kcentroid'] = self.kcentroid.query(data)
        indices['kembed'], sim['kembed'] = self.kembed.query(data)        
        if self.knn is not None:
            indices['knn'], sim['knn'] = self.knn.query(data)
        indices, sim = self.merge(indices, sim, len(data))
        return indices, sim

    def save(self, fname):
        # Returns the filename on disk; useful in purging checkpoints
        pickle.dump(
            {'num_labels': self.num_labels,
             'pad_ind': self.pad_ind,
             'gamma': self.gamma},
            open(fname+".metadata", 'wb'))
        self.kcentroid.save(fname+'.kcentroid')
        self.kembed.save(fname+'.kembed')
        if self.knn is not None:
            self.knn.save(fname+'.knn')

    def purge(self, fname):
        # purge files from disk
        self.kembed.purge(fname)
        self.kcentroid.purge(fname)
        if self.knn is not None:
            self.knn.purge(fname)

    def load(self, fname):
        obj = pickle.load(
            open(fname+".metadata", 'rb'))
        self.num_labels = obj['num_labels']
        self.pad_ind = obj['pad_ind']
        self.gamma = obj['gamma']
        self.kcentroid.load(fname+'.kcentroid')
        self.kembed.load(fname+'.kembed')
        if self.knn is not None:
            self.knn.load(fname+'.knn')

    def reset(self):
        self.kcentroid.reset()
        self.kembed.reset()
        if self.knn is not None:
            self.knn.reset()

    @property
    def model_size(self):
        _size = self.kcentroid.model_size + self.kembed.model_size
        if self.knn is not None:
            _size = _size + self.knn.model_size
        return _size
