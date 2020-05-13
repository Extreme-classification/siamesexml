import numpy as np
import _pickle as pickle
import operator
from scipy.sparse import csr_matrix, diags
from .lookup import Table
import xclib.utils.ann as ANN
from xclib.utils.dense import compute_centroid
from xclib.utils.sparse import topk, csr_from_arrays
import os
import numba
import math
from operator import itemgetter
from libs.clustering import Cluster


class Shortlist(object):
    """Get nearest neighbors using brute or HNSW algorithm
    Parameters
    ----------
    method: str
        brute or hnsw
    num_neighbours: int
        number of neighbors
    M: int
        HNSW M (Usually 100)
    efC: int
        construction parameter (Usually 300)
    efS: int
        search parameter (Usually 300)
    num_threads: int, optional, default=-1
        use multiple threads to cluster
    """

    def __init__(self, method, num_neighbours, M, efC, efS, num_threads=24):
        self.method = method
        self.num_neighbours = num_neighbours
        self.M = M
        self.efC = efC
        self.efS = efS
        self.num_threads = num_threads
        self.index = None
        self._construct()

    def _construct(self):
        if self.method == 'brute':
            self.index = ANN.NearestNeighbor(
                num_neighbours=self.num_neighbours,
                method='brute',
                num_threads=self.num_threads
            )
        elif self.method == 'hnsw':
            self.index = ANN.HNSW(
                M=self.M,
                efC=self.efC,
                efS=self.efS,
                num_neighbours=self.num_neighbours,
                num_threads=self.num_threads
            )
        else:
            print("Unknown NN method!")

    def fit(self, data):
        self.index.fit(data)

    def query(self, data, *args, **kwargs):
        indices, distances = self.index.predict(data, *args, **kwargs)
        return indices, 1-distances

    def save(self, fname):
        self.index.save(fname)

    def load(self, fname):
        self.index.load(fname)

    def reset(self):
        # TODO Do we need to delete it!
        del self.index
        self._construct()

    @property
    def model_size(self):
        # size on disk; see if there is a better solution
        import tempfile
        with tempfile.NamedTemporaryFile() as tmp:
            self.index.save(tmp.name)
            _size = os.path.getsize(tmp.name)/math.pow(2, 20)
        return _size


class ShortlistCentroids(Shortlist):
    """Get nearest labels using KCentroids
    * centroid(l) = mean_{i=1}^{N}{x_i*y_il}
    * brute or HNSW algorithm for search
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
    num_clusters: int, optional, default=1
        cluster instances => multiple representatives for chosen labels
    threshold: int, optional, default=5000
        cluster instances if a label appear in more than 'threshold'
        training points
    """
    def __init__(self, method='hnsw', num_neighbours=300, M=100, efC=300,
                 efS=300, num_threads=24, space='cosine', verbose=True,
                 num_clusters=1, threshold=5000):
        super().__init__(method, num_neighbours, M, efC, efS, num_threads)
        self.num_clusters = num_clusters
        self.space = space
        self.pad_ind = -1
        self.mapping = None
        self.ext_head = None
        self.threshold = threshold

    def _cluster_multiple_rep(self, features, labels, label_centroids,
                              multi_centroid_indices):
        embedding_dims = features.shape[1]
        _cluster_obj = Cluster(
            indices=multi_centroid_indices,
            embedding_dims=embedding_dims,
            num_clusters=self.num_clusters,
            max_iter=50, n_init=2, num_threads=-1)
        _cluster_obj.fit(features, labels)
        label_centroids = np.vstack(
            [label_centroids, _cluster_obj.predict()])
        return label_centroids

    def process_multiple_rep(self, features, labels, label_centroids):
        freq = np.array(labels.sum(axis=0)).ravel()
        if np.max(freq) > self.threshold and self.num_clusters > 1:
            self.ext_head = np.where(freq >= self.threshold)[0]
            print("Found {} super-head labels".format(len(self.ext_head)))
            self.mapping = np.arange(label_centroids.shape[0])
            for idx in self.ext_head:
                self.mapping = np.append(
                    self.mapping, [idx]*self.num_clusters)
            return self._cluster_multiple_rep(
                features, labels, label_centroids, self.ext_head)
        else:
            return label_centroids

    def fit(self, features, labels, *args, **kwargs):
        self.pad_ind = labels.shape[1]
        label_centroids = compute_centroid(features, labels, reduction='mean')
        label_centroids = self.process_multiple_rep(
            features, labels, label_centroids)
        norms = np.sum(np.square(label_centroids), axis=1)
        super().fit(label_centroids)

    def query(self, data, *args, **kwargs):
        indices, sim = super().query(data, *args, **kwargs)
        return self._remap(indices, sim)

    def _remap(self, indices, sims):
        if self.mapping is None:
            return indices, sims
        print("Re-mapping code not optimized")
        mapped_indices = np.full_like(indices, self.pad_ind)
        # minimum similarity for padding index
        mapped_sims = np.full_like(sims, -1000.0)
        for idx, (ind, sim) in enumerate(zip(indices, sims)):
            _ind, _sim = self._remap_one(ind, sim)
            mapped_indices[idx, :len(_ind)] = _ind
            mapped_sims[idx, :len(_sim)] = _sim
        return mapped_indices, mapped_sims

    def _remap_one(self, indices, vals, _func=max, _limit=-1000):
        """
            Remap multiple centroids to original labels
        """
        indices = map(lambda x: self.mapping[x], indices)
        _dict = dict({})
        for idx, ind in enumerate(indices):
            _dict[ind] = _func(_dict.get(ind, _limit), vals[idx])
        indices, values = zip(*_dict.items())
        return np.fromiter(indices, dtype=np.int64), \
            np.fromiter(values, dtype=np.float32)

    def load(self, fname):
        temp = pickle.load(open(fname+".metadata", 'rb'))
        self.pad_ind = temp['pad_ind']
        self.mapping = temp['mapping']
        self.ext_head = temp['ext_head']
        super().load(fname+".index")

    def save(self, fname):
        metadata = {
            'pad_ind': self.pad_ind,
            'mapping': self.mapping,
            'ext_head': self.ext_head
        }
        pickle.dump(metadata, open(fname+".metadata", 'wb'))
        super().save(fname+".index")

    def purge(self, fname):
        # purge files from disk
        if os.path.isfile(fname+".index"):
            os.remove(fname+".index")
        if os.path.isfile(fname+".metadata"):
            os.remove(fname+".metadata")


class ShortlistEmbeddings(Shortlist):
    """Get nearest labels using their embeddings
    * brute or HNSW algorithm for search
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
                 efS=300, space='cosine', verbose=True, num_threads=24):
        super().__init__(method, num_neighbours, M, efC, efS, num_threads)

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
    def __init__(self, method='hnsw', num_neighbours=500,
                 M={'kcentroid': 100, 'kembed': 100},
                 efC={'kcentroid': 300, 'kembed': 300},
                 efS={'kcentroid': 300, 'kembed': 300},
                 num_threads=24, space='cosine', verbose=True,
                 num_clusters=1, pad_val=-10000, gamma=0.075):
        self.kcentroid = ShortlistCentroids(
            method=method, num_neighbours=efS['kcentroid'],
            M=M['kcentroid'], efC=efC['kcentroid'], efS=efS['kcentroid'],
            num_threads=num_threads, space=space, verbose=True)
        self.kembed = ShortlistEmbeddings(
            method=method, num_neighbours=efS['kembed'], M=M['kembed'],
            efC=efC['kembed'], efS=efS['kembed'], num_threads=num_threads,
            space=space, verbose=True)
        self.num_labels = None
        self.num_neighbours = num_neighbours
        self.pad_val = pad_val
        self.pad_ind = -1
        self.gamma = gamma

    def fit(self, X, Y, Yf, *args, **kwargs):
        assert Y.shape[1] == Yf.shape[0], \
            "#label should match in kembed and kcentroid"
        self.pad_ind = Y.shape[1]
        self.num_labels = Y.shape[1]
        self.index_kcentroid.fit(X, Y)
        self.index_kembed.fit(Yf)

    def merge(self, indices_kcentroid, sim_kcentroid,
              indices_kembed, sim_kembed):
        _shape = (len(indices_kcentroid), self.num_labels)
        _kcentroid = csr_from_arrays(indices_kcentroid, sim_kcentroid, _shape)
        _kembed = csr_from_arrays(indices_kembed, sim_embed, _shape)
        temp = (gamma*_kembed + (1-gamma)*_kcentroid).tolil()
        indices = [np.array(item, dtype=np.int64) for item in temp.rows]
        sim = [np.array(item, dtype=np.float32) for item in temp.data]
        return indices, sim

    def query(self, data):
        indices_kcentroid, sim_kcentroid = self.index_kcentroid.query(data)
        indices_kembed, sim_kembed = self.index_kembed.query(data)
        indices, sim = self.merge(
            indices_kcentroid, sim_kcentroid, indices_kembed, sim_kembed)
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

    def purge(self, fname):
        # purge files from disk
        self.kembed.purge(fname)
        self.kcentroid.purge(fname)

    def load(self, fname):
        obj = pickle.load(
            open(fname+".metadata", 'rb'))
        self.num_labels = obj['num_labels']
        self.pad_ind = obj['pad_ind']
        self.gamma = obj['gamma']
        self.kcentroid.load(fname+'.kcentroid')
        self.kembed.load(fname+'.kembed')

    def reset(self):
        self.kcentroid.reset()
        self.kembed.reset()
