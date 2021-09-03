from xclib.data.features import DenseFeatures, SparseFeatures
from xclib.data import data_utils


def construct(data_dir, fname, X=None, normalize=False, _type='sparse'):
    """Construct feature class based on given parameters
    Arguments
    ----------
    data_dir: str
        data directory
    fname: str
        load data from this file
    X: csr_matrix or None, optional, default=None
        data is already provided
    normalize: boolean, optional, default=False
        Normalize the data or not
    _type: str, optional, default=sparse
        -sparse
        -dense
        -sequential
    """
    if _type == 'sparse':
        return _SparseFeatures(data_dir, fname, X, normalize)
    elif _type == 'dense':
        return DenseFeatures(data_dir, fname, X, normalize)
    elif _type == 'sequential':
        raise NotImplementedError()
    else:
        raise NotImplementedError("Unknown feature type")


class _SparseFeatures(SparseFeatures):
    """Class for sparse features

    * Difference: treat 0 as padding index
    
    Arguments
    ----------
    data_dir: str
        data directory
    fname: str
        load data from this file
    X: csr_matrix or None, optional, default=None
        data is already provided
    normalize: boolean, optional, default=False
        Normalize the data or not
    """

    def __init__(self, data_dir, fname, X=None, normalize=False):
        super().__init__(data_dir, fname, X, normalize)

    def __getitem__(self, index):
        # Treat idx:0 as Padding
        x = self.X[index].indices + 1
        w = self.X[index].data
        return x, w
