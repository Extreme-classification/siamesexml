import torch
import _pickle as pickle
import os
import sys
from scipy.sparse import lil_matrix
import numpy as np
from sklearn.preprocessing import normalize
from .dataset_base import DatasetBase
import xclib.data.data_utils as data_utils
import operator
from .lookup import Table
from .shortlist_handler import construct_handler


def construct_dataset(data_dir, fname_features, fname_labels, fname_lbl_features=None, 
                      data=None, model_dir='', mode='train', size_shortlist=-1,
                      normalize_features=True, normalize_labels=True,
                      keep_invalid=False, num_centroids=1,
                      feature_type='sparse', feature_indices=None,
                      label_indices=None, shortlist_method='static',
                      shorty=None):
    if size_shortlist == -1:
        return DatasetDense(
            data_dir, fname_features, fname_labels, fname_lbl_features, data,
            model_dir, mode, feature_indices, label_indices, keep_invalid,
            normalize_features, normalize_labels, feature_type)
    else:
        #  Construct dataset for sparse data
        return DatasetSparse(
            data_dir, fname_features, fname_labels, fname_lbl_features, 
            data, model_dir, mode, feature_indices, label_indices, keep_invalid,
            normalize_features, normalize_labels, size_shortlist, num_centroids,
            feature_type, shortlist_method, shorty)


class DatasetDense(DatasetBase):
    """Dataset to load and use XML-Datasets with full output space only
    Parameters
    ---------
    data_dir: str
        data files are stored in this directory
    fname_features: str
        feature file (libsvm or pickle)
    fname_labels: str
        labels file (libsvm or pickle)    
    data: dict, optional, default=None
        Read data directly from this obj rather than files
        Files are ignored if this is not None
        Keys: 'X', 'Y'
    model_dir: str, optional, default=''
        Dump data like valid labels here
    mode: str, optional, default='train'
        Mode of the dataset
    feature_indices: np.ndarray or None, optional, default=None
        Train with selected features only
    label_indices: np.ndarray or None, optional, default=None
        Train for selected labels only
    keep_invalid: bool, optional, default=False
        Don't touch data points or labels
    normalize_features: bool, optional, default=True
        Normalize data points to unit norm
    normalize_lables: bool, optional, default=False
        Normalize labels to convert in probabilities
        Useful in-case on non-binary labels
    feature_type: str, optional, default='sparse'
        sparse or dense features
    label_type: str, optional, default='dense'
        sparse (i.e. with shortlist) or dense (OVA) labels
    """

    def __init__(self, data_dir, fname_features, fname_labels, fname_lbl_features=None, 
                 data=None, model_dir='', mode='train', feature_indices=None,
                 label_indices=None, keep_invalid=False, normalize_features=True,
                 normalize_labels=False, feature_type='sparse', label_type='dense'):
        """
            Expects 'libsvm' format with header
            Args:
                data_file: str: File name for the set
            Can Support datasets w/o any label
        """
        super().__init__(data_dir, fname_features, fname_labels, data,
                         model_dir, mode, feature_indices, label_indices,
                         keep_invalid, normalize_features, normalize_labels,
                         feature_type, label_type)
        if self.mode == 'train':
            # Remove samples w/o any feature or label
            self._remove_samples_wo_features_and_labels()
        if not keep_invalid and self.labels._valid:
            # Remove labels w/o any positive instance
            self._process_labels(model_dir)
        self.feature_type = feature_type
        # TODO Take care of this select and padding index
        self.label_padding_index = self.num_labels

    def __getitem__(self, index):
        """
            Get features and labels for index
            Args:
                index: for this sample
            Returns:
                features: : non zero entries
                labels: : numpy array

        """
        x = self.features[index]
        y = self.labels[index]
        return x, y


class DatasetSparse(DatasetBase):
    """Dataset to load and use XML-Datasets with shortlist
    Parameters
    ---------
    data_dir: str
        data files are stored in this directory
    fname_features: str
        feature file (libsvm or pickle)
    fname_labels: str
        labels file (libsvm or pickle)    
    fname_lbl_features: str
        features for each label (libsvm or pickle)    
    data: dict, optional, default=None
        Read data directly from this obj rather than files
        Files are ignored if this is not None
        Keys: 'X', 'Y'
    model_dir: str, optional, default=''
        Dump data like valid labels here
    mode: str, optional, default='train'
        Mode of the dataset
    feature_indices: np.ndarray or None, optional, default=None
        Train with selected features only
    label_indices: np.ndarray or None, optional, default=None
        Train for selected labels only
    keep_invalid: bool, optional, default=False
        Don't touch data points or labels
    normalize_features: bool, optional, default=True
        Normalize data points to unit norm
    normalize_lables: bool, optional, default=False
        Normalize labels to convert in probabilities
        Useful in-case on non-binary labels
    num_centroids: int, optional, default=1
        Multiple representations for labels
    feature_type: str, optional, default='sparse'
        sparse or dense features
    shortlist_type: str, optional, default='static'
        type of shortlist (static or dynamic)
    shorty: obj, optional, default=None
        Useful in-case of dynamic shortlist
    label_type: str, optional, default='dense'
        sparse (i.e. with shortlist) or dense (OVA) labels
    shortlist_in_memory: boolean, optional, default=True
        Keep shortlist in memory if True otherwise keep on disk
    """

    def __init__(self, data_dir, fname_features, fname_labels, fname_lbl_features, 
                 data=None, model_dir='', mode='train', feature_indices=None,
                 label_indices=None, keep_invalid=False,
                 normalize_features=True, normalize_labels=False,
                 size_shortlist=-1, num_centroids=1, feature_type='sparse',
                 shortlist_method='static', shorty=None, label_type='sparse',
                 shortlist_in_memory=True, corruption=200):
        """
            Expects 'libsvm' format with header
            Args:
                data_file: str: File name for the set
        """
        super().__init__(data_dir, fname_features, fname_labels, data,
                         model_dir, mode, feature_indices, label_indices,
                         keep_invalid, normalize_features, normalize_labels,
                         feature_type, label_type)
        if self.labels is None:
            NotImplementedError(
                "No support for shortlist w/o any label, \
                    consider using dense dataset.")
        self.feature_type = feature_type
        self.num_centroids = num_centroids
        self.shortlist_in_memory = shortlist_in_memory
        # Load label features
        # Assumption: fixed pre-supplied label features
        self.lbl_features = self.load_label_features(fname_lbl_features, data, model_dir, label_indices)
        self.size_shortlist = size_shortlist
        self.shortlist_method = shortlist_method
        if self.mode == 'train':
            self._remove_samples_wo_features_and_labels()
        
        if not keep_invalid:
            # Remove labels w/o any positive instance
            self._process_labels(model_dir)

        self.shortlist = construct_handler(
            shortlist_method=shortlist_method,
            num_labels=self.num_labels,
            shortlist=None,
            model_dir=model_dir,
            mode=mode,
            size_shortlist=size_shortlist,
            in_memory=shortlist_in_memory,
            shorty=shorty,
            corruption=corruption)
        self.use_shortlist = True if self.size_shortlist > 0 else False
        self.label_padding_index = self.num_labels

    def load_label_features(self, fname, data, model_dir, label_indices):
        lbl_features = None
        if fname is not None or data['Yf'] is not None:
            # label embeddings will be kept in model directory
            lbl_features = self.load_features(
                data_dir=os.path.dirname(model_dir), fname=fname,
                X=data['Yf'], normalize_features=False,
                feature_type='dense')
            if label_indices is not None:
                label_indices = np.loadtxt(label_indices, dtype=np.int32)
                lbl_features.index_select(label_indices, axis=0)
            print("Label embedding loaded.")
        return lbl_features

    def update_shortlist(self, shortlist, dist, fname='tmp', idx=-1):
        """Update label shortlist for each instance
        """
        self.shortlist.update_shortlist(shortlist, dist, fname, idx)

    def save_shortlist(self, fname):
        """Save label shortlist and distance for each instance
        """
        self.shortlist.save_shortlist(fname)

    def load_shortlist(self, fname):
        """Load label shortlist and distance for each instance
        """
        self.shortlist.load_shortlist(fname)

    def _process_labels_train(self, data_obj):
        """Process labels for train data
            - Remove labels without any training instance
        """
        data_obj['num_labels'] = self.num_labels
        ind_0 = self.labels.get_valid()
        ind_1 = np.where(np.square(self.lbl_features.data).sum(axis=1))[0] #nnz vectors
        valid_labels = np.intersect1d(ind_0, ind_1)
        self._valid_labels = valid_labels
        print("#Valid labels: {}".format(len(valid_labels)))
        self.lbl_features.index_select(valid_labels, axis=0)
        self.labels.index_select(valid_labels, axis=1)
        data_obj['valid_labels'] = valid_labels

    def _process_labels_retrain_w_shortlist(self, data_obj):
        """Process labels for retrain with shortlist
        Useful for training labels shortlist after OVA training
        """
        super()._process_labels_predict(data_obj)

    def _process_labels(self, model_dir):
        """
            Process labels to handle labels without any training instance;
        """
        data_obj = {}
        fname = os.path.join(
            model_dir, 'labels_params.pkl' if self._split is None else
            "labels_params_split_{}.pkl".format(self._split))
        if self.mode == 'train':
            self._process_labels_train(data_obj)
            pickle.dump(data_obj, open(fname, 'wb'))
        elif self.mode == 'retrain_w_shortlist':
            data_obj = pickle.load(open(fname, 'rb'))
            self._process_labels_retrain_w_shortlist(
                data_obj)
            pickle.dump(data_obj, open(fname, 'wb'))
        else:
            data_obj = pickle.load(open(fname, 'rb'))
            self._process_labels_predict(data_obj)

    def get_shortlist(self, index):
        """
            Get data with shortlist for given data index
        """
        pos_labels, _ = self.labels[index]
        return self.shortlist.get_shortlist(index, pos_labels)

    def __getitem__(self, index):
        """Get features and labels for index
        Arguments
        ---------
        index: int
            data for this index
        Returns
        -------
        features: np.ndarray or tuple
            for dense: np.ndarray
            for sparse: feature indices and their weights
        labels: tuple
            shortlist: label indices in the shortlist
            labels_mask: 1 for relevant; 0 otherwise
            dist: distance (used during prediction only)
        """
        x = self.features[index]
        y = self.get_shortlist(index)
        return x, y
