import torch
import torch.nn as nn
import numpy as np
import math
import models.transform_layer as transform_layer
import models.linear_layer as linear_layer
import torch.nn.functional as F
from collections import OrderedDict
import pickle
import os


__author__ = 'KD'


def _to_device(x, device):
    if x is None:
        return None
    elif isinstance(x, (tuple, list)):
        out = []
        for item in x:
            out.append(_to_device(item, device))
        return out
    else:
        return x.to(device)


class DeepXMLBase(nn.Module):
    """DeepXMLBase: Base class for DeepXML architecture

    * Identity op as classifier by default
    (derived class should implement it's own classifier)
    * embedding and classifier shall automatically transfer
    the vector to the appropriate device

    Arguments:
    ----------
    vocabulary_dims: int
        number of tokens in the vocabulary
    embedding_dims: int
        size of word/token representations
    trans_config: list of strings
        configuration of the transformation layer
    padding_idx: int, default=0
        padding index in words embedding layer
    """

    def __init__(self, config, device="cuda:0"):
        super(DeepXMLBase, self).__init__()
        self.transform = self._construct_transform(config)
        self.classifier = self._construct_classifier()
        self.device = torch.device(device)

    def _construct_classifier(self):
        return nn.Identity()

    def _construct_transform(self, trans_config):
        if trans_config is None:
            return None
        return transform_layer.Transform(
            transform_layer.get_functions(trans_config))

    @property
    def representation_dims(self):
        return self._repr_dims

    @representation_dims.setter
    def representation_dims(self, dims):
        self._repr_dims = dims

    def encode(self, x):
        """Forward pass
        * Assumes features are dense if x_ind is None

        Arguments:
        -----------
        x: tuple
            torch.FloatTensor or None
                (sparse features) contains weights of features as per x_ind or
                (dense features) contains the dense representation of a point
            torch.LongTensor or None
                contains indices of features (sparse or seqential features)

        Returns
        -------
        out: logits for each label
        """
        return self.transform(
            _to_device(x, self.device))

    def forward(self, batch_data, *args):
        """Forward pass
        * Assumes features are dense if X_w is None
        * By default classifier is identity op

        Arguments:
        -----------
        batch_data: dict
            * 'X': torch.FloatTensor
                feature weights for given indices or dense rep.
            * 'X_ind': torch.LongTensor
                feature indices (LongTensor) or None

        Returns
        -------
        out: logits for each label
        """
        return self.classifier(
            self.encode(batch_data['X'], batch_data['X_ind']))

    def initialize(self, x):
        """Initialize embeddings from existing ones
        Parameters:
        -----------
        word_embeddings: numpy array
            existing embeddings
        """
        self.transform.initialize(x)

    def to(self):
        """Send layers to respective devices
        """
        self.transform.to()
        self.classifier.to()

    def purge(self, fname):
        if os.path.isfile(fname):
            os.remove(fname)

    @property
    def num_params(self, ignore_fixed=False):
        if ignore_fixed:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    @property
    def model_size(self):  # Assumptions: 32bit floats
        return self.num_params * 4 / math.pow(2, 20)

    def __repr__(self):
        return f"{self.embeddings}\n(Transform): {self.transform}"


class DeepXMLf(DeepXMLBase):
    """DeepXMLf: Network for DeepXML's architecture
    with fully-connected o/p layer (a.k.a 1-vs.-all in literature)

    Allows additional transform layer to transform features from the
    base class. e.g. base class can handle intermediate rep. and transform
    could be used to the intermediate rep. from base class
    """

    def __init__(self, params):
        self.num_labels = params.num_labels
        transform_config_dict = transform_layer.fetch_json(
            params.arch, params)
        trans_config_coarse = transform_config_dict['transform_coarse']
        self.representation_dims = int(
            transform_config_dict['representation_dims'])
        super(DeepXMLf, self).__init__(trans_config_coarse)
        if params.freeze_intermediate:
            print("Freezing intermediate model parameters!")
            for params in self.transform.parameters():
                params.requires_grad = False
        trans_config_fine = transform_config_dict['transform_fine']
        self.transform_fine = self._construct_transform(
            trans_config_fine)

    def encode_fine(self, x):
        """Forward pass (assumes input is coarse computation)

        Arguments:
        -----------
        x: torch.FloatTensor
            (sparse features) contains weights of features as per x_ind or
            (dense features) contains the dense representation of a point

        Returns
        -------
        out: torch.FloatTensor
            encoded x with fine encoder
        """
        return self.transform_fine(_to_device(x, self.device))

    def encode(self, x, x_ind=None, bypass_fine=False):
        """Forward pass
        * Assumes features are dense if x_ind is None

        Arguments:
        -----------
        x: torch.FloatTensor
            (sparse features) contains weights of features as per x_ind or
            (dense features) contains the dense representation of a point
        x_ind: torch.LongTensor or None, optional (default=None)
            contains indices of features (sparse features)
        bypass_fine: boolean, optional (default=False)
            Return coarse features or not

        Returns
        -------
        out: logits for each label
        """
        encoding = super().encode((x, x_ind))
        return encoding if bypass_fine else self.transform_fine(encoding)

    def forward(self, batch_data, bypass_coarse=False):
        """Forward pass
        * Assumes features are dense if X_w is None
        * By default classifier is identity op

        Arguments:
        -----------
        batch_data: dict
            * 'X': torch.FloatTensor
                feature weights for given indices or dense rep.
            * 'X_ind': torch.LongTensor
                feature indices (LongTensor) or None

        Returns
        -------
        out: logits for each label
        """
        if bypass_coarse:
            return self.classifier(
                self.encode_fine(batch_data['X']))
        else:
            return self.classifier(
                self.encode(batch_data['X'], batch_data['X_ind']))

    def _construct_classifier(self):
        return linear_layer.Linear(
            input_size=self.representation_dims,
            output_size=self.num_labels,  # last one is padding index
            bias=True
        )

    def get_token_embeddings(self):
        return self.transform.get_token_embeddings()

    def save_intermediate_model(self, fname):
        torch.save(self.transform.state_dict(), fname)

    def load_intermediate_model(self, fname):
        self.transform.load_state_dict(torch.load(fname))

    def to(self):
        """Send layers to respective devices
        """
        self.transform_fine.to()
        super().to()

    def initialize_classifier(self, weight, bias=None):
        """Initialize classifier from existing weights

        Arguments:
        -----------
        weight: numpy.ndarray
        bias: numpy.ndarray or None, optional (default=None)
        """
        self.classifier.weight.data.copy_(torch.from_numpy(weight))
        if bias is not None:
            self.classifier.bias.data.copy_(
                torch.from_numpy(bias).view(-1, 1))

    def get_clf_weights(self):
        """Get classifier weights
        """
        return self.classifier.get_weights()

    def __repr__(self):
        s = f"{self.transform}\n"
        s += f"(Transform fine): {self.transform_fine}"
        s += f"\n(Classifier): {self.classifier}\n"
        return s


class DeepXMLs(DeepXMLBase):
    """DeepXMLs: DeepXML architecture to be trained with
                 a label shortlist
    * Allows additional transform layer for features
    """

    def __init__(self, params):
        self.num_labels = params.num_labels
        self.label_padding_index = params.label_padding_index
        transform_config_dict = transform_layer.fetch_json(
            params.arch, params)
        trans_config_coarse = transform_config_dict['transform_coarse_doc']
        self.representation_dims = int(
            transform_config_dict['representation_dims'])
        self.metric = params.metric
        super(DeepXMLs, self).__init__(trans_config_coarse)
        if params.freeze_intermediate:
            for params in self.transform.parameters():
                params.requires_grad = False
        trans_config_fine = transform_config_dict['transform_fine']
        self.transform_fine = self._construct_transform(
            trans_config_fine)

    def save_intermediate_model(self, fname):
        out = {'net': self.transform.state_dict()}
        torch.save(out, fname)

    def load_intermediate_model(self, fname):
        try: #Shared architecture
            self.transform.load_state_dict(torch.load(fname)['net'])
        except KeyError:
            self.transform.load_state_dict(torch.load(fname)['document_net'])

    def encode_fine(self, x):
        """Forward pass (assumes input is coarse computation)

        Arguments:
        -----------
        x: torch.FloatTensor
            (sparse features) contains weights of features as per x_ind or
            (dense features) contains the dense representation of a point

        Returns
        -------
        out: torch.FloatTensor
            encoded x with fine encoder
        """
        return self.transform_fine(_to_device(x, self.device))

    def encode_document(self, x, x_ind=None, return_coarse=False):
        return self.encode(x, x_ind, return_coarse)

    def encode_label(self, x, x_ind=None, return_coarse=False):
        return self.encode(x, x_ind, return_coarse)

    def encode(self, x, x_ind=None, bypass_fine=False):
        #TODO: Implement stuff for non-shared arch
        """Forward pass
        * Assumes features are dense if x_ind is None

        Arguments:
        -----------
        x: torch.FloatTensor
            (sparse features) contains weights of features as per x_ind or
            (dense features) contains the dense representation of a point
        x_ind: torch.LongTensor or None, optional (default=None)
            contains indices of features (sparse features)
        bypass_fine: boolean, optional (default=False)
            Return coarse features or not

        Returns
        -------
        out: logits for each label
        """
        encoding = super().encode((x, x_ind))
        return encoding if bypass_fine else self.transform_fine(encoding)

    def forward(self, batch_data, bypass_coarse=False):
        """Forward pass
        * Assumes features are dense if X_w is None
        * By default classifier is identity op

        Arguments:
        -----------
        batch_data: dict
            * 'X': torch.FloatTensor
                feature weights for given indices or dense rep.
            * 'X_ind': torch.LongTensor
                feature indices (LongTensor) or None

        Returns
        -------
        out: logits for each label
        """
        if bypass_coarse:
            return self.classifier(
                self.encode_fine(batch_data['X']), batch_data['Y_s'])
        else:
            return self.classifier(
                self.encode(batch_data['X'], batch_data['X_ind']),
                batch_data['Y_s'])

    def _construct_classifier(self):
        offset = 0
        if self.label_padding_index:
            offset += 1
        # last one is padding index
        if self.metric == 'cosine':
            return linear_layer.UNSparseLinear(
                input_size=self.representation_dims,
                output_size=self.num_labels + offset,
                padding_idx=self.label_padding_index)
        else:
            return linear_layer.SparseLinear(
                input_size=self.representation_dims,
                output_size=self.num_labels + offset,
                padding_idx=self.label_padding_index,
                bias=True)

    def to(self):
        """Send layers to respective devices
        """
        self.transform_fine.to()
        super().to()

    def initialize_classifier(self, weight, bias=None):
        """Initialize classifier from existing weights

        Arguments:
        -----------
        weight: numpy.ndarray
        bias: numpy.ndarray or None, optional (default=None)
        """
        self.classifier.weight.data.copy_(torch.from_numpy(weight))
        if bias is not None:
            self.classifier.bias.data.copy_(
                torch.from_numpy(bias).view(-1, 1))

    def get_clf_weights(self):
        """Get classifier weights
        """
        return self.classifier.get_weights()

    def __repr__(self):
        s = f"{self.transform}\n"
        s += f"(Transform fine): {self.transform_fine}"
        s += f"\n(Classifier): {self.classifier}\n"
        return s


class SiameseXML(DeepXMLBase):
    """
    Siamese to embed document and labels together
    * Allows different or same embeddings for documents and labels
    * Allows different or same transformation for documents and labels
    """
    def __init__(self, params, device="cuda:0"):
        super(SiameseXML, self).__init__(None)
        self.share_weights = params.share_weights
        self.metric = params.metric
        transform_config_dict = transform_layer.fetch_json(
            params.arch, params)
        self.representation_dims = int(
            transform_config_dict['representation_dims'])
        ts_coarse_document = transform_config_dict['transform_coarse_doc']
        ts_fine_document = transform_config_dict['transform_fine_doc']
        transform_config_dict = transform_layer.fetch_json(
            params.arch, params)
        ts_coarse_label = transform_config_dict['transform_coarse_lbl']
        ts_fine_label = transform_config_dict['transform_fine_lbl']

        #  Network to embed document
        self.document_net = self._construct_transform(ts_coarse_document)
        self.label_net = self._construct_transform(ts_coarse_label)

        self.transform_fine_document = self._construct_transform(
            ts_fine_document)
        self.transform_fine_label = self._construct_transform(
            ts_fine_label)
        if self.share_weights:
            self._create_shared_net()

    def _create_shared_net(self):
        self.label_net = self.document_net
        
    def encode(self, x, x_ind=None, return_coarse=False):
        return self.encode_document(x, x_ind, return_coarse)

    def encode_document(self, x, x_ind=None, return_coarse=False):
        """Forward pass
        * Assumes features are dense if x_ind is None

        Arguments:
        -----------
        x: torch.FloatTensor
            (sparse features) contains weights of features as per x_ind or
            (dense features) contains the dense representation of a point
        x_ind: torch.LongTensor or None, optional (default=None)
            contains indices of features (sparse features)
        return_coarse: boolean, optional (default=False)
            Return coarse features or not

        Returns
        -------
        out: logits for each label
        """
        encoding = self.document_net.encode(
            _to_device((x, x_ind), self.device))
        if not return_coarse:
            encoding = self.transform_fine_document(encoding)
        return encoding

    def encode_label(self, x, x_ind=None, return_coarse=False):
        """Forward pass
        * Assumes features are dense if x_ind is None

        Arguments:
        -----------
        x: torch.FloatTensor
            (sparse features) contains weights of features as per x_ind or
            (dense features) contains the dense representation of a point
        x_ind: torch.LongTensor or None, optional (default=None)
            contains indices of features (sparse features)
        return_coarse: boolean, optional (default=False)
            Return coarse features or not

        Returns
        -------
        out: logits for each label
        """
        encoding = self.label_net.encode(
            _to_device((x, x_ind), self.device))
        if not return_coarse:
            encoding = self.transform_fine_label(encoding)
        return encoding

    def similarity(self, doc_rep, lbl_rep):
        #  Units vectors in case of cosine similarity
        if self.metric == 'cosine':
            doc_rep = F.normalize(doc_rep, dim=1)
            lbl_rep = F.normalize(lbl_rep, dim=1)
        return doc_rep @ lbl_rep.T

    def forward(self, batch_data, *args, **kwargs):
        """Forward pass
        * For batch sampling, i.e., labels are shared across the batch
        * TODO: if each document had it's shortlist?

        Arguments:
        -----------
        batch_data: dict
            * 'X': torch.FloatTensor
                feature weights for given indices or dense rep. (document)
            * 'X_ind': torch.LongTensor
                feature indices (LongTensor) or None (document)
            * 'YX': torch.FloatTensor
                feature weights for given indices or dense rep. (label)
            * 'YX_ind': torch.LongTensor (label)
                feature indices (LongTensor) or None

        Returns
        -------
        out: logits for each label in the shortlist
        """
        doc_rep = self.encode_document(batch_data['X'], batch_data['X_ind'])
        lbl_rep = self.encode_label(batch_data['YX'], batch_data['YX_ind'])
        cnt_rep = self.encode_document(batch_data['CX'], batch_data['CX_ind'])
        # compute the centroid with 2 positive documents
        cnt_rep = F.normalize(cnt_rep + lbl_rep)
        return self.similarity(doc_rep, lbl_rep), self.similarity(cnt_rep, lbl_rep)

    def to(self):
        """Send layers to respective devices
        """
        self.transform_fine_document.to()
        self.transform_fine_label.to()
        self.document_net.to()
        self.label_net.to()

    def initialize(self, x):
        """Initialize embeddings from existing ones
        Parameters:
        -----------
        word_embeddings: numpy array
            existing embeddings
        """
        self.document_net.initialize(x)
        if not self.share_weights:
            self.label_net.initialize(x)

    def save_intermediate_model(self, fname):
        out = {}
        if not self.share_weights:
            out['label_net'] = self.label_net.state_dict()
            out['document_net'] = self.document_net.state_dict()
        else:
            out = {'net': self.document_net.state_dict()}
        torch.save(out, fname)

    def load_intermediate_model(self, fname):
        out = pickle.load(open(fname, 'rb'))
        if not self.share_weights:
            self.document_net.load_state_dict(out['document_net'])
            self.label_net.load_state_dict(out['label_net'])
        else:
            self.document_net.load_state_dict(out['net'])

    def named_parameters(self, recurse=True, return_shared=False):
        if self.share_weights and not return_shared:
            # Assuming label_net is a copy of document_net
            for name, param in super().named_parameters(recurse=recurse):
                if 'label_net' not in name:
                    yield name, param
        else:
            for name, param in super().named_parameters(recurse=recurse):
                yield name, param

    def parameters(self, recurse=True, return_shared=False):
        if self.share_weights and not return_shared:
            # Assuming label_net is a copy of document_net
            for name, param in super().named_parameters(recurse=recurse):
                if 'label_net' not in name:
                    yield param
        else:
            for name, param in self.named_parameters(recurse=recurse):
                yield param

    @property
    def modules_(self, return_shared=False):
        out = OrderedDict()
        for k, v in self._modules.items():
            if not return_shared and self.share_weights and 'label_net' in k:
                continue
            out[k] = v
        return out

    def __repr__(self):
        s = f"{self.__class__.__name__} (Weights shared: {self.share_weights})"
        s += f"\n(DocNet): {self.document_net}\n"
        s += f"(Transform fine Doc): {self.transform_fine_document} \n"
        s += f"(LabelNet): {self.label_net}\n"
        s += f"(Transform fine Label): {self.transform_fine_label} \n"
        return s
