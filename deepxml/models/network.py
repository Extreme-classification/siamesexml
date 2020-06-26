import torch
import torch.nn as nn
import numpy as np
import math
import models.custom_embeddings as custom_embeddings
import models.transform_layer as transform_layer
import models.linear_layer as linear_layer


__author__ = 'KD'


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

    def __init__(self, vocabulary_dims, embedding_dims,
                 trans_config, padding_idx=0):
        super(DeepXMLBase, self).__init__()
        self.vocabulary_dims = vocabulary_dims+1
        self.embedding_dims = embedding_dims
        self.trans_config = trans_config
        self.padding_idx = padding_idx
        self.embeddings = self._construct_embedding()
        self.transform = self._construct_transform(trans_config)
        self.classifier = self._construct_classifier()
        # Keep embeddings on first device
        self.device_embeddings = torch.device("cuda:0")

    def _construct_embedding(self):
        return custom_embeddings.CustomEmbedding(
            num_embeddings=self.vocabulary_dims,
            embedding_dim=self.embedding_dims,
            padding_idx=self.padding_idx,
            scale_grad_by_freq=False,
            sparse=True)

    def _construct_classifier(self):
        return nn.Identity()

    def _construct_transform(self, trans_config):
        return transform_layer.Transform(
            transform_layer.get_functions(trans_config))

    @property
    def representation_dims(self):
        return self.transform.representation_dims

    def encode(self, x, x_ind=None):
        """Forward pass
        * Assumes features are dense if x_ind is None

        Arguments:
        -----------
        x: torch.FloatTensor
            (sparse features) contains weights of features as per x_ind or
            (dense features) contains the dense representation of a point
        x_ind: torch.LongTensor or None, optional (default=None)
            contains indices of features (sparse features)

        Returns
        -------
        out: logits for each label
        """
        if x_ind is None:
            embed = x.to(self.device_embeddings)
        else:
            embed = self.embeddings(
                x_ind.to(self.device_embeddings),
                x.to(self.device_embeddings))
        return self.transform(embed)

    def forward(self, batch_data):
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

    def initialize_embeddings(self, word_embeddings):
        """Initialize embeddings from existing ones
        Parameters:
        -----------
        word_embeddings: numpy array
            existing embeddings
        """
        self.embeddings.from_pretrained(word_embeddings)

    def to(self):
        """Send layers to respective devices
        """
        self.embeddings.to()
        self.transform.to()
        self.classifier.to()

    def purge(self, fname):
        if os.path.isfile(fname):
            os.remove(fname)

    @property
    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def model_size(self):  # Assumptions: 32bit floats
        return self.num_trainable_params * 4 / math.pow(2, 20)


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
            params.trans_method, params)
        trans_config_coarse = transform_config_dict['transform_coarse']
        super(DeepXMLf, self).__init__(
            vocabulary_dims=params.vocabulary_dims,
            embedding_dims=params.embedding_dims,
            trans_config=trans_config_coarse,
            padding_idx=params.padding_idx)
        trans_config_fine = transform_config_dict['transform_fine']
        self.transform_fine = self._construct_transform(
            trans_config_fine)

    def encode(self, x, x_ind=None, return_coarse=False):
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
        encoding = super().encode(x, x_ind)
        return encoding if return_coarse else self.transform_fine(encoding)

    def _construct_classifier(self):
        return linear_layer.Linear(
            input_size=self.representation_dims,
            output_size=self.num_labels,  # last one is padding index
            bias=True
        )

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
        return "x"
        # return "{}\n{}\n{}\n(Transform fine): {}\n(Classifier): {}\n".format(str(self.embeddings), self.transform, self.transform_fine, self.classifier)


class DeepXMLs(DeepXMLBase):
    """DeepXMLt: DeepXML architecture to be trained with
                 a label shortlist
    * Allows additional transform layer for features
    """

    def __init__(self, params):
        transform_config_dict = transform_layer.fetch_json(
            params.trans_method, params)
        trans_config_coarse = transform_config_dict['transform_coarse']
        self.num_labels = params.num_labels
        self.label_padding_index = params.label_padding_index
        super(DeepXMLs, self).__init__(
            vocabulary_dims=params.vocabulary_dims,
            embedding_dims=params.embedding_dims,
            trans_config=trans_config_coarse,
            padding_idx=params.padding_idx)
        trans_config_fine = transform_config_dict['transform_fine']
        self.transform_fine = self._construct_transform(
            trans_config_fine)

    def encode(self, x, x_ind=None, return_coarse=False):
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
        encoding = super().encode(x, x_ind)
        return encoding if return_coarse else self.transform_fine(encoding)

    def forward(self, batch_data):
        """Forward pass

        Arguments:
        -----------
        batch_data: dict
            * 'X': torch.FloatTensor
                feature weights for given indices or dense rep.
            * 'X_ind': torch.LongTensor
                feature indices (LongTensor) or None
            * 'Y_s': torch.LongTensor
                indices of labels to pick for each document

        Returns
        -------
        out: logits for each label in the shortlist
        """
        return self.classifier(
            self.encode(batch_data['X'], batch_data['X_ind']),
            batch_data['Y_s'])

    def _construct_classifier(self):
        return linear_layer.SparseLinear(
            input_size=self.representation_dims,
            output_size=self.num_labels,
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
        return "{}\n{}\n{}\n(Transform fine): {}\n(Classifier): {}\n".format(
            self.embeddings, self.transform,
            self.transform_fine, self.classifier)


class DeepXMLpp(nn.Module):
    """
    DeepXML++ to embed document and labels together
    * Allows different or same embeddings for documents and labels
    * Allows different or same transformation for documents and labels
    """
    def __init__(self, params):
        self.tie_embeddings = params.tie_embeddings
        self.metric = params.metric
        transform_config_dict = transform_layer.fetch_json(
            params.trans_method, params)
        ts_coarse_document = transform_config_dict['ts_coarse_doc']
        ts_fine_document = transform_config_dict['ts_fine_doc']
        ts_coarse_label = transform_config_dict['ts_coarse_lbl']
        ts_fine_label = transform_config_dict['ts_label_lbl']

        #  Network to embed document
        self.document_net = DeepXMLBase(
            vocabulary_dims=params.vocabulary_dims_doc,
            embedding_dims=params.embedding_dims,
            trans_config=ts_coarse_document,
            padding_idx=params.padding_idx)
        #  Network to embed labels
        self.label_net = DeepXMLBase(
            vocabulary_dims=params.vocabulary_dims_lbl,
            embedding_dims=params.embedding_dims,
            trans_config=ts_coarse_document_lbl,
            padding_idx=params.padding_idx)
        self.transform_fine_document = self._construct_transform(
            ts_config_fine_doc)
        self.transform_fine_lbl = self._construct_transform(
            ts_config_fine_lbl)

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
        encoding = self.document_net.encode(x, x_ind)
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
        encoding = self.label_net.encode(x, x_ind)
        if not return_coarse:
            encoding = self.transform_fine_label(encoding)
        return encoding

    def similarity(self, doc_rep, lbl_rep):
        #  Units vectors in case of cosine similarity
        if self.metric == 'cosine':
            doc_rep = normalize(doc_rep, dim=1)
            lbl_rep = normalize(lbl_rep, dim=1)
        return doc_rep @ lbl_rep.T

    def forward(self, batch_data):
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
        lbl_rep = self.encode_document(batch_data['YX'], batch_data['YX_ind'])
        return self.similarity(doc_rep, lbl_rep)

    def to(self):
        """Send layers to respective devices
        """
        self.transform_fine_document.to()
        self.transform_fine_label.to()
        self.document_net.to()
        self.label_net.to()
