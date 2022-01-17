import os
import numpy as np
import pickle
import torch
import torch.utils.data
import libs.utils as utils
import models.network as network
import libs.shortlist as shortlist
import libs.model as model_utils
import libs.optimizer as optimizer
import libs.sampling as sampling
import libs.loss as loss


__author__ = 'KD'


def set_seed(value):
    """
        Set seed value for numpy and pytorch
    """
    print("\nSetting the seed value: {}".format(value))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    np.random.seed(value)


def load_emeddings(params):
    """Load word embeddings from numpy file
    * Support for:
        - loading pre-trained embeddings
        - generating random embeddings
    * vocabulary_dims must match #rows in embeddings
    """
    try:
        fname = os.path.join(
            params.data_dir, params.dataset, params.embeddings)
        embeddings = np.load(fname)
    except FileNotFoundError:
        exit("Embedding File not found. Check path or set 'init' to null")
    if params.feature_indices is not None:
        indices = np.genfromtxt(params.feature_indices, dtype=np.int32)
        embeddings = embeddings[indices, :]
        del indices
    return embeddings


def train(model, params):
    """Train the model with given data

    Arguments
    ----------
    model: DeepXML
        train this model (typically DeepXML model)
    params: NameSpace
        parameter of the model
    """
    trn_pretrained_shortlist = None
    val_pretrained_shortlist = None

    # Names harcoded for now; changed if required
    # useful in case of re-ranker or where shortlist is already available
    if params.use_pretrained_shortlist:
        trn_pretrained_shortlist = os.path.join(
            params.model_dir, 'trn_shortlist.npz')
        if params.validate:
            val_pretrained_shortlist = os.path.join(
                params.model_dir, 'tst_shortlist.npz')
    train_time, model_size = model.fit(
        data_dir=params.data_dir,
        model_dir=params.model_dir,
        result_dir=params.result_dir,
        dataset=params.dataset,
        data={'X': None, 'Y': None, 'Yf': None},
        learning_rate=params.learning_rate,
        num_epochs=params.num_epochs,
        trn_feat_fname=params.trn_feat_fname,
        val_feat_fname=params.val_feat_fname,
        trn_label_fname=params.trn_label_fname,
        val_label_fname=params.val_label_fname,
        batch_size=params.batch_size,
        lbl_feat_fname=params.lbl_feat_fname,
        num_workers=params.num_workers,
        normalize_features=params.normalize,
        normalize_labels=params.nbn_rel,
        shuffle=params.shuffle,
        feature_type=params.feature_type,
        validate=params.validate,
        beta=params.beta,
        init_epoch=params.last_epoch,
        keep_invalid=params.keep_invalid,
        shortlist_method=params.shortlist_method,
        validate_after=params.validate_after,
        feature_indices=params.feature_indices,
        use_intermediate_for_shorty=params.use_intermediate_for_shorty,
        label_indices=params.label_indices,
        batch_type=params.batch_type,
        sampling_type=params.sampling_type,
        trn_pretrained_shortlist = None,
        val_pretrained_shortlist = None)
    model.save(params.model_dir, params.model_fname)
    return train_time, model_size


def get_document_embeddings(model, params, _save=True):
    """Get document embedding for given test file
    * save as a memmap file if huge_dataset is True

    Arguments
    ----------
    model: DeepXML
        train this model (typically DeepXML model)
    params: NameSpace
        parameter of the model
    _save: boolean, optional, default=True
        Save embeddings as well (fname=params.out_fname)
    """
    fname_temp = None
    if params.huge_dataset:
        fname_temp = os.path.join(
            params.result_dir, params.out_fname + ".memmap.npy")
    doc_embeddings = model.get_embeddings(
        data_dir=os.path.join(params.data_dir, params.dataset),
        fname=params.tst_feat_fname,
        data=None,
        fname_out=fname_temp,
        feature_type=params.feature_type,
        use_intermediate=params.use_intermediate_for_shorty,
        batch_size=params.batch_size,
        normalize=params.normalize,
        num_workers=params.num_workers,
        indices=params.feature_indices)
    fname = os.path.join(params.result_dir, params.out_fname)
    if _save:  # Save
        np.save(fname, doc_embeddings)
    if fname_temp is not None and os.path.exists(fname_temp):
        os.remove(fname_temp)
        del doc_embeddings
        doc_embeddings = None
    return doc_embeddings


def get_word_embeddings(model, params):
    """Extract word embeddings for the given model
    Arguments
    ----------
    model: DeepXML
        train this model (typically DeepXML model)
    params: NameSpace
        parameter of the model
    """
    _embeddings = model.net.get_token_embeddings()
    fname = os.path.join(params.result_dir, params.out_fname)
    np.save(fname, _embeddings)


def get_classifier_wts(model, params):
    """Get classifier weights and biases for given model
    * -inf bias for untrained classifiers i.e. labels without any data
    * default path: params.result_dir/export/classifier.npy
    * Bias is appended in the end
    Arguments
    ----------
    model: DeepXML
        train this model (typically DeepXML model)
    params: NameSpace
        parameter of the model
    """
    print("TODO: Classifier weights.")
    exit()


def inference(model, params):
    """Predict the top-k labels for given test data
    Arguments
    ----------
    model: DeepXML
        train this model (typically DeepXML model)
    params: NameSpace
        parameter of the model
    """
    pretrained_shortlist = None
    classifier_type = 'full'
    if params.use_shortlist:
        classifier_type = 'shortlist'
    if params.use_pretrained_shortlist:
        pretrained_shortlist = os.path.join(
            params.model_dir, 'tst_shortlist.npz')
    predicted_labels, prediction_time, avg_prediction_time = model.predict(
        data_dir=params.data_dir,
        dataset=params.dataset,
        tst_label_fname=params.tst_label_fname,
        tst_feat_fname=params.tst_feat_fname,
        lbl_feat_fname=params.lbl_feat_fname,
        normalize_features=params.normalize,
        normalize_labels=params.nbn_rel,
        beta=params.beta,
        result_dir=params.result_dir,
        num_workers=params.num_workers,
        top_k=params.top_k,
        data={'X': None, 'Y': None, 'Yf': None},
        keep_invalid=params.keep_invalid,
        feature_indices=params.feature_indices,
        label_indices=params.label_indices,
        use_intermediate_for_shorty=params.use_intermediate_for_shorty,
        shortlist_method=params.shortlist_method
    )
    # Real number of labels
    num_samples, num_labels = utils.get_header(
        os.path.join(params.data_dir, params.dataset, params.tst_label_fname))
    label_mapping = None
    if not params.keep_invalid:
        _split = None
        if params.label_indices is not None:
            _split = params.label_indices.split("_")[-1].split(".")[0]
        temp = "labels_params_split_{}.pkl".format(_split)
        if _split is None:
            temp = 'labels_params.pkl'
        fname = os.path.join(params.model_dir, temp)
        temp = pickle.load(open(fname, 'rb'))
        label_mapping = temp['valid_labels']
        num_labels = temp['num_labels']
    utils.save_predictions(
        predicted_labels, params.result_dir,
        label_mapping, num_samples, num_labels,
        prefix=params.pred_fname, get_fnames=params.get_only)
    return predicted_labels, prediction_time, avg_prediction_time


def construct_network(params):
    """Construct DeepXML network
    """
    if params.network_type == 'shortlist':
        return network.DeepXMLs(params)
    elif params.network_type == 'siamese':
        return network.SiameseXML(params)
    else:
        return network.DeepXMLf(params)


def construct_shortlist(params):
    """Construct shorty

    Arguments:
    ----------
    params: NameSpace
        parameters of the model with following sampling methods (ns_method)

        * negative sampling (ns)
          sample labels at random for each data point
        * kcentroid
          sample labels using an ANN graph learned over label centroids
        * ensemble
          sample labels using an ANN graph learned over:
          label centroids and training instances
    """
    if params.use_pretrained_shortlist:
        return None

    if not params.use_shortlist:
        return None

    if params.ns_method == 'random':  # Negative Sampling
        shorty = sampling.Sampler(
            num_labels=params.num_labels,
            num_samples=params.num_nbrs,
            prob=None,
            replace=True)
    elif params.ns_method == 'centroids':
        shorty = shortlist.ShortlistCentroids(
            method=params.ann_method,
            num_neighbours=params.num_nbrs,
            M=params.M,
            efC=params.efC,
            efS=params.efS,
            num_threads=params.ann_threads,
            num_clusters=params.num_centroids)
    elif params.ns_method == 'mips':
        shorty = shortlist.ShortlistMIPS(
            method=params.ann_method,
            num_neighbours=params.num_nbrs,
            M=params.M,
            efC=params.efC,
            efS=params.efS,
            num_threads=params.ann_threads)
    elif params.ns_method == 'ensemble':
        shorty = shortlist.ShortlistEnsemble(
            method=params.ann_method,
            num_neighbours={'ens': params.num_nbrs,
                'kcentroid': params.efS,
                'knn': params.efS,
                'kembed': params.efS},
            M={'kcentroid': params.M,
                'kembed': params.M,
                'knn': params.M//2},
            efC={'kcentroid': params.efC,
                'kembed': params.efC,
                'knn': params.efC//6},
            efS={'kcentroid': params.efS,
                'kembed': params.efS,
                'knn': params.efS//3},
            num_threads=params.ann_threads,
            verbose=True,
            use_knn=True,
            num_clusters=params.num_centroids, 
            gamma=0.025)
    else:
        raise NotImplementedError("Not yet implemented!")
    return shorty


def construct_model(params, net, criterion, opt, shorty):
    """Construct model object

    Arguments:
    ----------
    params: NameSpace
        parameters of the model

        * siamese: train the encoder using the Siamese pipeline
        * shortlist: train (or predict) using a label shortlist
          typically generated from a nearest neighbor graph or a
          pre-trained shortlist
        * full: with OVA classifier
    net: models.network.DeepXMLBase 
        * DeepXMLs: network with a label shortlist
        * DeepXMLf: network with fully-connected classifier
    criterion: libs.loss._Loss 
        to compute loss given y and y_hat
    opt: libs.optimizer.Optimizer
        to back-propagate and updating the parameters        
    shorty: libs.shortlist.Shortlist
        to generate a shortlist of labels        
    """
    if params.model_method == 'ns':  # Random negative Sampling
        model = model_utils.ModelNS(
            params, net, criterion, opt, shorty)
    elif params.model_method == 'shortlist':  # Approximate Nearest Neighbor
        model = model_utils.ModelShortlist(
            params, net, criterion, opt, shorty)
    elif params.model_method == 'full':
        model = model_utils.ModelFull(
            params, net, criterion, opt)
    elif params.model_method == 'siamese':
        model = model_utils.ModelSiamese(
            params, net, criterion, opt, shorty)
    else:
        raise NotImplementedError("Unknown model_method.")
    return model


def construct_loss(params, pos_weight=1.0):
    """
    Return the loss

    Arguments:
    ----------
    params: NameSpace
        parameters of the model
        * mean: mean over all entries/terms (used with OVA setting)
        * sum: sum over all entries/terms (used with a shortlist)
               - the loss is then divided by batch_size resulting in
                 sum over labels and mean over data-points in a batch
    pos_weight: int or None, optional, default=None
        weight the loss terms where y_nl = 1
    """
    _reduction = 'sum' if params.use_shortlist else 'mean'
    # pad index is for OVA training and not shortlist
    # pass mask for shortlist
    _pad_ind = None if params.use_shortlist else params.label_padding_index
    if params.loss == 'bce':
        return loss.BCEWithLogitsLoss(
            reduction=_reduction,
            pad_ind=_pad_ind,
            pos_weight=None)
    elif params.loss == 'triplet_margin_onm':
        return loss.TripletMarginLossOHNM(
            reduction=_reduction,
            margin=params.margin)
    elif params.loss == 'hinge_contrastive':
        return loss.HingeContrastiveLoss(
            reduction=_reduction,
            pos_weight=pos_weight,
            margin=params.margin)
    elif params.loss == 'prob_contrastive':
        return loss.ProbContrastiveLoss(
            reduction=_reduction,
            c=0.75,
            d=3.0,
            pos_weight=pos_weight,
            threshold=params.margin)
    elif params.loss == 'kprob_contrastive':
        return loss.kProbContrastiveLoss(
            k=params.k,
            reduction='custom',
            c=0.9,
            d=1.5,
            apply_softmax=False,
            pos_weight=pos_weight)


def main(params):
    """
        Main function
    """
    output = None
    set_seed(params.seed)
    if params.mode == 'train':
        # Use last index as padding label
        if params.network_type == 'shortlist':
            params.label_padding_index = params.num_labels
        net = construct_network(params)
        if params.init == 'intermediate':
            print("Loading intermediate representation.")
            net.load_intermediate_model(
                os.path.join(os.path.dirname(params.model_dir), "Z.pkl"))
        elif params.init == 'token_embeddings':
            print("Loading pre-trained embeddings.")
            embeddings = load_emeddings(params)
            net.initialize(embeddings)
            del embeddings
        else:  # trust the random init
            print("Random initialization.")
        criterion = construct_loss(params)
        print("Model parameters: ", params)
        print("\nModel configuration: ", net)
        opt = optimizer.Optimizer(
            opt_type=params.optim,
            learning_rate=params.learning_rate,
            momentum=params.momentum,
            weight_decay=params.weight_decay)
        opt.construct(net)
        shorty = construct_shortlist(params)
        model = construct_model(params, net, criterion, opt, shorty)
        model.transfer_to_devices()
        output = train(model, params)
        fname = os.path.join(params.result_dir, 'params.json')
        if params.save_intermediate:
            net.save_intermediate_model(
                os.path.join(os.path.dirname(params.model_dir), "Z.pkl"))
        utils.save_parameters(fname, params)

    elif params.mode == 'predict':
        fname = os.path.join(params.result_dir, 'params.json')
        utils.load_parameters(fname, params)
        net = construct_network(params)
        print("Model parameters: ", params)
        print("\nModel configuration: ", net)
        shorty = None
        shorty = construct_shortlist(params)
        model = construct_model(params, net, None, None, shorty)
        model.transfer_to_devices()
        model.load(params.model_dir, params.model_fname)
        output = inference(model, params)

    elif params.mode == 'extract':
        fname = os.path.join(params.result_dir, 'params.json')
        utils.load_parameters(fname, params)
        net = construct_network(params)
        print("Model parameters: ", params)
        print("\nModel configuration: ", net)
        shorty = construct_shortlist(params)
        model = construct_model(
            params, net, criterion=None, opt=None, shorty=shorty)
        model.load(params.model_dir, params.model_fname)
        model.transfer_to_devices()
        if params.tst_feat_fname == "0":
            get_word_embeddings(model, params)
            get_classifier_wts(model, params)
        else:
            get_document_embeddings(model, params)

    else:
        raise NotImplementedError("Unknown mode!")
    return output


if __name__ == '__main__':
    pass
