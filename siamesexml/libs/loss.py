import torch
import torch.nn.functional as F
import math


class _Loss(torch.nn.Module):
    def __init__(self, reduction='mean', pad_ind=None):
        super(_Loss, self).__init__()
        self.reduction = reduction
        self.pad_ind = pad_ind

    def _reduce(self, loss):
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'custom':
            return loss.sum(dim=1).mean()
        else:
            return loss.sum()

    def _mask_at_pad(self, loss):
        """
        Mask the loss at padding index, i.e., make it zero
        """
        if self.pad_ind is not None:
            loss[:, self.pad_ind] = 0.0
        return loss

    def _mask(self, loss, mask=None):
        """
        Mask the loss at padding index, i.e., make it zero
        * Mask should be a boolean array with 1 where loss needs
        to be considered.
        * it'll make it zero where value is 0
        """
        if mask is not None:
            loss = loss.masked_fill(~mask, 0.0)
        return loss


def _convert_labels_for_svm(y):
    """
        Convert labels from {0, 1} to {-1, 1}
    """
    return 2.*y - 1.0


class HingeLoss(_Loss):
    r""" Hinge loss
    * it'll automatically convert target to +1/-1 as required by hinge loss

    Arguments:
    ----------
    margin: float, optional (default=1.0)
        the margin in hinge loss
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    pad_ind: int/int64 or None (default=None)
        ignore loss values at this index
        useful when some index has to be used as padding index
    """

    def __init__(self, margin=1.0, reduction='mean', pad_ind=None):
        super(HingeLoss, self).__init__(reduction, pad_ind)
        self.margin = margin

    def forward(self, input, target, mask=None):
        """
        Arguments:
        ---------
        input: torch.FloatTensor
            real number pred matrix of size: batch_size x output_size
            typically logits from the neural network
        target:  torch.FloatTensor
            0/1 ground truth matrix of size: batch_size x output_size
            * it'll automatically convert to +1/-1 as required by hinge loss
        mask: torch.BoolTensor or None, optional (default=None)
            ignore entries [won't contribute to loss] where mask value is zero

        Returns:
        -------
        loss: torch.FloatTensor
            dimension is defined based on reduction
        """
        loss = F.relu(self.margin - _convert_labels_for_svm(target)*input)
        loss = self._mask_at_pad(loss)
        loss = self._mask(loss, mask)
        return self._reduce(loss)


class SquaredHingeLoss(_Loss):
    r""" Squared Hinge loss
    * it'll automatically convert target to +1/-1 as required by hinge loss

    Arguments:
    ----------
    margin: float, optional (default=1.0)
        the margin in squared hinge loss
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    pad_ind: int/int64 or None (default=None)
        ignore loss values at this index
        useful when some index has to be used as padding index
    """

    def __init__(self, margin=1.0, size_average=None, reduction='mean'):
        super(SquaredHingeLoss, self).__init__(size_average, reduction)
        self.margin = margin

    def forward(self, input, target, mask=None):
        """
        Arguments:
        ---------
        input: torch.FloatTensor
            real number pred matrix of size: batch_size x output_size
            typically logits from the neural network
        target:  torch.FloatTensor
            0/1 ground truth matrix of size: batch_size x output_size
            * it'll automatically convert to +1/-1 as required by hinge loss
        mask: torch.BoolTensor or None, optional (default=None)
            ignore entries [won't contribute to loss] where mask value is zero

        Returns:
        -------
        loss: torch.FloatTensor
            dimension is defined based on reduction
        """
        loss = F.relu(self.margin - _convert_labels_for_svm(target)*input)
        loss = loss**2
        loss = self._mask_at_pad(loss)
        loss = self._mask(loss, mask)
        return self._reduce(loss)


class BCEWithLogitsLoss(_Loss):
    r""" BCE loss (expects logits; numercial stable)
    This loss combines a `Sigmoid` layer and the `BCELoss` in one single
    class. This version is more numerically stable than using a plain `Sigmoid`
    followed by a `BCELoss` as, by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.

    Arguments:
    ----------
    weight: torch.Tensor or None, optional (default=None))
        a manual rescaling weight given to the loss of each batch element.
        If given, has to be a Tensor of size batch_size
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    pos_weight: torch.Tensor or None, optional (default=None)
        a weight of positive examples.
        it must be a vector with length equal to the number of classes.
    pad_ind: int/int64 or None (default=None)
        ignore loss values at this index
        useful when some index has to be used as padding index
    """
    __constants__ = ['weight', 'pos_weight', 'reduction']

    def __init__(self, weight=None, reduction='mean',
                 pos_weight=None, pad_ind=None):
        super(BCEWithLogitsLoss, self).__init__(reduction, pad_ind)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input, target, mask=None):
        """
        Arguments:
        ---------
        input: torch.FloatTensor
            real number pred matrix of size: batch_size x output_size
            typically logits from the neural network
        target:  torch.FloatTensor
            0/1 ground truth matrix of size: batch_size x output_size
        mask: torch.BoolTensor or None, optional (default=None)
            ignore entries [won't contribute to loss] where mask value is zero

        Returns:
        -------
        loss: torch.FloatTensor
            dimension is defined based on reduction
        """
        loss = F.binary_cross_entropy_with_logits(input, target,
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction='none')
        loss = self._mask_at_pad(loss)
        loss = self._mask(loss, mask)
        return self._reduce(loss)


class HingeContrastiveLoss(_Loss):
    r""" Hinge contrastive loss (expects cosine similarity)

    Arguments:
    ----------
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    margin: float, optional (default=0.8)
        margin in Hinge contrastive loss
    pos_weight: float, optional (default=1.0)
        weight of loss with positive target
    """

    def __init__(self, reduction='mean', margin=0.8, pos_weight=1.0):
        super(HingeContrastiveLoss, self).__init__(reduction=reduction)
        self.margin = margin
        self.pos_weight = pos_weight

    def forward(self, input, target, mask=None):
        """
        Arguments:
        ---------
        input: torch.FloatTensor
            real number pred matrix of size: batch_size x output_size
            cosine similarity b/w label and document
        target:  torch.FloatTensor
            0/1 ground truth matrix of size: batch_size x output_size
        mask: torch.BoolTensor or None, optional (default=None)
            ignore entries [won't contribute to loss] where mask value is zero

        Returns:
        -------
        loss: torch.FloatTensor
            dimension is defined based on reduction
        """
        loss = torch.where(target > 0, (1-input) * self.pos_weight,
                           torch.max(
                               torch.zeros_like(input), input - self.margin))
        loss = self._mask_at_pad(loss)
        loss = self._mask(loss, mask)
        return self._reduce(loss)


class TripletMarginLossOHNM(_Loss):
    r""" Triplet Margin Loss with Online Hard Negative Mining

    * Applies loss using the hardest negative in the mini-batch
    * Assumes diagonal entries are ground truth (for multi-class as of now)

    Arguments:
    ----------
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    margin: float, optional (default=0.8)
        margin in triplet margin loss
    k: int, optional (default=2)
        compute loss only for top-k negatives in each row 
    apply_softmax: boolean, optional (default=2)
        promotes hard negatives using softmax
    """

    def __init__(self, reduction='mean', margin=0.8, k=3, apply_softmax=False):
        super(TripletMarginLossOHNM, self).__init__(reduction=reduction)
        self.margin = margin
        self.k = k
        self.apply_softmax = apply_softmax

    def forward(self, input, target, mask):
        """
        Arguments:
        ---------
        input: torch.FloatTensor
            real number pred matrix of size: batch_size x output_size
            cosine similarity b/w label and document
        target:  torch.FloatTensor
            0/1 ground truth matrix of size: batch_size x output_size
        mask: torch.BoolTensor or None, optional (default=None)
            ignore entries [won't contribute to loss] where mask value is False

        Returns:
        -------
        loss: torch.FloatTensor
            dimension is defined based on reduction
        """
        sim_p = torch.diagonal(input).view(-1, 1)
        similarities = torch.min(input, 1-target)
        _, indices = torch.topk(similarities, largest=True, dim=1, k=self.k)
        sim_n = input.gather(1, indices)
        loss = torch.max(torch.zeros_like(sim_p), sim_n - sim_p + self.margin)
        mask = torch.where(loss != 0, torch.ones_like(loss), torch.zeros_like(loss))
        if self.apply_softmax:
            prob = torch.softmax(sim_n * mask, dim=1)
            loss = loss * prob
        loss = self._mask_at_pad(loss)
        loss = self._mask(loss, mask)
        return self._reduce(loss)


class ProbContrastiveLoss(_Loss):
    r""" A probabilistic contrastive loss 
    * expects cosine similarity
    * or <w, x> b/w normalized vectors

    Arguments:
    ----------
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    pos_weight: float, optional (default=1.0)
        weight of loss with positive target
    c: float, optional (default=0.75)
        c in PC loss (0, 1)
    d: float, optional (default=3.0)
        d in PC loss >= 1
    threshold: float, optional (default=0.05)
        clip loss values less than threshold for negatives 
    """

    def __init__(self, reduction='mean', pos_weight=1.0,
                 c=0.75, d=3.0, threshold=0.05):
        super(ProbContrastiveLoss, self).__init__(reduction=reduction)
        self.pos_weight = pos_weight
        self.d = d
        self.c = math.log(c)
        self.scale = 1/d
        self.threshold = threshold
        self.constant = c/math.exp(d)

    def forward(self, input, target, mask=None):
        """
        Arguments:
        ---------
        input: torch.FloatTensor
            real number pred matrix of size: batch_size x output_size
            cosine similarity b/w label and document
        target:  torch.FloatTensor
            0/1 ground truth matrix of size: batch_size x output_size
        mask: torch.BoolTensor or None, optional (default=None)
            ignore entries [won't contribute to loss] where mask value is False

        Returns:
        -------
        loss: torch.FloatTensor
            dimension is defined based on reduction
        """
        # top_k negative entries
        loss = torch.where(
            target > 0, -self.c + (1-input)*self.d*self.pos_weight,
            -torch.log(1 - torch.exp(self.d*input)*self.constant))
        loss = loss * self.scale
        loss[torch.logical_and(target == 0, loss < self.threshold)] = 0.0
        loss = self._mask_at_pad(loss)
        loss = self._mask(loss, mask)
        return self._reduce(loss)


class kProbContrastiveLoss(_Loss):
    r""" A probabilistic contrastive loss 
    *expects cosine similarity
    * or <w, x> b/w normalized vectors

    Arguments:
    ----------
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    pos_weight: float, optional (default=1.0)
        weight of loss with positive target
    c: float, optional (default=0.75)
        c in PC loss (0, 1)
    d: float, optional (default=3.0)
        d in PC loss >= 1
    k: int, optional (default=2)
        compute loss only for top-k negatives in each row 
    apply_softmax: boolean, optional (default=2)
        promotes hard negatives using softmax
    """

    def __init__(self, reduction='mean', pos_weight=1.0, c=0.9,
                 d=1.5, k=2, apply_softmax=False):
        super(kProbContrastiveLoss, self).__init__(reduction=reduction)
        self.pos_weight = pos_weight
        self.d = d
        self.k = k
        self.c = math.log(c)
        self.scale = 1/d
        self.constant = c/math.exp(d)
        self.apply_softmax = apply_softmax

    def forward(self, input, target, mask=None):
        """
        Arguments:
        ---------
        input: torch.FloatTensor
            real number pred matrix of size: batch_size x output_size
            cosine similarity b/w label and document
        target:  torch.FloatTensor
            0/1 ground truth matrix of size: batch_size x output_size
        mask: torch.BoolTensor or None, optional (default=None)
            ignore entries [won't contribute to loss] where mask value is False

        Returns:
        -------
        loss: torch.FloatTensor
            dimension is defined based on reduction
        """
        loss = torch.where(
            target > 0, -self.c + (1-input)*self.d*self.pos_weight,
            -torch.log(1 - torch.exp(self.d*input)*self.constant))
        neg_vals, neg_ind = torch.topk(loss-target*3, k=self.k)
        loss_neg = torch.zeros_like(target)
        if self.apply_softmax:
            neg_probs = torch.softmax(neg_vals, dim=1)
            loss_neg = loss_neg.scatter(1, neg_ind, neg_probs*neg_vals)
        else:
          loss_neg = loss_neg.scatter(1, neg_ind, neg_vals)
        loss = torch.where(
            target > 0, loss, loss_neg)        
        loss = self._mask(loss, mask)
        return self._reduce(loss)
