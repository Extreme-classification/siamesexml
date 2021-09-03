import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

__author__ = 'KD'


class Linear(nn.Module):
    """Linear layer
    Parameters:
    -----------
    input_size: int
        input size of transformation
    output_size: int
        output size of transformation
    bias: boolean, default=True
        whether to use bias or not
    device: str, default="cuda:0"
        keep on this device
    """

    def __init__(self, input_size, output_size,
                 bias=True, device="cuda:0"):
        super(Linear, self).__init__()
        self.device = device  # Useful in case of multiple GPUs
        self.input_size = input_size
        self.output_size = output_size
        self.weight = Parameter(torch.Tensor(self.output_size, self.input_size))
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        if self.bias is not None:
            return F.linear(input.to(self.device), self.weight, self.bias.view(-1))
        else:
            return F.linear(input.to(self.device), self.weight)

    def to(self):
        """Transfer to device
        """
        super().to(self.device)

    def reset_parameters(self):
        """Initialize vectors
        """
        torch.nn.init.xavier_uniform_(
            self.weight.data,
            gain=torch.nn.init.calculate_gain('relu'))
        if self.bias is not None:
            self.bias.data.fill_(0)

    def get_weights(self):
        """Get weights as numpy array
        Bias is appended in the end
        """
        _wts = self.weight.detach().cpu().numpy()
        if self.bias is not None:
            _bias = self.bias.detach().cpu().numpy()
            _wts = np.hstack([_wts, _bias])
        return _wts

    def __repr__(self):
        s = '{name}({input_size}, {output_size}, {device}'
        if self.bias is not None:
            s += ', bias=True'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    @property
    def sparse(self):
        return False


class SparseLinear(Linear):
    """Sparse Linear linear with sparse gradients
    Parameters:
    -----------
    input_size: int
        input size of transformation
    output_size: int
        output size of transformation
    padding_idx: int
        index for dummy label; embedding is not updated
    bias: boolean, default=True
        whether to use bias or not
    device: str, default="cuda:0"
        keep on this device
    """

    def __init__(self, input_size, output_size, padding_idx=None,
                 bias=True, device="cuda:0"):
        self.padding_idx = padding_idx
        super(SparseLinear, self).__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            device=device)

    def forward(self, embed, shortlist):
        """Forward pass for Linear sparse layer
        Parameters:
        ----------
        embed: torch.Tensor
            input to the layer
        shortlist: torch.LongTensor
            evaluate these labels only
        Returns
        -------
        out: torch.Tensor
            logits for each label in provided shortlist
        """
        embed = embed.to(self.device)
        shortlist = shortlist.to(self.device)
        short_weights = F.embedding(shortlist,
                                    self.weight,
                                    sparse=self.sparse,
                                    padding_idx=self.padding_idx)
        out = torch.matmul(embed.unsqueeze(1), short_weights.permute(0, 2, 1))
        if self.bias is not None:
            short_bias = F.embedding(shortlist,
                                     self.bias,
                                     sparse=self.sparse,
                                     padding_idx=self.padding_idx)
            out = out + short_bias.permute(0, 2, 1)
        return out.squeeze()

    def reset_parameters(self):
        """Initialize weights vectors
        """
        super().reset_parameters()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def __repr__(self):
        s = '{name}({input_size}, {output_size}, {device}'
        if self.bias is not None:
            s += ', bias=True'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        s += ', sparse=True)'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def get_weights(self):
        """Get weights as numpy array
        Bias is appended in the end
        """
        _wts = self.weight.detach().cpu().numpy()
        if self.padding_idx is not None:
            _wts = _wts[:-1, :]
        if (self.bias is not None):
            _bias = self.bias.detach().cpu().numpy()
            if self.padding_idx is not None:
                _bias = _bias[:-1, :]
            _wts = np.hstack([_wts, _bias])
        return _wts

    @property
    def sparse(self):
        return True


class UNSparseLinear(SparseLinear):
    """Sparse Linear linear with sparse gradients
    * will normalize document and label representations to unit l2 norm

    Parameters:
    -----------
    input_size: int
        input size of transformation
    output_size: int
        output size of transformation
    padding_idx: int
        index for dummy label; embedding is not updated
    bias: boolean, default=True
        whether to use bias or not
    device: str, default="cuda:0"
        keep on this device
    """

    def __init__(self, input_size, output_size,
                 padding_idx=None, device="cuda:0"):
        super(UNSparseLinear, self).__init__(
            input_size=input_size,
            output_size=output_size,
            padding_idx=padding_idx,
            bias=False,
            device=device)

    def forward(self, embed, shortlist):
        """Forward pass for Linear sparse layer
        Parameters:
        ----------
        embed: torch.Tensor
            input to the layer
        shortlist: torch.LongTensor
            evaluate these labels only

        Returns
        -------
        out: torch.Tensor
            logits for each label in provided shortlist
        """
        embed = F.normalize(embed.to(self.device), dim=1)
        shortlist = shortlist.to(self.device)
        short_weights = F.embedding(shortlist,
                                    self.weight,
                                    sparse=self.sparse,
                                    padding_idx=self.padding_idx)
        short_weights = F.normalize(short_weights, dim=2)
        out = torch.matmul(embed.unsqueeze(1), short_weights.permute(0, 2, 1))
        return out.squeeze()
