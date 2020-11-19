import torch
import torch.nn as nn
import torch.nn.functional as F


__author__ = 'KD'


class SCLinear(nn.Module):
    """
    A Linear layer with constrained spectral norm

    Arguments:
    ----------
    input_size: int
        input dimension
    output_size: int
        output dimension
    fix_radius: boolean, optional, default=True
        Learn the spectral radius as a parameter if False
        spectral radius is fixed at given value, if True
    init_radius: float, optional, default=1.0
        Initial value of spectral radius
    init_params: str, optional, default='eye'
        initialization strategy (best results with eye)
    """
    def __init__(self, input_size, output_size, fix_radius=False,
                 init_radius=1.0, init_params='eye'):
        super(SCLinear, self).__init__()
        self.hidden = nn.utils.spectral_norm(
            nn.Linear(input_size, output_size, bias=True))
        self.radius = nn.Parameter(torch.Tensor(1, 1))
        if fix_radius:
            self.radius.requires_grad = False
        self.initialize(init_params, init_radius)

    def forward(self, x):
        return self.radius*self.hidden(x)

    def initialize(self, init_params, init_radius):
        """Initialize units

        Arguments:
        -----------
        init_params: str
            Initialize hidden layer with 'random' or 'eye'
        init_radius: float
            Initial value of spectral radius
        """
        nn.init.constant_(self.radius, init_radius)
        if init_params == 'random':
            nn.init.xavier_uniform_(
                self.hidden.weight,
                gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(self.hidden.bias, 0.0)
        else:
            nn.init.eye_(self.hidden.weight)
            nn.init.constant_(self.hidden.bias, 0.0)


class Residual(nn.Module):
    """Implementation of a Residual block
    
    Arguments:
    ----------
    input_size: int
        input dimension
    output_size: int
        output dimension
    dropout: float
        dropout probability
    fix_radius: boolean, optional, default=True
        Learn the spectral radius as a parameter if False
        spectral radius is fixed at given value, if True
    init_radius: float, optional, default=1.0
        Initial value of spectral radius
    init_params: str, optional, default='eye'
        initialization strategy (best results with eye)
    """

    def __init__(self, input_size, output_size, dropout, fix_radius=False,
                 init_radius=1.0, init_params='eye'):
        super(Residual, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.padding_size = self.output_size - self.input_size
        self.hidden_layer = nn.Sequential(
            SCLinear(input_size, output_size, fix_radius,
                    init_radius, init_params),
            nn.ReLU(),
            nn.Dropout(self.dropout)
            )

    def forward(self, x):
        """Forward pass for Residual
        
        Arguments:
        ----------
        x: torch.Tensor
            dense document embedding

        Returns
        -------
        out: torch.Tensor
            dense document embeddings transformed via residual block
        """
        temp = F.pad(x, (0, self.padding_size), 'constant', 0)
        return self.hidden_layer(x) + temp
