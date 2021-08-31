"""
This file is used to store the implementation of Correct And Smooth from dgl examples
"""

from dgl import DGLGraph
from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class LabelPropagation(nn.Module):
    r"""
    Description
    -----------
    Introduced in `Learning from Labeled and Unlabeled Data with Label Propagation
     <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.3864&rep=rep1&type=pdf>`_
    .. math::
        \mathbf{Y}^{\prime} = \alpha \cdot \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{Y} + (1 - \alpha) \mathbf{Y},
    where unlabeled data is inferred by labeled data via propagation.
    Parameters
    ----------
        num_layers: int
            The number of propagations.
        alpha: float
            The :math:`\alpha` coefficient.
        adj: str
            'DAD': D^-0.5 * A * D^-0.5
            'DA': D^-1 * A
            'AD': A * D^-1
    """

    def __init__(self, num_layers: int, alpha: float, adj: str = 'DA'):
        """
        Sets attributes of label propagation object
        Args:
            num_layers: number of propagation iterations
            alpha: weight of the original labels
            adj: matrix with normalized weights between nodes
        """
        super(LabelPropagation, self).__init__()

        self.num_layers = num_layers
        self.alpha = alpha
        self.adj = adj

    @torch.no_grad()
    def forward(self, g: DGLGraph, labels: torch.tensor, mask: Optional[torch.tensor] = None,
                post_step: Callable = lambda y: y.clamp_(0., 1.)) -> torch.tensor:
        """
        Executes label propagation

        Args:
            g: homogeneous undirected graph
            labels: ground truth associated to nodes in the graph
            mask: training idx
            post_step: function to apply after each step of propagation

        Returns:

        """

        with g.local_scope():

            # If labels are integers (meaning they are classes), we change them in
            # one-hot encoding
            if labels.dtype == torch.long:
                labels = F.one_hot(labels.view(-1)).to(torch.float32)
            y = labels

            # If only keep train data in mask as non zero values
            if mask is not None:
                y = torch.zeros_like(labels)
                y[mask] = labels[mask]

            # We save the part of original labels kept
            last = (1 - self.alpha) * y

            # We get the power of degree matrix needed for the propagation
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5 if self.adj == 'DAD' else -1).to(labels.device)

            # We proceed to propagation
            for _ in range(self.num_layers):

                # Assume the graphs to be undirected
                if self.adj in ['DAD', 'AD']:
                    y = norm * y

                g.ndata['h'] = y
                g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                y = self.alpha * g.ndata.pop('h')

                if self.adj in ['DAD', 'DA']:
                    y = y * norm

                y = post_step(last + y)

            return y


class CorrectAndSmooth(nn.Module):
    r"""
    Description
    -----------
    Introduced in `Combining Label Propagation and Simple Models Out-performs Graph Neural Networks <https://arxiv.org/abs/2010.13993>`_
    Parameters
    ----------
        num_correction_layers: int
            The number of correct propagations.
        correction_alpha: float
            The coefficient of correction.
        correction_adj: str
            'DAD': D^-0.5 * A * D^-0.5
            'DA': D^-1 * A
            'AD': A * D^-1
        num_smoothing_layers: int
            The number of smooth propagations.
        smoothing_alpha: float
            The coefficient of smoothing.
        smoothing_adj: str
            'DAD': D^-0.5 * A * D^-0.5
            'DA': D^-1 * A
            'AD': A * D^-1
        autoscale: bool, optional
            If set to True, will automatically determine the scaling factor :math:`\sigma`. Default is True.
        scale: float, optional
            The scaling factor :math:`\sigma`, in case :obj:`autoscale = False`. Default is 1.
    """

    def __init__(self,
                 num_correction_layers: int,
                 correction_alpha: float,
                 num_smoothing_layers: int,
                 smoothing_alpha: float,
                 correction_adj: str = 'DA',
                 smoothing_adj: str = 'DA',
                 autoscale=True,
                 scale=1.):
        super(CorrectAndSmooth, self).__init__()

        self.autoscale = autoscale
        self.scale = scale

        self.prop1 = LabelPropagation(num_correction_layers,
                                      correction_alpha,
                                      correction_adj)
        self.prop2 = LabelPropagation(num_smoothing_layers,
                                      smoothing_alpha,
                                      smoothing_adj)

    def correct(self, g: DGLGraph, y_soft: torch.tensor, y_true: torch.tensor,
                mask: torch.tensor) -> torch.tensor:
        """
        Propagates the errors in the homogeneous undirected graph using the following iterative solution:
        E^(t+1) = (1 - alpha)E + alpha*SE^(t)

        Args:
            g: dgl homogeneous undirected graph
            y_soft: tensor of shape N_all x C with probabilities of belonging to each of the C classes
            y_true: tensor of shape N_labeled x C with one-hot encoding indicating the class associated
                    to each row in the "labeled" dataset

            mask: tensor with idx associated to labeled data

        Returns: corrected probabilities predictions
        """
        with g.local_scope():

            # We save the number of element in the mask
            assert abs(float(y_soft.sum()) / y_soft.size(0) - 1.0) < 1e-2
            numel = int(mask.sum()) if mask.dtype == torch.bool else mask.size(0)
            assert y_true.size(0) == numel

            # We modify the y_true if it is not already in the one-hot encoding format
            if y_true.dtype == torch.long:
                y_true = F.one_hot(y_true.view(-1), y_soft.size(-1)).to(y_soft.dtype)

            # We initialize all the errors to 0
            error = torch.zeros_like(y_soft)

            # We calculate the real errors on the labeled set
            error[mask] = y_true - y_soft[mask]

            # We proceed to error propagation and return the correction prediction
            if self.autoscale:
                smoothed_error = self.prop1(g, error, post_step=lambda x: x.clamp_(-1., 1.))
                sigma = error[mask].abs().sum() / numel
                scale = sigma / smoothed_error.abs().sum(dim=1, keepdim=True)
                scale[scale.isinf() | (scale > 1000)] = 1.0

                result = y_soft + scale * smoothed_error
                result[result.isnan()] = y_soft[result.isnan()]
                return result
            else:
                def fix_input(x):
                    x[mask] = error[mask]
                    return x

                smoothed_error = self.prop1(g, error, post_step=fix_input)

                result = y_soft + self.scale * smoothed_error
                result[result.isnan()] = y_soft[result.isnan()]
                return result

    def smooth(self, g, y_soft, y_true, mask):
        """
        Smooth the predicted probabilities using label propagation

        Args:
            g: dgl homogeneous undirected graph
            y_soft: tensor of shape N_all x C with probabilities of belonging to each of the C classes
            y_true: tensor of shape N_labeled x C with one-hot encoding indicating the class associated
                    to each row in the "labeled" dataset

            mask: tensor with idx associated to labeled data

        Returns: corrected probabilities predictions

        """
        with g.local_scope():

            # We save the number of element in the mask
            numel = int(mask.sum()) if mask.dtype == torch.bool else mask.size(0)
            assert y_true.size(0) == numel

            # We modify the y_true if it is not already in the one-hot encoding format
            if y_true.dtype == torch.long:
                y_true = F.one_hot(y_true.view(-1), y_soft.size(-1)).to(y_soft.dtype)

            # We set the predicted probabilities to the real labels for the labeled set
            y_soft[mask] = y_true

            # We proceed to label propagation
            return self.prop2(g, y_soft)