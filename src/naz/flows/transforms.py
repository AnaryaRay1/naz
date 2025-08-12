"""Transformation functions for constructing normalizing flows"""
__author__ = "Anarya Ray <anarya.ray@northwestern.edu>"

import torch
from torch import nn

import pyro
import pyro.distributions.transforms as T
from pyro.distributions.conditional import ConditionalTransformModule, ConditionalComposeTransformModule
from pyro.distributions import constraints
from pyro.nn import ConditionalAutoRegressiveNN, ConditionalDenseNN, AutoRegressiveNN, DenseNN
try:
    from utils import set_device, device
except ImportError:
    from ..utils import set_device, device

import copy


def bounding_transform(x, low, high):
    y = (x - low.expand(x.shape))/((high-low).expand(x.shape))
    log_jac = - torch.sum( torch.log(y)+torch.log1p(-y), axis=-1) - torch.sum(torch.log(high-low))
    return torch.logit(y), log_jac

def inverse_bounding_transform(y, low, high):
    x = torch.sigmoid(y)
    return x*((high-low).expand(y.shape)) + low.expand(y.shape)

class ConditionalAutoRegressiveNNDropout(ConditionalAutoRegressiveNN):
    def __init__(self, *args, dropout_p=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp_layers = []
        #for i in range(1, len(self.layers)-1):
        for i in range(len(self.layers)-1):
            self.dp_layers.append(nn.Dropout(dropout_p))
        self.dp_layers = nn.ModuleList(self.dp_layers)

    def _forward(self, x: torch.Tensor):# -> Union[Sequence[torch.Tensor], torch.Tensor]:
        h = x
        for i,layer in enumerate(self.layers[:-1]):
            h = self.dp_layers[i](self.f(layer(h)))

        h = self.layers[-1](h)

        if self.skip_layer is not None:
            h = h + self.skip_layer(x)

        # Shape the output, squeezing the parameter dimension if all ones
        if self.output_multiplier == 1:
            return h
        else:
            h = h.reshape(
                list(x.size()[:-1]) + [self.output_multiplier, self.input_dim]
            )

            # Squeeze dimension if all parameters are one dimensional
            if self.count_params == 1:
                return h

            elif self.all_ones:
                return torch.unbind(h, dim=-2)

            # If not all ones, then probably don't want to squeeze a single dimension parameter
            else:
                return tuple([h[..., s, :] for s in self.param_slices])

class ConditionalDenseNNDropout(ConditionalDenseNN):

    def __init__(self, *args, dropout_p=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp_layers = []
        for i in range(len(self.layers)-1):
            self.dp_layers.append(nn.Dropout(dropout_p))
        self.dp_layers = nn.ModuleList(self.dp_layers)

    def _forward(self, x):
        """
        The forward method
        """
        h = x
        for layer in self.layers[:-1]:
            h = self.dropout_layers[i](self.f(layer(h)))
        h = self.layers[-1](h)

        # Shape the output, squeezing the parameter dimension if all ones
        if self.output_multiplier == 1:
            return h
        else:
            h = h.reshape(list(x.size()[:-1]) + [self.output_multiplier])

            if self.count_params == 1:
                return h

            else:
                return tuple([h[..., s] for s in self.param_slices])

class AutoRegressiveNNDropout(ConditionalAutoRegressiveNNDropout):

    def __init__(self, *args, **kwargs):
        super(AutoRegressiveNNDropout, self).__init__(args[0], 0, *args[1:], **kwargs)

    def forward(self, x):
        return self._forward(x)

class DenseNNDropout(ConditionalDenseNNDropout):

    def __init__(self, *args, **kwargs):
        super(DenseNNDropout, self).__init__(args[0], 0, *args[1:], **kwargs)

    def forward(self, x):
        return self._forward(x)

class ConditionalSplineCoupling(ConditionalTransformModule):
    
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, input_dim, split_dim, dense_nn, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.split_dim = split_dim
        self.nn = dense_nn
        self.kwargs = kwargs

    def condition(self, context):
        
        cond_nn = partial(self.nn, context=context)
        return SplineCoupling(self.input_dim, self.split_dim, cond_nn, **self.kwargs)



def masked_affine_autoregressive(theta_dim, condition_dim, hidden_dim, num_layers, activation = nn.Tanh(), use_batchnorm = False, random_mask = True, random_perm = False, dropout_p = None):
    transforms, nets = [ ], [ ]
    for _ in range(num_layers):
        if type(hidden_dim) == list:
            hidden_dims = hidden_dim
        else:
            hidden_dims = [hidden_dim]
        if random_mask:
            if dropout_p is None:
                arn = ConditionalAutoRegressiveNN(theta_dim, condition_dim, hidden_dims, nonlinearity=activation).to(device) if condition_dim > 0 else AutoRegressiveNN(theta_dim, hidden_dims, nonlinearity=activation).to(device)
            else:
                arn = ConditionalAutoRegressiveNNDropout(theta_dim, condition_dim, hidden_dims, nonlinearity=activation, dropout_p = dropout_p).to(device) if condition_dim > 0 else AutoRegressiveNNDropout(theta_dim, hidden_dims, nonlinearity=activation, dropout_p = dropout_p).to(device)

        else:
            if dropout_p is None:
                arn = ConditionalAutoRegressiveNN(theta_dim, condition_dim, hidden_dims, nonlinearity=activation, permutation = set_device(torch.arange(theta_dim)).type(dtype=torch.int64)).to(device) if condition_dim > 0 else AutoRegressiveNN(theta_dim, hidden_dims, nonlinearity=activation, permutation = set_device(torch.arange(theta_dim)).type(dtype=torch.int64)).to(device) 
            else:
                arn = ConditionalAutoRegressiveNNDropout(theta_dim, condition_dim, hidden_dims, nonlinearity=activation, permutation = set_device(torch.arange(theta_dim), dropout_p = dropout_p).type(dtype=torch.int64)).to(device) if condition_dim > 0 else AutoRegressiveNNDropout(theta_dim, hidden_dims, nonlinearity=activation, permutation = set_device(torch.arange(theta_dim), dropout_p = dropout_p).type(dtype=torch.int64)).to(device)
        nets.append(arn)
        transform = T.ConditionalAffineAutoregressive(arn).to(device) if condition_dim > 0 else T.AffineAutoregressive(arn).to(device)
        transforms.append(transform)
        if random_perm:
            transforms.append(T.Permute(set_device(torch.randperm(theta_dim))))
        if use_batchnorm:
            transforms.append(T.BatchNorm(theta_dim).to(device))
         
    flow = T.ComposeTransformModule(transforms).to(device) if condition_dim <= 0 else ConditionalComposeTransformModule(transforms).to(device)
    return flow, transforms, nets
         



def neural_spline_autoregressive(theta_dim, condition_dim, hidden_dim, num_layers, count_bins, order = "quadratic", activation = nn.Tanh(), use_batchnorm = False, random_mask = True, random_perm = False, dropout_p = None):
    if order == "linear":
        paramdim = [count_bins, count_bins, count_bins -1, count_bins]
    elif order == "quadratic":
        paramdim = [count_bins, count_bins, count_bins -1]
    else:
        raise
    transforms, nets = [ ], [ ]
    for _ in range(num_layers):
        if type(hidden_dim) == list:
            hidden_dims = hidden_dim
        else:
            hidden_dims = [hidden_dim]
        if random_mask:
            if dropout_p is None:
                arn = ConditionalAutoRegressiveNN(theta_dim, condition_dim, hidden_dims, nonlinearity=activation, param_dims = paramdim).to(device) if condition_dim > 0 else AutoRegressiveNN(theta_dim, hidden_dims, nonlinearity=activation, param_dims = paramdim).to(device)
            else:
                arn = ConditionalAutoRegressiveNNDropout(theta_dim, condition_dim, hidden_dims, nonlinearity=activation, dropout_p = dropout_p, param_dims = paramdim).to(device) if condition_dim > 0 else AutoRegressiveNNDropout(theta_dim, hidden_dims, nonlinearity=activation, dropout_p = dropout_p, param_dims = paramdim).to(device)

        else:
            if dropout_p is None:
                arn = ConditionalAutoRegressiveNN(theta_dim, condition_dim, hidden_dims, nonlinearity=activation, permutation = set_device(torch.arange(theta_dim), param_dims = paramdim).type(dtype=torch.int64)).to(device) if condition_dim > 0 else AutoRegressiveNN(theta_dim, nonlinearity=activation, permutation = set_device(torch.arange(theta_dim), param_dims = paramdim).type(dtype=torch.int64)).to(device)
            else:
                arn = ConditionalAutoRegressiveNNDropout(theta_dim, condition_dim, hidden_dims, nonlinearity=activation, permutation = set_device(torch.arange(theta_dim), dropout_p = dropout_p, param_dims = paramdim).type(dtype=torch.int64)).to(device) if condition_dim > 0 else AutoRegressiveNNDropout(theta_dim, hidden_dims, nonlinearity=activation, permutation = set_device(torch.arange(theta_dim), dropout_p = dropout_p, param_dims = paramdim).type(dtype=torch.int64)).to(device)
        nets.append(arn)
        transform = T.ConditionalSplineAutoregressive(theta_dim, arn, count_bins = count_bins, order = order).to(device) if condition_dim > 0 else T.SplineAutoregressive(theta_dim, arn, count_bins = count_bins, order = order).to(device)
        transforms.append(transform)
        if random_perm:
            transforms.append(T.Permute(set_device(torch.randperm(theta_dim))))
        if use_batchnorm:
            transforms.append(T.BatchNorm(theta_dim).to(device))
         
    flow = T.ComposeTransformModule(transforms).to(device) if condition_dim <= 0  else ConditionalComposeTransformModule(transforms).to(device)
    return flow, transforms, nets
            

def neural_spline_coupling(theta_dim, condition_dim, hidden_dim, num_layers, count_bins, split_dim, order = "quadratic", activation = nn.Tanh(), use_batchnorm = False, random_perm = False, dropout_p = None):

    if order == "linear":
        param_dims = [(input_dim - split_dim) * count_bins,
            (input_dim - split_dim) * count_bins,
            (input_dim - split_dim) * (count_bins - 1),
            (input_dim - split_dim) * count_bins]

    elif order == "quadratic":
        param_dims = [(input_dim - split_dim) * count_bins,
            (input_dim - split_dim) * count_bins,
            (input_dim - split_dim) * (count_bins - 1)]

    else:
        raise
    transforms, nets = [ ], [ ]
    for _ in range(num_layers):
        if type(hidden_dim) == list:
            hidden_dims = hidden_dim
        else:
            hidden_dims = [hidden_dim]
        if dropout_p is None:
            arn = ConditionalDenseNN(split_dim, condition_dim, hidden_dims, nonlinearity=activation, param_dims = paramdim).to(device) if condition_dim > 0 else DenseNN(split_dim, hidden_dims, nonlinearity=activation, param_dims = paramdim).to(device)
        else:
            arn = ConditionalDenseNNDropout(theta_dim, condition_dim, hidden_dims, nonlinearity=activation, dropout_p = dropout_p, param_dims = paramdim).to(device) if condition_dim > 0 else DenseNNDropout(split_dim, hidden_dims, nonlinearity=activation, dropout_p = dropout_p, param_dims = paramdim).to(device)

        nets.append(arn)
        transform = ConditionalSplineCoupling(theta_dim, split_dim, arn, count_bins = count_bins, order = order).to(device) if condition_dim > 0 else T.SplineAutoregressive(theta_dim, split_dim, arn, count_bins = count_bins, order = order).to(device)
        transforms.append(transform)
        if random_perm:
            transforms.append(T.Permute(set_device(torch.randperm(theta_dim))))
        if use_batchnorm:
            transforms.append(T.BatchNorm(theta_dim).to(device))
         
    flow = T.ComposeTransformModule(transforms).to(device) if condotion_dim <=0 else ConditionalComposeTransformModule(transforms).to(device)
    return flow, transforms, nets
