try:
    from utils import device, set_device
except ImportError:
    from ..utils import device, set_device

try:
    from neural_nets.fully_connected import FullyConnectedResNet
    from neural_nets.neural_odes.ffjord import Ffjord
except ImportError:
    from ..neural_nets.fully_connected import FullyConnectedResNet
    from ..neural_nets.neural_odes.ffjord import Ffjord


import torch
from torch import nn
from pyro.distributions.transforms  import ComposeTransformModule
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.conditional import ConditionalTransformModule, ConditionalComposeTransformModule
from torch.distributions import constraints
from functools import partial

class Struct:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class FCRN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FCRN, self).__init__()
        args[1]*=2
        self.dim = args[1]
        self.net = FullyConnectedResNet(*args, **kwargs)
    def forward(self, x):
        out = self.net(x)
        return (out[:, :dim], out[:, dim:])

class FFJORDTransform(TransformModule):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    
    def __init__(self, net, loc = set_device(0), log_scale = set_device(0)):
        super(FFJORDTransform, self).__init__()
        self.nn = net
        self.loc = loc
        self.log_scale = log_scale

    def _inverse(self, x):  # forward: z1 → z0
        loc = self.loc.expand(x.shape)
        log_scale = self.log_scale.expand(x.shape)
        z, logdet = self.nn(x)
        self._cached_logdet = logdet-self.log_scale.sum(-1)
        return (z-loc)/torch.exp(log_scale)

    def _call(self, z):  # inverse: z0 → z1
        loc = self.loc.expand(z.shape)
        log_scale = self.log_scale.expand(z.shape)
        x, logdet = self.nn.backward((z+loc)*torch.exp(log_scale))
        return x

    def log_abs_det_jacobian(self, x, y):
        return self._cached_logdet


class ConditionalFFJORDTransform(ConditionalTransformModule):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, net, condition_net):
        super().__init__()
        self.nn = net
        self.cond_nn = condition_net


    def condition(self, context):
        loc, log_scale = self.cond_nn(context)
        return FFJORDTransform(self.nn, loc = loc, log_scale = log_scale)


def continuous_free_form(theta_dim, condition_dim, hidden_dims, cond_activation = nn.ELU, use_batchnorm = False, dropout_p = None, **kwargs):
    datatype = kwargs.pop("datatype", torch.float32)
    net = Ffjord([theta_dim], cfg = Struct(**kwargs))

    if dropout_p is None:
        dropout_p = 0.0
    if condition_dim == 0:
        transform = FFJORDTransform(net).to(device)
        nets = [net]
    else:
        condition_net = FCRN(condition_dim, input_dim, hidden_dims, act = cond_activation, use_batch_norm = use_batchnorm, dropout_p = dropout_p).to(device)
        transform = ConditionalFFJORDTransform(condition_net, net).to(device)
        nets = [net, condition_net]


    flow = ComposeTransformModule([transform]).to(device) if condition_dim <= 0 else ConditionalComposeTransformModule([transform]).to(device)
    return flow, [transform], nets
    
    




