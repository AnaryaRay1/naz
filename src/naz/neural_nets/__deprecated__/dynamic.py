from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint_adjoint as odeint


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv') != -1:
        nn.init.constant_(m.weight, 0)
        nn.init.normal_(m.bias, 0, 0.01)

class HyperLinear(nn.Module):
    def __init__(self, dim_in, dim_out, hypernet_dim=8, n_hidden=1, activation=nn.Tanh, **unused_kwargs):
        super(HyperLinear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.params_dim = self.dim_in * self.dim_out + self.dim_out

        layers = []
        dims = [1] + [hypernet_dim] * n_hidden + [self.params_dim]
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i - 1], dims[i]))
            if i < len(dims) - 1:
                layers.append(activation())
        self._hypernet = nn.Sequential(*layers)
        self._hypernet.apply(weights_init)

    def forward(self, t, x):
        params = self._hypernet(t.view(1, 1)).view(-1)
        b = params[:self.dim_out].view(self.dim_out)
        w = params[self.dim_out:].view(self.dim_out, self.dim_in)
        return F.linear(x, w, b)

class HyperResidualBlock(nn.Module):
    def __init__(self, features, act = nn.ELU, use_batch_norm=True, dropout_p = 0.0, hyperlayer = HyperLinear, **kwargs): 
        super(HyperResidualBlock, self).__init__()
        self.layer_1 = nn.Sequenctial(*[nn.GroupNorm(16, features, eps=1e-4), act()])
        self.hyper_layer_1 = hyperlayer(features,features, **kwargs)
        self.layer_2 = nn.Sequential(*[nn.GroupNorm(16, dim, eps=1e-4), act(), nn.Dropout(dropout_p)])
        self.hyper_layer_2 = hyperlayer(features, features, **kwargs)

    def forward(self, t, x):
        res = x
        out = self.layer_1(x)
        out = self.hyper_layer_1(t,out)
        out = self.layer_2(out)
        out = self.hyper_layer_2(t,out)
        out += res
        return out



class HyperNet(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims, act = nn.ELU, hyperlayer = HyperLinear, use_batch_norm = True, dropout_p = 0.0, **kwargs):
        super(HyperNet, self).__init__()
        layers = []
        if type(hidden_dims) != list:
            hidden_dims = [ hidden_dims]
        hidden_dims = [input_dim,]+hidden_dims+[output_dim,]
        for i, hidden_dim in enumerate(hidden_dims[1:]):
             layers.append(hyperlayer(hidden_dims[i-1], hidden_dim, **kwargs))
        self.layers = nn.ModuleList(blocks)
        self.f = act()
        self.dropout  = nn.Dropout(p = dropout_p)

    def forward(self, t, x):
        out = x
        for layer in self.layers:
            out = self.dropout(self.f(layer(t,out)))
        return out



class HyperResNet(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims, **kwargs):
        super(HyperResNet, self).__init__()
        blocks = []
        resize_layers = []
        if type(hidden_dims) != list:
            hidden_dims = [ hidden_dims]
        for i, hidden_dim in enumerate(hidden_dims):
             blocks.append(HyperResidualBlock(hidden_dim,**kwargs))
             if i<len(hidden_dims)-1:
                 resize_layers.append(kwargs["hyperlayer"](hidden_dim,hidden_dims[i+1], **kwargs) )
        resize_layers.append(nn.Linear(hidden_dims[-1],output_dim))
        self.blocks = nn.ModuleList(blocks)
        self.resize_layers = nn.ModuleList(resize_layers)
        self.input_layer = kwargs["hyperlayer"](input_dim, hidden_dims[0], **kwargs)

    def forward(self, t, x):
        out = self.input_layer(t,x)
        for i, block in enumerate(self.blocks):
            out = block(t,out)
            out = self.resize_layers[i](t,out)

        return out

class ODEfunc(nn.Module):
    def __init__(self, odenet, residual = True):
        super(ODEfunc, self).__init__()
        self.odenet = odenet
        self.residual = residual

    def forward(self, t, states):
        z, _ = states
        z.requires_grad_(True)
        with torch.set_grad_enabled(True):
            dz = self.odenet(t, z)
            eps = torch.randn_like(z)
            eps.requires_grad_(True)
            jacobian_vector = torch.autograd.grad(dz, z, eps, create_graph=True)[0]
            trace = torch.sum(jacobian_vector * eps, dim=1, keepdim=True)
            if self.residual:
                dz = dz - z
                trace -= torch.ones_like(divergence) * torch.tensor(np.prod(z.shape[1:]), dtype=torch.float32
                                                                     ).to(trace)
            dlogp = -trace

        return (dz, dlogp) 



        







