import torch
import torch.nn as nn
try:
    from utils import set_device
except ImportError:
    from ..utils import set_device


def batch_norm(features,use_batch_norm,**kwargs):
    if batch_norm:
        return BatchNorm1d(features, **kwargs)
    else:
        return nn.Identity()

class BatchNorm1d(nn.Module):
    def __init__(self,*args,**kwargs):
        super(BatchNorm1d, self).__init__()
        self.BN1D = nn.BatchNorm1d(*args,**kwargs)
        self.IN1D = nn.InstanceNorm1d(*args,**kwargs)

    def forward(self, x):
        if len(x.shape)<=1 or x.shape[0]<=1:
            return self.IN1D(x)
        else:
            return self.BN1D(x)



class ResidualBlock(nn.Module):
    def __init__(self, features, act = nn.ELU, use_batch_norm=True, dropout_p = 0.0):
        super(ResidualBlock, self).__init__()
        layers = [batch_norm(features,use_batch_norm),act(),
                  nn.Linear(features, features), batch_norm(features,use_batch_norm),act(),
                  nn.Dropout(p=dropout_p),nn.Linear(features,features)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        out = self.block(x)
        out += identity
        return out 

class FullyConnectedResNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, **kwargs):
        super(FullyConnectedResNet, self).__init__()

        # Define blocks of different dimensions
        blocks = []
        resize_layers = []
        for i, hidden_dim in enumerate(hidden_dims):
             blocks.append(ResidualBlock(hidden_dim,**kwargs))
             if i<len(hidden_dims)-1:
                 resize_layers.append(nn.Linear(hidden_dim,hidden_dims[i+1]) if hidden_dim != hidden_dims[i+1] else nn.Identity())
        resize_layers.append(nn.Linear(hidden_dims[-1],output_dim))
        self.blocks = nn.ModuleList(blocks)
        self.resize_layers = nn.ModuleList(resize_layers) 
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])

    def forward(self, x):
        out = self.input_layer(x)
        # Pass through blocks, adjusting dimensions as needed
        for i, block in enumerate(self.blocks):
            out = self.resize_layers[i](block(out))
        return out

class TwoStageEmbeddingNet(nn.Module):
    def __init__(self,stage_1_input_dim, unaugmented_data, stage_2_module, *stage_2_args,**stage_2_kwargs):
        super(TwoStageEmbeddingNet, self).__init__()
        stage_1_layer = nn.Linear(stage_1_input_dim,stage_2_args[0])
        stage_2_net = stage_2_module(*stage_2_args, **stage_2_kwargs)
        self._initialize_with_weights(stage_1_layer,unaugmented_data,stage_2_args[0])
        self.embedding_net = nn.Sequential(stage_1_layer,stage_2_net)
    
    def _initialize_with_weights(self, layer, data, n_comp):
        """Initialize the weights of the layer using SVD."""
        with torch.no_grad():
            U, S, Vt = torch.linalg.svd(data, full_matrices=False)
            layer.weight.data = set_device(torch.zeros_like(layer.weight.data, device=data.device))
            layer.weight.data[:,:data.shape[-1]] = Vt[:n_comp] 
            layer.bias.data = torch.zeros_like(layer.bias.data,device=data.device)

    def forward(self,x):
        return self.embedding_net(x)

class FCEmbeddingNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, act = nn.ReLU,use_batch_norm=True,dropout_p=0):
        super(FCEmbeddingNet, self).__init__()
        
        if len(hidden_dims)==0:
            self.embedding_net = nn.Identity()
        else:
            layers = [nn.Linear(input_dim,hidden_dims[0]),batch_norm(hidden_dims[0],use_batch_norm),act()]
            for i,hidden_dim in enumerate(hidden_dims[1:]):
                layers.append(nn.Linear(hidden_dims[i],hidden_dim))
                layers.append(batch_norm(hidden_dim,use_batch_norm))
                layers.append(act())
                layers.append(nn.Dropout(p=dropout_p))
            layers.append(nn.Linear(hidden_dims[-1],output_dim))
            layers.append(batch_norm(output_dim,use_batch_norm))
            layers.append(act())
            self.embedding_net = nn.Sequential(*layers)

    def forward(self,x):
        return self.embedding_net(x)


class Module_merger(nn.Module):
    def __init__(self, split_index,module_1, module_2):
        super(Module_Merger, self).__init__()
        self.module_1 = module_1
        self.module_2 = module_2
        self.idx = split_index

    def forward(self,x):
        out = torch.cat([self.module_1(x[:,:self.idx]),self.module_2(x[:,self.idx:])],dim=1)
        return out
        



