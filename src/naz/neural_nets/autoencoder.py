import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from sbi import utils; from sbi.neural_nets import posterior_nn
from sbi.inference import SNPE

import torch.optim as optim
import tqdm
import h5py
import copy

from ..utils import set_device
from ..neural_nets.fully_connected import ResidualBlock

def extract_features(rates):
     pca = PCA(svd_solver="full",whiten=True)
     pca.fit(rates)
     vr = pca.explained_variance_ratio_
     cum_vr  = np.cumsum(vr)
     n_comp = len(cum_vr[cum_vr<0.990])
     pca = PCA(n_components = n_comp, svd_solver="full",whiten=True)
     rates_transformed = pca.fit_transform(rates)
     components = pca.components_
     means = pca.mean_
     vrs = pca.explained_variance_
     #rates[0,:] = rates_transformed[0,:] @ ((vrs[:,None]**0.5)*components) + means
     return rates_transformed, components, vrs, means, pca, n_comp

def auto_encoder_nets(input_dim, hidden_features=[512,256], latent_dim=128,act = nn.ReLU, dropout_p=None,batch_norm=True):
    layers = [ ]
    for i, num_features in enumerate(hidden_features):
        if i==0:
            layers.append(nn.Linear(input_dim,num_features))
        else:
            layers.append(nn.Linear(hidden_features[i-1],num_features))
        
        if batch_norm:
            layers.append(nn.BatchNorm1d(num_features))
        layers.append(act())
        if dropout_p is not None:
            layers.append(nn.Dropout(p=dropout_p))
        
    layers.append(nn.Linear(hidden_features[-1],latent_dim))
    if batch_norm:
        layers.append(nn.BatchNorm1d(latent_dim))
    layers.append(nn.Softmax())
    
    encoder_net = nn.Sequential(*layers)

    layers = [ ]
    for i in range(len(hidden_features)):
        if i==0:
            layers.append(nn.Linear(latent_dim,hidden_features[-1-i]))
        else:
            layers.append(nn.Linear(hidden_features[-i],hidden_features[-1-i]))
        layers.append(act())
    layers.append(nn.Linear(hidden_features[0],input_dim))
    layers.append(nn.Softmax())
    decoder_net = nn.Sequential(*layers)
    return encoder_net, decoder_net

def res_auto_encoder_blocks(input_dim, latent_dim, hidden_features, **kwargs):
    
    encoder_blocks, encoder_resize = [ ], [ ]
    for i, hidden_dim in enumerate(hidden_features):
        encoder_blocks.append(ResidualBlock(hidden_dim, **kwargs))
        if i<len(hidden_features)-1:
            encoder_resize.append(nn.Linear(hidden_dim,hidden_features[i+1]) if hidden_features[i+1]!=hidden_dim else nn.Identity())
    encoder_resize.append(nn.Linear(hidden_features[-1],latent_dim))

    decoder_blocks, decoder_resize = [ ], []
    for i in range(len(hidden_features)):
        decoder_blocks.append(ResidualBlock(hidden_features[-1-i], **kwargs))
        if i<len(hidden_features)-1:
            decoder_resize.append(nn.Linear(hidden_features[-1-i],hidden_features[-2-i]) if hidden_features[-i-1]!=hidden_features[-i-2] else nn.Identity())
    decoder_resize.append(nn.Linear(hidden_features[0],input_dim))
    return {"encoder_blocks":nn.ModuleList(encoder_blocks), "encoder_resize":nn.ModuleList(encoder_resize), "decoder_blocks":nn.ModuleList(decoder_blocks), "decoder_resize":nn.ModuleList(decoder_resize), "encoder_input_layer": nn.Linear(input_dim,hidden_features[0]), "decoder_input_layer":nn.Linear(latent_dim,hidden_features[-1])}


class AutoEncoder(nn.Module):

    def __init__(self, input_dim, hidden_features=[512,256], latent_dim=128,act = nn.ReLU, dropout_p=None,batch_norm=True):

        super().__init__()
        self.encoder,self.decoder = auto_encoder_nets(input_dim, hidden_features=hidden_features, latent_dim=latent_dim,act = act, dropout_p=None,batch_norm=True)


    def forward(self,x):
        latent_variable = self.encoder(x)
        reconstruction = self.decoder(latent_variable)
        return latent_variable, reconstruction

class ResAutoEncoder(nn.Module):

    def __init__(self, input_dim,latent_dim, hidden_features, **kwargs):
        super(ResAutoEncoder, self).__init__()
        _nets = res_auto_encoder_blocks(input_dim, latent_dim, hidden_features, **kwargs)
        encoder_net = [ _nets["encoder_input_layer"]]
        for i, encoder_block in enumerate(_nets["encoder_blocks"]):
            encoder_net+=[encoder_block, _nets["encoder_resize"][i]]
        decoder_net = [ _nets["decoder_input_layer"]]
        for i, decoder_block in enumerate(_nets["decoder_blocks"]):
            decoder_net+=[decoder_block, _nets["decoder_resize"][i]]
        encoder_net.append(nn.Softmax())
        decoder_net.append(nn.Softmax())
        self.encoder = nn.Sequential(*encoder_net)
        self.decoder = nn.Sequential(*decoder_net)
        

    def forward(self, x):
        latent_variable  = self.encoder(x)
        reconstruction = self.decoder(latent_variable)
        return latent_variable, reconstruction


def normalized_covariance_loss(latent_z, theta):
    """
    Computes the normalized covariance (correlation) loss between latent_z and theta.
    """
    # Center the variables
    z_centered = latent_z - latent_z.mean(dim=0, keepdim=True)
    theta_centered = theta - theta.mean(dim=0, keepdim=True)
    
    # Compute standard deviations
    z_std = latent_z.std(dim=0, keepdim=True) + 1e-8  # Add epsilon for numerical stability
    theta_std = theta.std(dim=0, keepdim=True) + 1e-8
    
    # Normalize
    z_normalized = z_centered / z_std
    theta_normalized = theta_centered / theta_std
    
    # Compute pairwise correlations
    correlation_matrix = torch.mm(theta_normalized.T, z_normalized) / (latent_z.size(0) - 1)
    
    # Return the negative mean correlation as the loss
    return 1.-(correlation_matrix**2).mean()





