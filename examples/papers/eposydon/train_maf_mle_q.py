import numpy as np
import pickle
import os
import tqdm
import sys
import h5py

import torch
from naz.utils import set_device
from naz.flows.flow import NormalizingFlow
from naz.trainers.train_flows import train, train_lightning

import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

parser = argparse.ArgumentParser(description = "Train MAF MLE")

parser.add_argument('--epistemic-only', type=str2bool, nargs='?', const=True, default=False,
                    help='Whether or not to re-run on the same dataset')
parser.add_argument('--nhidden', type=int, default=512,
                    help='number of hidden units')
parser.add_argument('--batchfrac', type=int, default=10000,
                    help='size of batches')
parser.add_argument('--nlayer', type=int, default=5,
                    help='number of hidden layers')
parser.add_argument('--nflow', type=int, default=16,
                    help='number of flow layers')
parser.add_argument('--fthin', type=int, default=1,
                    help='number of batches to split dataset into')
parser.add_argument('--popsynth-file', type=str,
                    help='h5 file containing synthesized binaries')
parser.add_argument('--dir',type=str)

args = parser.parse_args()

avg = args.epistemic_only
fthin = int(args.fthin)
popsynth_file = args.popsynth_file
nh = int(args.nhidden)
nhl = int(args.nlayer)
num_layers = int(args.nflow)

os.makedirs(args.dir, exist_ok=True)

with h5py.File(popsynth_file, "r") as hf:
    np.random.seed(42)
    thetas = hf["theta"][()]
    N = len(thetas)
    rand_indices = np.random.choice(N, size = int(N/fthin))
    thetas = thetas[rand_indices,:]
    thetas[:,:1] = np.log(thetas[:,:1]) # only logging m1
    thetas[:,2] = np.log(thetas[:,2]) # logging time
    lambdas = hf["lambda"][()][rand_indices,:]
    lambdas[:,0] = np.log(lambdas[:,0]) # logging metallicity
    
theta_scaler = StandardScaler()
lambda_scaler = StandardScaler()
thetas_scaled = theta_scaler.fit_transform(thetas)
lambdas_scaled = lambda_scaler.fit_transform(lambdas)

hidden_dims = [nh for i in range(nhl)]
num_layers = 16

flow = NormalizingFlow('maf', 
                       None, 
                       thetas.shape[-1],
                       lambdas.shape[-1], 
                       hidden_dims, 
                       num_layers)#, activation = nn.ReLU)

model, history, history_val, best_mse,best_epoch = train(flow, 
                                                         set_device(thetas), 
                                                         set_device(lambdas), 
                                                         train_frac = 0.89, 
                                                         patience = 64, 
                                                         lr = 1e-3, min_lr = 1e-9, 
                                                         num_epochs = 4096, 
                                                         batch_frac = 0.05, 
                                                         lr_decay = 0.5, 
                                                         return_final = True)

label = f"maf_{nh}_{nhl}_b{args.batchsize}_f{args.nflow}"

save_dict = {
    "model": model,
    "theta_scaler": theta_scaler,
    "lambda_scaler": lambda_scaler
}

with open(os.path.join(args.dir, f'{label}.pkl'), 'wb') as f:
    pickle.dump(save_dict, f)
