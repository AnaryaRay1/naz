import numpy as np
import pickle
import os
import tqdm
import sys
import h5py

import torch
from sklearn.preprocessing import StandardScaler

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


parser = argparse.ArgumentParser(description = "Train CNF MLE")

parser.add_argument('--epistemic-only', type=str2bool, nargs='?', const=True, default=False,
                    help='Whether or not to re-run on the same dataset')
parser.add_argument("--hiddendims", type=int, 
                    default=[128, 128, 128, 128], nargs='+',
                    help="List of hidden layer sizes, e.g. 256 128 128 128")
parser.add_argument('--nflow', type=int, default=1,
                    help='number of CNF blocks to stack')
parser.add_argument('--batchsize', type=int, default=10000,
                    help='size of batches')
parser.add_argument('--fthin', type=int, default=1,
                    help='number of batches to split dataset into')
parser.add_argument('--popsynth-file', type=str,
                    help='h5 file containing synthesized binaries')
parser.add_argument('--dir',type=str)
parser.add_argument('--suffix',type=str,default='')

args = parser.parse_args()

avg = args.epistemic_only
fthin = int(args.fthin)
popsynth_file = args.popsynth_file

os.makedirs(args.dir, exist_ok=True)

with h5py.File(popsynth_file, "r") as f:
    np.random.seed(42)
    thetas = f["theta"][()]
    N = len(thetas)
    rand_indices = np.random.choice(N, size = int(N/fthin))
    thetas = thetas[rand_indices,:]
    thetas[:,0] = np.log(thetas[:,0]) # logging m1
    thetas[:,2] = np.log(thetas[:,2]) # logging time
    lambdas = f["lambda"][()][rand_indices,:]
    lambdas[:,0] = np.log(lambdas[:,0]) # logging metallicity

theta_scaler = StandardScaler()
lambda_scaler = StandardScaler()
thetas_scaled = theta_scaler.fit_transform(thetas)
lambdas_scaled = lambda_scaler.fit_transform(lambdas)

flow = NormalizingFlow("cnf", 
                       None, 
                       thetas_scaled.shape[-1], 
                       lambdas_scaled.shape[-1], 
                       args.hiddendims, 
                       args.nflow)

model = train_lightning(flow, set_device(thetas_scaled), 
                        set_device(lambdas_scaled), 
                        num_epochs = 1024,
                        batch_size = args.batchsize)

hidden_str = "_".join(str(h) for h in args.hiddendims)
label = f"cnf_{hidden_str}_f{args.nflow}_b{args.batchsize}"

save_dict = {
    "model": model,
    "theta_scaler": theta_scaler,
    "lambda_scaler": lambda_scaler
}

with open(os.path.join(args.dir, f'{label}{args.suffix}.pkl'), 'wb') as f:
    pickle.dump(save_dict, f)
    
