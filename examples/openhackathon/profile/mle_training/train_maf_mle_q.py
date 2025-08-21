import numpy as np
import pickle
import os
import tqdm
import sys
import h5py

import torch
import torch.cuda.nvtx as nvtx
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp
import torch.distributed as dist
from sklearn.preprocessing import StandardScaler

with nvtx.range("import naz"):
    from naz.set_device_torch import set_device
    from naz.flows.flow import NormalizingFlow
    from naz.trainers.train_flows import train, train_lightning, train_manual


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

parser.add_argument('--nlayer', type=int, default=5,
                    help='number of hidden layers')

parser.add_argument('--nflow', type=int, default=16,
                    help='number of flow layers')

parser.add_argument('--index', type=int, default=0,
                    help='index of run')


parser.add_argument('--fthin', type=int, default=1,
                    help='number of batches to split dataset into')
parser.add_argument('--batchsize', type=int, default=100000,
                    help='per-gpu batch size')


parser.add_argument('--popsynth-file', type=str,
                    help='h5 file containing synthesized binaries')

parser.add_argument('--dir',type=str)

args = parser.parse_args()

index = int(args.index)
avg = args.epistemic_only
fthin = int(args.fthin)
popsynth_file = args.popsynth_file
nh = int(args.nhidden)
nhl = int(args.nlayer)
num_layers = int(args.nflow)
print(index, avg, fthin)

outdir = f"{args.dir}_mle_rerunrs_{'epistemic' if avg else 'aleatoric'}_{fthin}_4p/"

if not os.path.exists(outdir):
    try:
        os.mkdir(outdir)
    except:
        pass

with nvtx.range("load_data_from_file"):
    with h5py.File(popsynth_file, "r") as hf:
        np.random.seed(69+(index if not avg else 0 ))
        theta_train = hf["theta"][()]
        N = len(theta_train)
        rand_indices = np.random.choice(N, size = int(N/fthin))
        theta_train = theta_train[rand_indices,:]
        thetas = theta_train.copy()
        thetas[:,:1] = np.log(thetas[:,:1])
        thetas[:,2] = np.log(thetas[:,2])
        lambdas = hf["lambda"][()][rand_indices,:]
        lambdas[:,0] = np.log(lambdas[:,0])

theta_scaler = StandardScaler()
lambda_scaler = StandardScaler()
thetas_scaled = theta_scaler.fit_transform(thetas)
lambdas_scaled = lambda_scaler.fit_transform(lambdas)
        
hidden_dims = [nh for i in range(nhl)]

num_layers = 16

label = f"{int(hidden_dims[0])}_{len(hidden_dims)}_{int(num_layers)}_{index}_4p"

def run(rank, world_size, thetas, lambdas, hidden_dims, num_layers):

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    with nvtx.range("model_creation"):
        flow = NormalizingFlow('maf', None, thetas_scaled.shape[-1],
                            lambdas_scaled.shape[-1], 
                            hidden_dims, num_layers,device=device)
                            
        print("Flow first parameter device:", next(flow.parameters()).device)
        print("Base distribution loc device:", flow.base_dist.loc.device)
        print("Base distribution scale device:", flow.base_dist.scale.device)
        
        # Send input data to GPU immediately
        X = torch.as_tensor(thetas_scaled, dtype=torch.float16, device=device)
        Y = torch.as_tensor(lambdas_scaled, dtype=torch.float16, device=device)

    with nvtx.range("training_loop"):
        model, history, history_val, best_mse,best_epoch = train_manual(flow, X, Y, 
        rank=rank, world_size=world_size, device=device,
        train_frac = 0.89, patience = 64, lr = 1e-3, min_lr = 1e-9, 
        num_epochs = 10, per_gpu_batch_size = args.batchsize, lr_decay = 0.5, 
        return_final = True)
        
    if rank == 0:
        save_dict = {
        "model": model,
        "theta_scaler": theta_scaler,
        "lambda_scaler": lambda_scaler}
        with open(outdir + f'inference_mle_{label}.pkl', 'wb') as f:
            pickle.dump(save_dict, f)
            
    if dist.is_initialized():
        dist.destroy_process_group()


with nvtx.range("multiprocessing spawn"):
    if __name__ == "__main__":
        world_size = 4
        mp.spawn(run, args=(world_size, thetas, lambdas, hidden_dims, num_layers), 
                 nprocs=world_size,join=True)
                 
                 