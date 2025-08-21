import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import torch.distributed as dist

import torch.cuda.nvtx as nvtx

from pyro.infer import MCMC, NUTS, HMC, SVI, Importance, Trace_ELBO
import pyro.optim as poptim

import pytorch_lightning as pl


from sklearn.model_selection import train_test_split
import os
import numpy as np
import tqdm
import copy
from functools import partial


def get_params(flow):
    params = [ ]
    for t in flow.flow_dist.transforms:
        this_params = {}
        for name, param in t.named_parameters():
            this_params[name] = copy.deepcopy(param)
        params.append(this_params)

    return params

def set_params(flow, params, sample_idx = None):
    for i,t in enumerate(flow.flow_dist.transforms):
        for name, param in t.named_parameters():
            with torch.no_grad():
                if sample_idx is None:
                    param.copy_(params[i][name])
                else:
                    param.copy_(params[f"flow_{i}_{name}"][sample_idx])
                    
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '60640'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()
                    
def train(flow,x, y, rank, world_size, device,
            opt = optim.Adam, lr=0.001, num_epochs=1024, 
            train_frac=0.7, per_gpu_batch_size=64, lambda_l1=0., 
            lambda_l2 = 0., patience=32, min_epochs=128, clip_val=1.0, 
            lr_decay=0.5, min_lr=None, return_final = False):
            
    setup_ddp(rank, world_size)
    
    # Split train/test globally
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                            train_size=train_frac, shuffle=True)
    x_train, y_train = x_train.to(device), y_train.to(device)                                        
    x_test_gpu, y_test_gpu = x_test.to(device), y_test.to(device)

    with nvtx.range('wrap_DDP'):
        # Move flow to GPU and wrap in DDP
        flow = flow.to(device)
        ddp_flow = DDP(flow, device_ids=[rank])

    # Prepare dataset and sampler
    with nvtx.range('DataLoader_inside'):
        train_dataset = TensorDataset(x_train, y_train)
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, 
                                                    rank=rank, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=per_gpu_batch_size, 
                                    sampler=sampler, drop_last=True)

    # Optimizer & scheduler
    parameters = [p for t in flow.flow_dist.transforms for p in t.parameters()]
    optimizer = opt(parameters, lr=lr, weight_decay=lambda_l2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                    factor=lr_decay, patience=int(patience/2))

    best_mse = np.inf   # init to infinity
    best_weights = None
    history, history_val = [], []
    
    min_lr = lr*1e-3 if min_lr is None else min_lr
    n_noimprove = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        with nvtx.range(f"Epoch_{epoch}"):
            with nvtx.range("Train_flow"):
                ddp_flow.train()
                
            with nvtx.range("set epoch"):
                sampler.set_epoch(epoch)
            
            total_loss = 0.
            
            with nvtx.range("batch_loop"):
                for x_batch, y_batch in train_loader:
    
                    with nvtx.range("Batch_forward_backward"):
                        optimizer.zero_grad()
    
                        with nvtx.range("Forward_pass"):
                            loss = -ddp_flow.module.log_prob(x_batch, 
                                                        condition=y_batch).mean()
    
                        # L1 regularization
                        if lambda_l1 > 0.:
                            reg_loss = sum(param.abs().sum() for name,
                                        param in flow.named_parameters() if name.endswith('weight'))
                            loss += lambda_l1 * reg_loss
    
                        total_loss += float(loss)
    
                        with nvtx.range("Backward_pass"):
                            loss.backward()
    
                        with nvtx.range("Optimizer_step"):
                            if clip_val is not None:
                                nn.utils.clip_grad_norm_(flow.parameters(), clip_val)
                            optimizer.step()

            # Validation (only rank 0 prints/logs)
            if rank == 0:
                ddp_flow.eval()
                with torch.no_grad():
                    mse = -ddp_flow.module.log_prob(x_test_gpu, 
                                                    condition=y_test_gpu).mean()

                current_lr = optimizer.param_groups[0]['lr']
                scheduler.step(float(mse))

                history.append(float(total_loss)/len(train_loader))
                history_val.append(float(mse))

                if float(mse) < best_mse: # validation improved
                    best_epoch = epoch
                    best_mse = float(mse)
                    best_weights = copy.deepcopy(get_params(flow))
                    n_noimprove = 0
                elif epoch > min_epochs: # no improvement, over min epochs
                    n_noimprove += 1
                else: # no improvement, under min epochs
                    pass

                print(f"Epoch {epoch}, Train Loss={float(total_loss)/len(train_loader)}, "
                f"Val Loss={float(mse)}, Best Val Loss={best_mse}, LR={current_lr}, "
                f"No Improve={n_noimprove}")

                if epoch > min_epochs and n_noimprove > patience and current_lr < min_lr:
                    print(f"Network converged after {epoch} epochs")
                    break

    # Load best weights on rank 0
    if rank == 0 and best_weights is not None and not return_final:
        set_params(flow, best_weights)

    cleanup_ddp()

    if rank == 0:
        return flow, history, history_val, best_mse, best_epoch
    else:
        return None, None, None, None, None

def train_manual(flow,x, y, rank, world_size, device,
            opt = optim.Adam, lr=0.001, num_epochs=1024, 
            train_frac=0.7, per_gpu_batch_size=1e5, lambda_l1=0., 
            lambda_l2 = 0., patience=32, min_epochs=128, clip_val=1.0, 
            lr_decay=0.5, min_lr=None, return_final = False):
            
    setup_ddp(rank, world_size)
    
    # Split train/test globally
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                            train_size=train_frac, shuffle=True)
    # Move everything to GPU and wrap flow in DDP
    with nvtx.range("move_to_GPU"):
        x_train, y_train = x_train.to(device), y_train.to(device)
        x_test_gpu, y_test_gpu = x_test.to(device), y_test.to(device)
        num_samples = x_train.shape[0]
    
        flow = flow.to(device)
        ddp_flow = DDP(flow, device_ids=[rank])

    # Optimizer & scheduler
    parameters = [p for t in flow.flow_dist.transforms for p in t.parameters()]
    optimizer = opt(parameters, lr=lr, weight_decay=lambda_l2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                    factor=lr_decay, patience=int(patience/2))

    best_mse = np.inf 
    best_weights = None
    history, history_val = [], []
    
    min_lr = lr*1e-3 if min_lr is None else min_lr
    n_noimprove = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        with nvtx.range(f"Epoch_{epoch}"):
            ddp_flow.train()
                
            with nvtx.range("manual_shuffle"):
                if rank == 0:
                    perm = torch.randperm(num_samples, device=device)
                else:
                    perm = torch.empty(num_samples, dtype=torch.long, device=device)
                    
            with nvtx.range("broadcast"):
                dist.broadcast(perm, src=0)
                
            x_shuffle = x_train[perm]
            y_shuffle = y_train[perm]
            
            with nvtx.range("rank_slice"):   
                samples_per_rank = num_samples // world_size
                start_idx = rank * samples_per_rank
                end_idx = (rank + 1) * samples_per_rank if rank != world_size - 1 else num_samples
                
            x_rank = x_shuffle[start_idx:end_idx]
            y_rank = y_shuffle[start_idx:end_idx]

            total_loss = 0.

            with nvtx.range("manual_batching"):
                for i in range(0, x_rank.shape[0], per_gpu_batch_size):
                    x_batch = x_rank[i:i+per_gpu_batch_size]
                    y_batch = y_rank[i:i+per_gpu_batch_size]
        
                    with nvtx.range("Batch_forward_backward"):
                        optimizer.zero_grad()
    
                        with nvtx.range("Forward_pass"):
                            loss = -ddp_flow.module.log_prob(x_batch, 
                                                    condition=y_batch).mean()
    
                        # L1 regularization
                        if lambda_l1 > 0.:
                            reg_loss = sum(param.abs().sum() for name,
                                    param in flow.named_parameters() if name.endswith('weight'))
                            loss += lambda_l1 * reg_loss
    
                        total_loss += float(loss)
    
                        with nvtx.range("Backward_pass"):
                            loss.backward()
    
                        with nvtx.range("Optimizer_step"):
                            if clip_val is not None:
                                nn.utils.clip_grad_norm_(flow.parameters(), clip_val)
                            optimizer.step()

            # Validation (only rank 0 prints/logs)
            if rank == 0:
                ddp_flow.eval()
                with torch.no_grad():
                    mse = -ddp_flow.module.log_prob(x_test_gpu, 
                                                    condition=y_test_gpu).mean()

                scheduler.step(float(mse))
                history.append(float(total_loss) / len(range(0, x_rank.shape[0], per_gpu_batch_size)))
                history_val.append(float(mse))

                if float(mse) < best_mse: # validation improved
                    best_epoch = epoch
                    best_mse = float(mse)
                    best_weights = copy.deepcopy(get_params(flow))
                    n_noimprove = 0
                elif epoch > min_epochs: # no improvement, over min epochs
                    n_noimprove += 1
                else: # no improvement, under min epochs
                    pass

                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}, Train Loss={history[-1]}, Val Loss={history_val[-1]}, "
                      f"Best Val Loss={best_mse}, LR={current_lr}, No Improve={n_noimprove}")

                if epoch > min_epochs and n_noimprove > patience and current_lr < min_lr:
                    print(f"Network converged after {epoch} epochs")
                    break

    # Load best weights on rank 0
    if rank == 0 and best_weights is not None and not return_final:
        set_params(flow, best_weights)

    cleanup_ddp()

    if rank == 0:
        return flow, history, history_val, best_mse, best_epoch
    else:
        return None, None, None, None, None

def train_lightning(flow, theta_train, condition_train, opt = optim.AdamW, lr = 2e-3, lambda_l2 = 1e-5, batch_size = 10240, num_epochs = 600):
    X_train = torch.cat([theta_train, condition_train], dim = -1)
    trainloader = data.DataLoader(data.TensorDataset(X_train), batch_size=batch_size, shuffle=True)
    class Learner(pl.LightningModule):
        def __init__(self, model):#, context_model:nn.Module):
            super().__init__()
            self.model = model
            self.context_dim = condition_train.shape[-1]
            self.iters = 0

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            self.iters += 1
            x = batch[0]
            context = x[:,-self.context_dim:]

            theta = x[:,:-self.context_dim]
            with nvtx.range("Forward_pass"):

                logprob = self.model.log_prob(theta, condition = context) 
                loss = -torch.mean(logprob)
            return {'loss': loss}

        def configure_optimizers(self):
            return opt(self.model.parameters(), lr=lr, weight_decay=lambda_l2)

        def train_dataloader(self):
            return trainloader

    learn = Learner(flow)
    trainer = pl.Trainer(max_epochs=num_epochs)
    trainer.fit(learn)
    return learn.model

def train_hmc(flow, theta_train, condition_train, prior = False, svi_model=False, **kwargs):
    '''
    Train the Bayesian MAF by sampling the log_likelihood using HMC

    ----------
    Parameters
    ----------

    flow                           :: BayesianMAF
                                      Flow model to train

    theta_train                    :: torch.tensor (nbatch, theta_dim)
                                      theta samples from the training dataset

    condition_train                :: torch.tensor (nbatch, condition_dim)
                                      lambda samples from the training dataset

    prior                          :: bool
                                      whether or not to sample from the prior or the posterior
                                      default: False
    -------
    Returns
    -------

    samples                        :: dict
                                      posterior (or prior) samples of the flow parameters

    '''
    if not prior:
        kernel = NUTS((flow.model if not svi_model else flow.svi_model), max_tree_depth=kwargs["max_tree_depth"], step_size=kwargs["step_size"], transforms = flow.param_transforms(), full_mass = False)
    else:
        kernel = HMC(flow.prior_model, step_size=kwargs["step_size"], num_steps = 1, adapt_step_size = False, transforms = flow.param_transforms())

    mcmc = MCMC(kernel=kernel, num_samples=kwargs["num_samples"], warmup_steps=kwargs["num_warmup"])
    # Run MCMC to sample the posterior
    print("Running HMC...")
    torch.cuda.empty_cache()
    if not prior:
        mcmc.run(theta_train, condition_train)
    else:
        mcmc.run(guide = False, set_param_bounds = False)
    samples = mcmc.get_samples()
    print("HMC complete!")
    return samples

def train_svi(flow, theta_train, condition_train,  lr = 0.01, epochs = 5000, prior_dist = 'Normal', clip_val = 1.0, **kwargs):
    optimizer = poptim.Adam({'lr':lr})
    # setup the inference algorithm
    svi = SVI(flow.model, flow.guide, optimizer, loss=Trace_ELBO())
    min_loss = np.inf
    max_loss = -np.inf
    losses = []
    smoothing_window = 50
    for epoch in tqdm.tqdm(range(epochs)):
        loss = svi.step(theta_train, condition_train)#, prior_dist = prior_dist)#, retain_graph=True)
        if clip_val is not None:
            param_store = pyro.get_param_store()
            for param in param_store.values():
                nn.utils.clip_grad_norm_(param, clip_val)

        losses.append(loss)
        avg_loss = np.mean(losses[-smoothing_window:])
        if min_loss > avg_loss:
            min_loss = avg_loss
        if max_loss < avg_loss:
            max_loss = avg_loss
        if (epoch+1) % 100 == 0:
            print(epoch, np.log(avg_loss), np.log(min_loss), np.log(max_loss))
    store = pyro.get_param_store()
    svi_params = {}
    svi_params["scale_mu_q"] = store.get_param("scale_mu_q")
    svi_params["scale_sigma_q"] = store.get_param("scale_sigma_q")
    for i,t in enumerate(flow.flow_dist.transforms):
        for name, param in t.named_parameters():
            svi_params[f"flow_{i}_{name}_mean_q"] =  store.get_param(f"flow_{i}_{name}_mean_q")#.item()
    
    return svi_params

def train_importance(flow, theta_train, condition_train, num_samples, svi_params = None):
    if svi_params is None:
        kernel = Importance(flow.model, guide = None, num_samples = num_samples)
    else:
        kernel = Importance(flow.model, guide = partial(flow.guide, params = svi_params), num_samples = num_samples)
        
    posterior = kernel.run(theta_train, condition = condition_train)
    posterior_samples = {"scale": set_device([])}
    for j in range(num_samples):
        trace = posterior()
        posterior_samples["scale"] = trace.nodes["scale"]["value"].item() if j == 0 else trace.nodes
        for i,t in enumerate(flow.flow_dist.transforms):
            for n,_ in t.named_parameters():
                name = f"flow_{i}_{n}"
                if j == 0:
                    posterior_samples[name] = torch.unsqueze(trace.nodes[name]["value"].item(), 0)
                else:
                    posterior_samples[name] = torch.cat([posterior_samples[name], torch.unsqueze(trace.nodes["name"]["value"], 0)], dim = 0)
    
    ess = posterior.ESS()
    print(f"Importance Sampling finished, effective sample size: {ess} out of total {num_samples}")
    
    return posterior_samples, ess

    

def predict(flow, cond, posterior_samples, Nsamples):
    '''
    Draw an emulated population corresponding to each sample of flow parameters.

    ----------
    Parameters
    ----------

    flow                :: BayesianMAF
                           flow model that was used in training

    cond                :: torch.tensor (condition_dim)
                           lambda value corresponding to which p(theta|lambda) is needed

    posterior_samples   :: dict
                           posterior (or prior) samples of flow parameters

    Nsamples            :: int
                           number of theta samples to draw for each posterior sample

    -------
    Returns
    -------

    theta_samples_all   :: numpy.ndarray (N_posterior_samples, Nsamples, theta_dim)
                           samples from the emulated population corresponding to each posterior draw


    '''
    theta_samples_all = []
    for sample_idx in tqdm.tqdm(range(len(posterior_samples["flow_0_nn.layers.0.weight"]))):
        # Update flow parameters with posterior samples
        set_params(flow, posterior_samples, sample_idx = sample_idx)

        # Condition the flow and sample theta
        theta_samples = flow.sample(cond,[Nsamples])
        theta_samples_all.append(theta_samples.cpu().detach().numpy())

    return np.array(theta_samples_all)
