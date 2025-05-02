
import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as data

from pyro.infer import MCMC, NUTS, HMC, SVI, Importance, Trace_ELBO
import pyro.optim as poptim

import pytorch_lightning as pl


from sklearn.model_selection import train_test_split
import numpy as np
import tqdm
import copy
from functools import partial


def get_params(flow):
    '''
    Get model parameters for a flow

    ----------
    Parameters
    ----------

    flow         :: MAF


    -------
    Returns
    -------

    params       :: list
                    list containing named  parameters of every flow transformation
    '''
    params = [ ]
    for t in flow.flow_dist.transforms:
        this_params = {}
        for name, param in t.named_parameters():
            this_params[name] = copy.deepcopy(param)
        params.append(this_params)

    return params

def set_params(flow, params, sample_idx = None):
    '''
    Set model parameters for a flow

    ----------
    Parameters
    ----------

    flow         :: MAF


    params       :: list
                    output of get_params

    sample_idx   :: int or None
                    if samples of flow parameters are

    '''
    for i,t in enumerate(flow.flow_dist.transforms):
        for name, param in t.named_parameters():
            with torch.no_grad():
                if sample_idx is None:
                    param.copy_(params[i][name])
                else:
                    param.copy_(params[f"flow_{i}_{name}"][sample_idx])

def train(flow,x, y, opt = optim.Adam, lr=0.001, num_epochs=1024, train_frac=0.7, batch_frac  = 0.005, lambda_l1=0., lambda_l2 = 0., patience=32, min_epochs=128, clip_val=1.0, lr_decay=0.5, min_lr=None):

    '''
    Train a flow using MLE: traditional training to minimize the KL divergence or maximize
    the average log likelihood using something like gradient decent. No uncertainty quantification

    ----------
    Parameters
    ----------

    flow                           ::  MAF


    x                              :: torch.tensor (nbatch, theta_dim)
                                      theta samples from the training dataset

    y                              :: torch.tensor (nbatch, condition_dim)
                                      lambda samples from the training dataset

    opt                            :: torch.optim
                                      optimizer for minimizing loss
                                      default: Adam

    lr                             :: float
                                      learning rate
                                      default: 0.001


    num_epochs                     :: int
                                      maximum number of epochs to iterate over
                                      default: 1024


    train_frac                     :: float
                                      fraction of data points to use in training
                                      default: 0.7

    batch_frac                     :: float
                                      fraction of training points in a single batch
                                      default : 0.0005


    lambda_l1                      :: float
                                      coefficient of L1 regularization
                                      default: 0 (no regularization)

    lambda_l2                      :: float
                                      coefficient of L2 regularization
                                      default: 0 (no regularization)


    patience                       :: int
                                      patience for reducing lr on a plateau
                                      default: 32


    min_epochs                     :: int
                                      minimum number of iterations before early stop can be implemented
                                      default: 128

    lr_decay                       :: float
                                      fraction used to reduce lr when on a plateau
                                      default: 0.5 (lr is halfed after patience is exhausted on plateau

    clip_val                       :: float
                                      value used to clip (normalize) the weight gradients
                                      default: 1.0

    min_lr                         :: float
                                      minimum learning rate after which training ends in early stop.
                                      default: None (1e-3*lr)


    -------
    Returns
    -------

    flow         :: MAF
                    trained MAF

    history      :: list
                    training loss evolution

    history_val  :: list
                    validation loss evolution

    best_mse     :: float
                    lowest validation loss

    best_epoch   :: int
                    epoch of lowest validation loss

    '''
    parameters = [ ]
    for t in flow.flow_dist.transforms:
        parameters.extend(list(t.parameters()))
    optimizer = opt(parameters, lr=lr, weight_decay = lambda_l2) # Optimizer with L2 regulerization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_decay, patience=int(patience/2))#, verbose=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_frac, shuffle=True)
    best_mse = np.inf   # init to infinity
    best_weights = None
    history = []
    history_val = [ ]
    batch_size = int(len(x_train) * batch_frac)
    batch_start = torch.arange(0, len(x_train), batch_size)
    flow.to(x.device)
    n_noimprove=0
    min_lr = lr*1e-3 if min_lr is None else min_lr
    best_epoch=0
    for epoch in range(num_epochs):
        flow.train()
        shuffle_idx = torch.randperm(x_train.shape[0])
        total_loss = 0.
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                batch_indices = shuffle_idx[start:start+batch_size]
                x_batch = x_train[batch_indices]
                y_batch = y_train[batch_indices]
                # forward pass
                optimizer.zero_grad()
                loss =  -flow.log_prob(x_batch, condition = y_batch).mean()

                #L1 regularization
                if lambda_l1>0.:
                    #l1_penalty
                    reg_loss = 0
                    for name, param in flow.named_parameters():
                        if name.endswith('weight'):
                            reg_loss += lambda_l1 * param.abs().sum()

                    loss +=  reg_loss
                # backward pass
                total_loss+=loss
                loss.backward()
                # clip gradients
                if clip_val is not None:
                    nn.utils.clip_grad_norm_(flow.parameters(), clip_val)
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))

        # evaluate accuracy at end of each epoch
        flow.flow_dist.clear_cache()
        flow.eval()
        with torch.no_grad():
            mse = -flow.log_prob(x_test,condition = y_test).mean()
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(mse)
        print(f"epoch: {epoch}, validation_loss: {float(mse)}, best validation_loss:{best_mse},training_loss: {float(total_loss)}, learning_rate: {float(current_lr)}, min_lr: {min_lr}, no imrovement for {n_noimprove}")
        history.append(float(total_loss)/len(batch_start))
        history_val.append(float(mse))
        if float(mse) < best_mse:
            best_epoch = epoch
            best_mse = mse
            best_weights = copy.deepcopy(get_params(flow))
            n_noimprove=0
        elif epoch>min_epochs:
            n_noimprove+=1
        else:
            pass

        if epoch>min_epochs and n_noimprove>patience and current_lr<min_lr:
            print(f"network converged after {epoch} eopchs")
            break
    set_params(flow, best_weights)
    return flow, history, history_val, best_mse,best_epoch

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
