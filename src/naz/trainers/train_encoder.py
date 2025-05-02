import torch
from torch import nn
import torch.optim as optim

import numpy as np

def train_encoder(model, x_train, x_test, cov_args=None, loss=nn.MSELoss, opt = optim.Adam, lr=0.001, num_epochs=256, batch_frac  = 0.005, min_mse=0.05,lambda_l1=0., lambda_l2 = 0.,patience=32,min_epochs=128,clip_val=1.0,lr_decay=0.5,min_lr=None):
    #assert abs(x_train)<=1 and abs(x_test)<=1
    torch.cuda.empty_cache()
    loss_fn = loss()
    optimizer = opt(model.parameters(), lr=lr, weight_decay = lambda_l2) # Optimizer with L2 regulerization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_decay, patience=int(patience/2), verbose=True)
    best_mse = np.inf   # init to infinity
    best_weights = None
    history = []
    history_val = [ ]
    batch_size = int(len(x_train) * batch_frac)
    batch_start = torch.arange(0, len(x_train), batch_size)
    model.to(x_train.device)
    n_noimprove=0
    min_lr = lr*1e-3 if min_lr is None else min_lr
    best_epoch=0
    for epoch in range(num_epochs):
        model.train()
        shuffle_idx = torch.randperm(len(x_train))
        total_loss = 0.
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                batch_indices = shuffle_idx[start:start+batch_size]
                x_batch = x_train[batch_indices]
                # forward pass
                x_latent, x_rec = model(x_batch)
                loss = loss_fn(x_rec,x_batch)

                #Covariance loss
                if cov_args is not None:
                    cov_loss = normalized_covariance_loss(x_latent, cov_args["theta_train"][batch_indices])
                    loss= cov_args["weight"]*cov_loss+(1-cov_args["weight"])*loss

                #L1 regularization
                if lambda_l1>0.:
                    #l1_penalty
                    reg_loss = 0
                    for name, param in model.named_parameters():
                        if name.endswith('weight'):
                            reg_loss += lambda_l1 * param.abs().sum()

                    loss +=  reg_loss
                # backward pass
                total_loss+=loss
                optimizer.zero_grad()
                loss.backward()
                # clip gradients
                if clip_val is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))

        torch.cuda.empty_cache()
        # evaluate accuracy at end of each epoch
        model.eval()
        with torch.no_grad():
            y_pred, x_rec = model(x_test)
            mse =  loss_fn(x_rec,x_test)
            if cov_args is not None:
                cov_loss = normalized_covariance_loss(y_pred, cov_args["theta_test"])
                print(mse, cov_loss)
                mse = cov_args["weight"]*cov_loss +(1-cov_args["weight"])*mse


            #mse = float(mse)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(mse)
        print(f"epoch: {epoch}, validation_loss: {float(mse)}, training_loss: {float(total_loss)}, learning_rate: {float(current_lr)}")
        history.append(float(total_loss)/len(batch_start))
        history_val.append(float(mse))
        if float(mse) < best_mse:
            best_epoch = epoch
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())
            params_pred = y_pred.clone().detach()
            n_noimprove=0
        elif epoch>min_epochs:
            n_noimprove+=1
        else:
            pass

        if epoch>min_epochs and n_noimprove>patience and current_lr<min_lr:
            print(f"network converger after {epoch} eopchs")
            break
        torch.cuda.empty_cache()
    model.load_state_dict(best_weights)
    return model, history, history_val, best_mse,best_epoch

