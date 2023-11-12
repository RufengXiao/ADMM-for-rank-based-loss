import torch
import numpy as np
import time

from existing_methods.lerm_main.src.optim.objective import ORMObjective, get_erm_weights, get_extremile_weights, get_superquantile_weights, get_esrm_weights, get_aorr_weights,get_cpt_weights_a,get_cpt_weights_b, get_aorr_dc_weights
from existing_methods.lerm_main.src.optim.algorithms import LSVRG


def LSVRGmethod(X,y,weight_function,loss,l2_reg=None,l1_reg=None,lossB=None,
                max_iter = 20, lr = 0.01,train_loss=None, test_loss=None, uniform=None, verbose=True, args=None):
    y = y.copy()
    X = torch.tensor(X).double()
    if loss == 'logistic':
        y[y == -1] = 0
    y = torch.tensor(y.reshape(-1)).double()

    if weight_function == 'ehrm':
        weight_function = lambda n: get_cpt_weights_a(n)
        weight_function2 = lambda n: get_cpt_weights_b(n)
        train_objective = ORMObjective(
            X, 
            y, 
            weight_function=weight_function,
            weight_function2=weight_function2,
            lossB=lossB,
            loss=loss,
            l2_reg=l2_reg,
            l1_reg=l1_reg
        )
    else:
        if weight_function == 'erm':
            weight_function = lambda n: get_erm_weights(n)
        elif args is None:
            raise ValueError("args for framework is None")
        else:
            if weight_function == 'superquantile':
                q = args[0]
                weight_function = lambda n: get_superquantile_weights(n, q)
            elif weight_function == 'extremile':
                r = args[0]
                weight_function = lambda n: get_extremile_weights(n, r)
            elif weight_function == 'esrm':
                rho = args[0]
                weight_function = lambda n:get_esrm_weights(n, rho)
            elif weight_function == 'aorr':
                qlow, qup = args[0], args[1]
                weight_function = lambda n:get_aorr_weights(n, qlow, qup)
            elif weight_function == "aorr_dc":
                qlow, qup = args[0], args[1]
                weight_function = lambda n:get_aorr_dc_weights(n, qlow, qup)
            else:
                raise ValueError(
                    f"weight_function '{weight_function}' is not supported! Options: ['erm','extremile', \
                    'superquantile','esrm','aorr','aorr_dc','ehrm']")

        train_objective = ORMObjective(
            X, 
            y, 
            weight_function=weight_function, 
            loss=loss, 
            l2_reg=l2_reg,
            l1_reg=l1_reg
        )
    
    # lr = 0.01
    if lr == 1:
        lr = 1/X.shape[0]
    elif lr == 2:
        lr = 1/(X.shape[0] * X.shape[1])
    
    optimizer = LSVRG(train_objective, lr=lr, uniform=uniform, epoch_len=100)

    if test_loss is not None:
        train_losses = [train_loss(optimizer.weights.detach().reshape(-1,1))]
    test_losses = [test_loss(optimizer.weights.detach().reshape(-1,1))]
    t_array = [0]
    t_start = time.time()

    for iter in range(max_iter):
        optimizer.start_epoch()
        for _ in range(optimizer.epoch_len):
            optimizer.step()
        optimizer.end_epoch()
        if test_loss is not None:
            train_losses.append(train_loss(optimizer.weights.detach().reshape(-1,1)))
        test_losses.append(test_loss(optimizer.weights.detach().reshape(-1,1)))
        t_array.append(time.time()-t_start)
        if verbose:
            if iter % 10 == 0:
                print("iter:", iter, "train loss:", train_losses[-1], "test loss:", test_losses[-1], "time:", t_array[-1])
       
    
    if train_loss is not None:
        return optimizer.weights.detach().reshape(-1,1).numpy(), train_losses, test_losses, t_array
    else:
        return optimizer.weights.detach().reshape(-1,1).numpy()
