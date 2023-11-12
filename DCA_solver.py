from existing_methods.SoRR.AoRR.logisticregression_DC import LogisticRegression_TOPK_DC
from existing_methods.SoRR.AoRR.hinge_DC import hinge_TOPK_DC
import numpy as np
from functools import partial
import time
import torch

def loss_function(X, label, sigma_array, reg, w):
    # num_row,num_feature = X.shape[0], X.shape[1]
    raw_product = (X @ w).reshape(-1, 1)
    z = (label * raw_product).reshape(-1)
    loss = np.where(1 - z > 0, 1 - z, 0)
    loss = np.sort(loss).reshape(-1)
    f = np.sum(loss * sigma_array)
    f += 0.5 * reg * (w.T @ w)
    return f


def DCAmethod(X,y,X_val,y_val,X_test,y_test,loss, sigma=None, reg=1e-4,
              train_loss=None, test_loss=None, kvalue=None, kvalue2=None, dataname=None):
    if dataname is None:
        raise ValueError("dataname is None!")
    if dataname == "synthetic":
        dataname = "random"
    if kvalue is None or kvalue2 is None:
        raise ValueError("kvalue or kvalue2 is None!")
    y = y.reshape(-1)
    y_val = y_val.reshape(-1)
    y_test = y_test.reshape(-1)
    if loss == "binary_cross_entropy":
        y[y == -1] = 0
        y_val[y_val == -1] = 0
        y_test[y_test == -1] = 0
        if dataname == "random":
            classifier = LogisticRegression_TOPK_DC(lr=0.01, num_iter=10, inner_iter=2000, k_value = kvalue, k2_value = kvalue2,dataname = dataname, Model_name="LogisticRegression", train_loss=train_loss)
        elif dataname == "monks-3":
            classifier = LogisticRegression_TOPK_DC(lr=0.01, num_iter=5, inner_iter=2000,k_value = kvalue, k2_value = kvalue2,dataname = dataname, Model_name="LogisticRegression",train_loss=train_loss)
        else:
            classifier = LogisticRegression_TOPK_DC(lr=0.01, num_iter=10, inner_iter=1000, k_value = kvalue, k2_value = kvalue2,dataname = dataname, Model_name="LogisticRegression",train_loss=train_loss)
    elif loss == "hinge":
        if dataname == "random":
            classifier = hinge_TOPK_DC(lr=0.01, num_iter=10, inner_iter=2000, k_value = kvalue, k2_value = kvalue2,dataname = dataname, Model_name="Hinge",train_loss=train_loss)
        else:
            if dataname == "monks-3":
                num_iter = 5
                inner_iter = 1000
            elif dataname == "australian":
                num_iter = 5
                inner_iter = 1000
            elif dataname == "phoneme":
                num_iter = 10
                inner_iter = 500
            elif dataname == "titanic":
                num_iter = 5
                inner_iter = 500
            elif dataname == "splice":
                num_iter = 10
                inner_iter = 1000
            else:
                raise ValueError(f"Unrecognized dataname '{dataname}'!")
            classifier = hinge_TOPK_DC(lr=0.01, num_iter=num_iter, inner_iter=inner_iter, k_value = kvalue, k2_value = kvalue2,dataname = dataname, Model_name="Hinge",train_loss=train_loss)
    else:
        raise ValueError(
            f"Unrecognized loss '{loss}'! Options: ['binary_cross_entropy', 'hinge']"
        )
    t_start = time.time()
    classifier.fit(X, y,X_val,y_val,X_test,y_test)
    t_total = time.time() - t_start

    predicted, dca_w = classifier.predict(X_test)
    train_loss_final = train_loss(torch.from_numpy(dca_w).double().reshape(-1,1))
    test_loss_final = test_loss(torch.from_numpy(dca_w).double().reshape(-1,1))

    return dca_w, t_total, train_loss_final, test_loss_final


    