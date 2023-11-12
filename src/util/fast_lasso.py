'''
REFERENCES
[1] Cho Y, Kim J, Yu D. Comparative study of CUDA GPU implementations in
Python with the fast iterative shrinkage-thresholding algorithm for LASSO[J].
IEEE Access, 2022, 10: 53324-53343.
'''

import torch
import numpy as np
from sklearn.linear_model import Lasso
# from celer import Lasso
import time


def soft_thr(x, alpha):
    n = x.shape[0]
    # S = torch.maximum(torch.abs(x)-alpha, torch.zeros(n, device="cuda:0"))*torch.sign(x)  # noqa: E501
    S = torch.maximum(torch.abs(x)-alpha, torch.zeros(n, device="cpu"))*torch.sign(x)  # noqa: E501
    return S


def FISTA(beta, X, y, lam, L, eta, tol=1e-4, max_iter=5000, dtype=torch.float32):  # noqa: E501
    if (dtype == torch.float32):
        torch.set_default_tensor_type(torch.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)
    # print(torch.cuda.get_device_name(0))
    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    # n = X.shape[0]
    # p = X.shape[1]
    dbeta = torch.Tensor(beta).to(device)
    dX = torch.Tensor(X).to(device)
    dy = torch.Tensor(y).to(device)
    t = torch.ones(1, dtype=dtype, device=device)
    crit = np.zeros(max_iter)
    dbeta_p = torch.Tensor(beta).to(device)
    dbeta_prev = torch.Tensor(beta).to(device)
    L_prev = L
    for k in range(max_iter):
        dymXbp = dy-torch.matmul(dX, dbeta_p)
        drbp = torch.dot(dymXbp, dymXbp)
        dXTrbp = torch.matmul(torch.t(dX), dymXbp)
        i_k = -1
        cond = True
        while cond:
            i_k += 1
            L_cur = L_prev*(eta**i_k)
            dbstar = dbeta_p + dXTrbp/L_cur
            dbeta = soft_thr(dbstar, lam/L_cur)
            diff_beta = dbeta - dbeta_p
            RHS_1st = torch.dot(diff_beta, diff_beta)
            RHS_2nd = torch.dot(diff_beta, dXTrbp)
            RHS = L_cur * RHS_1st - 2.0 * RHS_2nd
            dymXb = dy-torch.matmul(dX, dbeta)
            LHS = torch.dot(dymXb, dymXb)-drbp
            cond = (LHS > RHS)
        L_prev = L_cur
        tnext = (1.0+torch.sqrt(1+4*t**2))/2.0
        diff_beta = dbeta-dbeta_prev
        t1 = (t-1.0)/tnext
        dbeta_p = dbeta+t1*diff_beta
        crit[k] = torch.norm(diff_beta)
        if crit[k] < tol:
            break
        t = tnext
        dbeta_prev = dbeta
    out = dbeta.to('cpu')
    return out.numpy()  # , crit, k


def func_value(w, X, y, alpha):
    n = X.shape[0]
    f = 1/(2*n)*np.linalg.norm(y - X @ w)**2 + alpha * np.linalg.norm(w, ord=1)
    return f


if __name__ == '__main__':
    np.random.seed(2017)
    n = 1500
    p = 3000
    X = np.random.randn(n, p)
    beta_tr = np.zeros(p)
    beta_tr[:int(0.05*p)] = 1.0
    y = np.dot(X, beta_tr) + np.random.randn(n)

    alpha = np.sqrt(2*np.log(p)/n**2, dtype=np.float32)
    L = np.float32(10)
    eta = np.float32(2)
    lasso_model = Lasso(alpha=alpha, tol=1e-8, fit_intercept=False, max_iter=10000, warm_start=False)  # noqa: E501
    t1 = time.time()
    lasso_model.fit(X=X, y=y)
    w = lasso_model.coef_.reshape(-1, 1)
    print("lasso time: ", time.time()-t1)
    print("lasso: ", func_value(w.reshape(-1, 1), X, y, alpha))
    t2 = time.time()
    w1 = FISTA(beta=np.zeros(p), X=X, y=y.reshape(-1,), lam=alpha*n, L=L, eta=eta, tol=1e-3, max_iter=5000, dtype=torch.float32)  # noqa: E501
    print("lasso time: ", time.time()-t2)
    print("lasso: ", func_value(w1.reshape(-1, 1), X, y, alpha))
