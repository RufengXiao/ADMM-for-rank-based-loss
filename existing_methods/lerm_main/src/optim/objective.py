import torch
import torch.nn.functional as F
import math
import numpy as np


def squared_error_loss(w, X, y):
    return 0.5 * (y - torch.matmul(X, w)) ** 2


def binary_cross_entropy_loss(w, X, y):
    logits = torch.matmul(X, w)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, y.double(), reduction="none"
    )


def multinomial_cross_entropy_loss(w, X, y, n_class):
    W = w.view(-1, n_class)
    logits = torch.matmul(X, W)
    return torch.nn.functional.cross_entropy(logits, y, reduction="none")


def get_loss(name, n_class=None):
    if name == "squared_error":
        return squared_error_loss
    elif name == "binary_cross_entropy":
        return binary_cross_entropy_loss
    elif name == "multinomial_cross_entropy":
        return lambda w, X, y: multinomial_cross_entropy_loss(w, X, y, n_class)
    else:
        raise ValueError(
            f"Unrecognized loss '{name}'! Options: ['squared_error', 'binary_cross_entropy', 'multinomial_cross_entropy']"
        )

def hinge_loss(w, X, y):
    return torch.maximum(1 - y * torch.matmul(X, w), torch.zeros(y.shape,dtype=torch.float64))


class ORMObjective:
    def __init__(
        self, X, y, weight_function, weight_function2=None, lossB=None, loss="squared_error", l2_reg=None,l1_reg=None, n_class=None
    ):
        self.X = X
        self.y = y
        self.n, self.d = X.shape
        self.weight_function = weight_function
        self.weight_function2 = weight_function2
        if loss == "hinge":
            self.loss = hinge_loss
        else:
            self.loss = get_loss(loss, n_class=n_class)
        if lossB is not None:
            self.lossB = torch.tensor(lossB)
        self.n_class = n_class
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.alphas = weight_function(self.n)
        if weight_function2 is not None:
            self.betas = weight_function2(self.n)

    def get_batch_loss(self, w, include_reg=True):
        with torch.no_grad():
            losses = self.loss(w, self.X, self.y)
            risk = torch.dot(self.alphas, torch.sort(losses, stable=True)[0])
            if self.l2_reg and include_reg:
                risk += 0.5 * self.l2_reg * torch.norm(w,p=2,dim =0) / self.n
            if self.l1_reg and include_reg:
                risk += 0.5 * self.l1_reg * torch.norm(w,p=1,dim=0)  / self.n
            return risk

    def get_batch_subgrad(self, w, idx=None, include_reg=True):
        if idx is not None:
            X, y = self.X[idx], self.y[idx]
            alphas = self.weight_function(len(X))
            if self.weight_function2 is not None:
                betas = self.weight_function2(len(X))
        else:
            X, y = self.X, self.y
            alphas = self.alphas
            if self.weight_function2 is not None:
                betas = self.betas
        losses = self.loss(w, X, y)
        sort = torch.sort(losses, stable=True)[0]
        if self.weight_function2 is not None:
            risk = torch.dot(alphas[sort <= self.lossB], sort[sort <= self.lossB])
            risk += torch.dot(betas[sort > self.lossB], sort[sort > self.lossB])
        else:
            risk = torch.dot(alphas, sort)
        g = torch.autograd.grad(outputs=risk, inputs=w)[0]
        if self.l2_reg and include_reg:
            g += self.l2_reg * w.detach() / self.n
        if self.l1_reg and include_reg:
            # l1 = torch.abs(w * self.l1_reg / (2 * self.n)).sum()
            # g += torch.autograd.grad(l1, w)[0]
            res = torch.zeros(w.shape)
            res[w > 0] = 1
            res[w < 0] = -1
            res[w == 0] = torch.rand(1) * 2 - 1
            g += res * self.l1_reg / (2 * self.n)
        return g

    def get_indiv_loss(self, w, with_grad=False):
        if with_grad:
            return self.loss(w, self.X, self.y)
        else:
            with torch.no_grad():
                return self.loss(w, self.X, self.y)

    def get_indiv_grad(self, w):
        raise NotImplementedError


def get_erm_weights(n):
    return torch.ones(n, dtype=torch.float64) / n


def get_extremile_weights(n, r):
    return (
        (torch.arange(n, dtype=torch.float64) + 1) ** r
        - torch.arange(n, dtype=torch.float64) ** r
    ) / (n ** r)


def get_superquantile_weights(n, q):
    weights = torch.zeros(n, dtype=torch.float64)
    idx = math.floor(n * q)
    frac = 1 - (n - idx - 1) / (n * (1 - q))
    if frac > 1e-12:
        weights[idx] = frac
        weights[(idx + 1) :] = 1 / (n * (1 - q))
    else:
        weights[idx:] = 1 / (n - idx)
    return weights


def get_esrm_weights(n, rho):
    upper = torch.exp(rho * ((torch.arange(n, dtype=torch.float64) + 1) / n))
    lower = torch.exp(rho * (torch.arange(n, dtype=torch.float64) / n))
    return math.exp(-rho) * (upper - lower) / (1 - math.exp(-rho))


def get_aorr_weights(n, qlow, qup):
    weights = torch.zeros(n, dtype=torch.float64)
    idxlow = math.floor(n * qlow)
    idxup = math.floor(n * qup)
    frac = 1 - (idxup - idxlow - 1) / (n * (qup - qlow))
    if frac > 1e-12:
        weights[idxlow] = frac
        weights[(idxlow + 1) : idxup] = 1 / (n * (qup - qlow))
    else:
        weights[idxlow : idxup] = 1 / (idxup - idxlow)
    return weights

def get_aorr_dc_weights(n,k,m):
    weights = torch.zeros(n, dtype=torch.float64)
    weights[m+1:k+1] = 1/(k-m)
    return weights


def distort(p,gamma):
    res = p**gamma/((p**gamma + (1-p)**gamma) **(1/gamma))
    return res


def get_cpt_weights_a(n):
    an = torch.zeros(n, dtype=torch.float64)
    for i in range(n):
        an[i] = distort((i+1)/n,0.69) - distort(i/n, 0.69)
    return an


def get_cpt_weights_b(n):
    bn = torch.zeros(n, dtype=torch.float64)
    for i in range(n):
        bn[i] = distort((n-i)/n, 0.61) - distort((n-i-1)/n,0.61)
    return bn
