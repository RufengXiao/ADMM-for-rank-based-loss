# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 08:59:28 2023

@author: XRF
"""
import torch
import torch.nn.functional as F
import math

def binary_cross_entropy_loss(w, X, y):
    y[y==-1] = 0
    logits = torch.matmul(X, w)
    return F.binary_cross_entropy_with_logits(
        logits, y.double(), reduction="none"
    )

def multinomial_cross_entropy_loss(w, X, y, n_class):
    W = w.view(-1, n_class)
    logits = torch.matmul(X, W)
    return F.cross_entropy(logits, y, reduction="none")

def hinge_loss(w, X, y):
    return torch.maximum(1 - y.double() * torch.matmul(X, w), torch.zeros(y.shape,dtype=torch.float64))


def get_loss(name, n_class=None):
    if name == "binary_cross_entropy":
        return binary_cross_entropy_loss
    elif name == "multinomial_cross_entropy":
        return lambda w, X, y: multinomial_cross_entropy_loss(w, X, y, n_class)
    elif name == "hinge":
        return hinge_loss
    else:
        raise ValueError(
            f"Unrecognized loss '{name}'! Options: ['binary_cross_entropy', 'multinomial_cross_entropy', 'hinge']"
        )

class rankbasedObjective:
    def __init__(
        self, X, y, weight_function="erm", loss="binary_cross_entropy", l2_reg=None,l1_reg=None, B=None, n_class=None, args=None
    ):
        self.X = X
        self.y = y
        self.n, self.d = X.shape
        weight_function = get_weights(weight_function, args)
        if isinstance(weight_function, tuple):
            self.weight_function, self.weight_function2 = weight_function
            self.alphas = self.weight_function(self.n).reshape(-1,1)
            self.betas = self.weight_function2(self.n).reshape(-1,1)
        else:
            self.weight_function = weight_function
            self.alphas = self.weight_function(self.n).reshape(-1,1)
            self.betas = self.alphas
        self.loss = get_loss(loss, n_class=n_class)
        if B is not None:
            if loss != "binary_cross_entropy":
                raise ValueError("erhm only can be with the binary_cross_entropy.")
            self.B = torch.tensor(B)
            if self.B > 0:
                self.lossB = self.B + torch.log(1 + torch.exp(-self.B))
            else:
                self.lossB = torch.log(1 + torch.exp(self.B))
        else:
            self.B = None
            self.lossB = None
        self.n_class = n_class
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        
    def get_arrogate_loss(self, w, include_reg=True):
        with torch.no_grad():
            losses = self.loss(w, self.X, self.y).reshape(-1)
            sort_losses = torch.sort(losses, stable=False)[0]
            alphas = self.alphas.reshape(-1)
            betas = self.alphas.reshape(-1)
            if self.lossB is not None:
                risk = torch.dot(alphas[sort_losses<=self.lossB], sort_losses[sort_losses<=self.lossB])
                risk += torch.dot(betas[sort_losses>self.lossB], sort_losses[sort_losses>self.lossB])
            else:
                risk = torch.dot(alphas, sort_losses)
            risk = risk.item()
            if self.l2_reg and include_reg:
                risk += 0.5 * self.l2_reg * torch.sum(w ** 2).item()
            if self.l1_reg and include_reg:
                risk += 0.5 * self.l1_reg * torch.norm(w,p=1,dim=0).item()
            return risk

    def get_indiv_loss(self, w, with_grad=False):
        if with_grad:
            return self.loss(w, self.X, self.y)
        else:
            with torch.no_grad():
                return self.loss(w, self.X, self.y)


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
    if k <= m:
        raise ValueError("need args[0] > args[1]!")
    weights = torch.zeros(n, dtype=torch.float64)
    weights[m+1:k] = 1/(k-m)
    weights[k+1] = 1 - (k-m-1)/(k-m)
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

def get_weights(name, args=None):
    if name == "erm":
        return get_erm_weights
    elif name == "ehrm":
        return get_cpt_weights_a, get_cpt_weights_b
    elif args is None:
        raise ValueError("args for framework is None!")
    else:
        if name == "extremile":
            return lambda n: get_extremile_weights(n, args[0])
        elif name == "superquantile":
            return lambda n: get_superquantile_weights(n, args[0])
        elif name == "esrm":
            return lambda n: get_esrm_weights(n, args[0])
        elif name == "aorr":
            return lambda n: get_aorr_weights(n, args[0], args[1])
        elif name == "aorr_dc":
            return lambda n: get_aorr_dc_weights(n, args[0], args[1])
        else:
            raise ValueError(
                f"Unrecognized framework '{name}'! Options: ['erm','extremile','superquantile','esrm','aorr','aorr_dc','ehrm']"
            )