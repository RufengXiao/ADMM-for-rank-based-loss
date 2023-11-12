# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 11:57:13 2023

@author: XRF
"""
import numpy as np
from functools import partial
from scipy.optimize import minimize

def wl1_fun_smooth(z, b, rho, D, reg, t, w):
    w = w.reshape(-1,1)
    temp = D @ w - b
    res1 = 0.5 * rho * np.sum(np.square(temp))
    w1 = w[np.abs(w)<=t]
    w2 = w[np.abs(w)>t]
    res2 = 0.5 * 0.5 * reg * np.sum(np.square(w1)) / t
    res2 += 0.5 * reg * np.sum(np.abs(w2)-0.5*t)
    return res1 + res2


def wl1_fun_smooth_gradient(z, b, rho, DTD, DTb, reg, t, w):
    w = w.reshape(-1,1)
    g1 = rho * (DTD @ w - DTb)
    g2 = np.zeros(shape=w.shape)
    g2[np.abs(w)<=t] = 0.5 * reg * w[np.abs(w)<=t] / t
    g2[np.abs(w)>t] = 0.5 * reg * np.sign(w[np.abs(w)>t])
    return g1 + g2


def wl2_fun(z, lagrangian, rho, D, reg, w):
    w = w.reshape(-1,1)
    b = z + lagrangian / rho
    temp = D @ w - b
    res1 = 0.5 * rho * (np.linalg.norm(temp, ord=2) ** 2)
    res2 = 0.5 * reg * np.sum(w * w)
    return res1 + res2


def wl2_fun_gradient(z, lagrangian, rho, D, reg, DTD, w):
    w = w.reshape(-1,1)
    b = z + lagrangian / rho
    g1 = rho * (DTD @ w - D.T @ b)
    g2 = reg * w
    return g1 + g2


def w_solver(w_flag, w0, z, lagrangian, rho, DTD, D, reg, t=None):
    if w_flag == 2:
        f =  partial(wl2_fun, z, lagrangian, rho, D, reg)
        g = partial(wl2_fun_gradient, z, lagrangian, rho, D, reg, DTD)
        res = minimize(f, w0.reshape(-1), jac=g, method="L-BFGS-B", options={'disp': False, 'maxiter': 1000})
        w = res.x.reshape(-1,1)
    elif w_flag == 1:
        if t is None:
            raise ValueError("The smoothess parameter is not given.")
        b = z + lagrangian / rho
        DTb = D.T @ b
        f = partial(wl1_fun_smooth, z, b, rho, D, reg, t)
        g = partial(wl1_fun_smooth_gradient, z, b, rho, DTD, DTb, reg, t)
        res = minimize(f, w0.reshape(-1), jac=g, method="L-BFGS-B", options={'disp': False, 'maxiter': 1000})
        w = res.x.reshape(-1,1)    
    else:
        raise ValueError("w_flag can only be 1 or 2.")
    return w