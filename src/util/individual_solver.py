# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 19:13:47 2023

@author: XRF
"""
import numpy as np
from functools import partial


def hinge_vec_fun(sigmas,rho,ms,zs):
    f = sigmas * np.maximum(np.sign(zs + 1),0) + rho * (zs - ms)
    return f

def vec_bisect_method(fun,sigmas,rho,ms,max_iter = 50, tol = 1e-5):
    lb = ms - sigmas / rho - 1 # vec_root(lb) < 0
    ub = ms + 1 # vec_root(ub) > 0
    mid = (lb + ub)/2
    flag = True
    i = 0
    while flag and i < max_iter:
        mid_fvalue = fun(mid)
        if np.sum(np.abs(mid_fvalue)) < tol:
            flag = False
            return mid
        else:
            index1 = mid_fvalue > 0
            index2 = mid_fvalue <= 0

            new_lb = np.zeros(shape = lb.shape)
            new_ub = np.zeros(shape = ub.shape)

            new_lb[index1] = lb[index1]
            new_ub[index1] = mid[index1]

            new_lb[index2] = mid[index2]
            new_ub[index2] = ub[index2]
            lb = new_lb
            ub = new_ub
            mid = (lb + ub)/2
            i += 1
    return mid

def safe_1divexp(x):
    # exp(x)/(1+exp(x))
    res = np.zeros(shape=x.shape)
    res[x > 0] = 1/(1+np.exp(-x[x > 0]))
    res[x <= 0] = np.exp(x[x <= 0])/(1+np.exp(x[x <= 0]))
    return res


def log1exp(vec):
    # log(1+exp(vec))
    res = np.zeros(shape=vec.shape)
    res[vec <= 0] = np.log(1 + np.exp(vec[vec <= 0]))
    res[vec > 0] = vec[vec > 0] + np.log(1 + np.exp(-vec[vec > 0]))
    return res


def logistic_fun(sigma,rho,m,z):
    f = (sigma * log1exp(z)).sum() + rho / 2 * (z-m).T @ (z-m)
    return f

def logistic_vec_fun(sigma,rho,m,z):
    f = sigma * log1exp(z) + rho / 2 * (z-m) ** 2
    return f

def log_fun_grad(sigma,rho,m,z):
    f = sigma * safe_1divexp(z) + rho * (z - m)  # noqa:E501
    return f

def safe_expdivexp2(x):
    res = np.zeros(shape=x.shape)
    res[x <= 0] = np.exp(x[x <= 0])/(1+np.exp(x[x <= 0])) ** 2
    res[x > 0] = np.exp(-x[x > 0])/(1+np.exp(-x[x > 0])) ** 2
    return res

def inv_hessian_vec(sigma, rho, z):
    df = 1 / (sigma * safe_expdivexp2(z) + rho)  # noqa: E501
    return df

def logistic_fun_num(sigma,rho,m,z):
    if z > 0:
        temp = z + np.log(1 + np.exp(-z))
    else:
        temp = np.log(1 + np.exp(z))
    f = sigma * temp + rho / 2 * (z-m)
    return f

def newton_method(rt, inv_rt_grad, fun, x0, sigma, rho, m, tol=1e-6, maxiter=50):
    x = x0
    for i in range(maxiter):
        rtx = rt(x)
        invg = inv_rt_grad(x)
        delta = -rtx * invg
        alpha = 1
        tempx = x + alpha * delta
        pre_funv = fun(x)
        while fun(tempx) > pre_funv + 0.0001 * alpha * rtx.reshape(-1).T @ delta:
            alpha = alpha * 0.7
            tempx = x + alpha * delta
        x = tempx
        if np.linalg.norm(delta) < tol:
            return x
        if np.linalg.norm(delta) > 1e5:
            break
    if np.linalg.norm(delta) > 1e5:
        x = vec_bisect_method(fun, sigma, rho, m)
    return x
    
    
def individual_solver(loss, sigma_array, rho, m_array):
    if loss == "binary_cross_entropy":
        rf = partial(log_fun_grad, sigma_array, rho, m_array)
        jacrf = partial(inv_hessian_vec, sigma_array, rho)
        fun = partial(logistic_fun, sigma_array, rho, m_array)
        opt_array = newton_method(rf, jacrf, fun, m_array,sigma_array, rho, m_array)
    elif loss == "hinge":
        m_array = m_array.reshape(-1)
        sigma_array = sigma_array.reshape(-1)
        fun = partial(hinge_vec_fun, sigma_array, rho, m_array)
        opt_array = vec_bisect_method(fun, sigma_array, rho, m_array)
        opt_array = np.array(opt_array)
    elif loss == "multinomial_cross_entropy":
        pass
    else:
        raise ValueError(
            f"Unrecognized loss '{loss}'! Options: ['binary_cross_entropy', 'multinomial_cross_entropy','hinge']"
        )
    return opt_array