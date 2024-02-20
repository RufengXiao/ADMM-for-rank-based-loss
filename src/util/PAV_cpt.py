#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 15:21:43 2022

@author: yanyifan xrf
"""

import numpy as np
from functools import partial
# import time
from scipy import optimize
from multiprocessing import Pool
# import concurrent.futures


def safe_1divexp_num(x):
    # exp(x)/(1+exp(x))
    if x > 0:
        return 1/(1+np.exp(-x))
    else:
        return np.exp(x)/(1+np.exp(x))


def safe_1divexp_vector(x):
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


def func_value(sigma, rho, m, z):
    f = np.sum(sigma * log1exp(z)) + rho / 2 * (z-m).T @ (z-m)
    return f


# the gradient of sigma * logloss(z) + rho/2 *(z-m)^2
def rootfun_vec(sigma, rho, m, z):
    # f = sigma * np.exp(z) / (1 + np.exp(z)) + rho * (z - m)
    f = sigma * safe_1divexp_vector(z) + rho * (z - m)  # noqa:E501
    return f.reshape(-1, )


# the gradient of sigma * logloss(z) + rho/2 *(z-m)^2
def rootfun_num(sigma, rho, m, z):
    # f = sigma * np.exp(z) / (1 + np.exp(z)) + rho * (z - m)
    f = sigma * safe_1divexp_num(z) + rho * (z - m)
    return f


def inv_rootfun_grad(sigma, rho, z):
    # df = sigma * np.exp(z) / (1+np.exp(z))**2 + rho
    df = 1 / (sigma * safe_1divexp_vector(z) / (1+np.exp(z)) + rho)  # noqa: E501
    return df


def rootfun_grad_num(sigma, rho, z):
    # df = sigma * np.exp(z) / (1+np.exp(z))**2 + rho
    df = sigma * safe_1divexp_vector(z) / (1+np.exp(z)) + rho  # noqa: E501
    return df


def newton_system(rt, inv_rt_grad, fun, x0, sigma, rho,tol=1e-4, maxiter=50):
    x = np.asarray(x0, dtype=float)
    for i in range(maxiter):
        rtx = rt(x)
        invg = inv_rt_grad(x)
        delta = -rtx * invg
        alpha = 1
        tempx = x + alpha * delta
        while fun(tempx) > fun(x) + 0.0001 * alpha * rtx.reshape(-1).T @ delta:
            alpha = alpha * 0.5
            tempx = x + alpha * delta
        x = tempx
        # print(np.linalg.norm(delta))
        if np.linalg.norm(delta) < tol:
            return x
        if np.linalg.norm(delta) > 1e10:
            break
    if np.linalg.norm(delta) > 1e10:
        for i in range(x.shape[0]):
            x[i] = get_init([sigma[i],x0[i],rho])
        return x
    return x
    raise ValueError("maxiter reach but tol not reach!")


def get_init(param):
    sigma = param[0]
    m = param[1]
    rho = param[2]
    rf = partial(rootfun_num, sigma, rho, m)
    # rf_grad = partial(rootfun_grad, mean_sigma, self.rho, mean_m)
    # result = optimize.newton(rf,mean_m,fprime = rf_grad,maxiter = 1000)
    result = optimize.bisect(rf, m-sigma/rho, m, maxiter=100)  # noqa: E501
    return result


def solve_rt(x):
    rf = partial(rootfun_num, x[0], x[1], x[2])
    # rf_grad = partial(rootfun_grad_num, x[0], x[1])
    # result = optimize.newton(rf, x[2], fprime=rf_grad, maxiter=1000)
    result = optimize.bisect(rf, x[2]-x[0]/x[1], x[2], maxiter=100)  # noqa: E501
    return result

class Block_cpt(object):
    def __init__(self, index_array: list, sigma_array1: list,sigma_array2: list, m_array: list):
        self.index_array = index_array
        self.sigma_array1 = sigma_array1
        self.sigma_array2 = sigma_array2
        self.m_array = m_array
        
    def param(self):
        return np.mean(self.sigma_array1),np.mean(self.sigma_array2),np.mean(self.m_array)

def merge_block_cpt(pool1: Block_cpt, pool2: Block_cpt):
    new_index_array = pool1.index_array + pool2.index_array
    new_sigma_array1 = pool1.sigma_array1 + pool2.sigma_array1
    new_sigma_array2 = pool1.sigma_array2 + pool2.sigma_array2
    new_m_array = pool1.m_array + pool2.m_array

    return Block_cpt(new_index_array,new_sigma_array1,new_sigma_array2,new_m_array)



class PAV_Pool(object):
    def __init__(self, index_array: list, sigma_array: list, m_array: list, rho, opt=None):  # noqa: E501
        self.index_array = index_array
        self.sigma_array = sigma_array
        self.m_array = m_array
        self.rho = rho
        # t1 = time.time()
        if opt:
            self.xopt = opt
        else:
            self.xopt = self.get_opt()
        # t2 = time.time()
        # print('pav-sub',t2-t1)

    def get_opt(self):
        mean_sigma = np.mean(self.sigma_array)
        mean_m = np.mean(self.m_array)

        rf = partial(rootfun_num, mean_sigma, self.rho, mean_m)
        # rf_grad = partial(rootfun_grad, mean_sigma, self.rho, mean_m)
        # result = optimize.newton(rf,mean_m,fprime = rf_grad,maxiter = 1000)
        result = optimize.bisect(rf, mean_m-mean_sigma/self.rho, mean_m, maxiter=100)  # noqa: E501

        return result


def merge(pool1: PAV_Pool, pool2: PAV_Pool):
    new_index_array = pool1.index_array + pool2.index_array
    new_sigma_array = pool1.sigma_array + pool2.sigma_array
    new_m_array = pool1.m_array + pool2.m_array

    return PAV_Pool(new_index_array, new_sigma_array, new_m_array, pool1.rho)


class PAV_solver_CPT(object):
    def __init__(self, sigma_array1: np.ndarray,sigma_array2: np.ndarray,B, m_array: np.ndarray, rho, multi=False):  # noqa: E501
        self.pools = [None for i in range(sigma_array1.shape[0])]

        # if multi:
        #     init_params = [() for i in range(sigma_array.shape[0])]
        #     for i in range(sigma_array.shape[0]):
        #         init_params[i] = (sigma_array[i], m_array[i], rho)
        #     with Pool(processes=4) as pool:
        #         res = pool.map(get_init, init_params)
        #     for i in range(sigma_array.shape[0]):
        #         self.pools[i] = PAV_Pool([i], [sigma_array[i]], [m_array[i]], rho, res[i])  # noqa: E501
        # else:
            # one by one
            # t1 = time.time()
            # for i in range(sigma_array.shape[0]):
            #     self.pools[i] = PAV_Pool([i], [sigma_array[i]], [m_array[i]],  rho, )  # noqa: E501
            # print("one and one time: ", time.time() - t1)
            # t2 = time.time()

            # multi
            # list1 = sigma_array.reshape(-1)
            # list2 = rho * np.ones(shape=list1.shape)
            # list3 = m_array.reshape(-1)
            # args = [(*x, y) for x, y in zip(zip(list1, list2), list3)]
            # print("multi time: ", time.time()-t2)
            # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:  # noqa: E501
            #     opt_array = list(executor.map(solve_rt, args))
            # print("multi time: ", time.time()-t2)
            # for i in range(sigma_array.shape[0]):
            #     self.pools[i] = PAV_Pool([i], [sigma_array[i]], [m_array[i]],  rho, opt_array[i])  # noqa: E501
            # print("multi time: ", time.time()-t2)

            # vector
        self.rho = rho
        self.B = B
        rf1 = partial(rootfun_vec, sigma_array1.reshape(-1), rho, m_array.reshape(-1))  # noqa: E501
        jacrf1 = partial(inv_rootfun_grad, sigma_array1.reshape(-1), rho)
        fun1 = partial(func_value, sigma_array1.reshape(-1), rho, m_array.reshape(-1))  # noqa: E501
        # print("vector time: ", time.time()-t2)
        opt_array1 = newton_system(rf1, jacrf1, fun1, m_array.reshape(-1),sigma_array1, rho,)  # noqa: E501
        # print("vector time: ", time.time()-t2)
        opt_array1[opt_array1>B] = B
        
        rf2 = partial(rootfun_vec, sigma_array2.reshape(-1), rho, m_array.reshape(-1))  # noqa: E501
        jacrf2 = partial(inv_rootfun_grad, sigma_array2.reshape(-1), rho)
        fun2= partial(func_value, sigma_array2.reshape(-1), rho, m_array.reshape(-1))  # noqa: E501
        # print("vector time: ", time.time()-t2)
        opt_array2 = newton_system(rf2, jacrf2, fun2, m_array.reshape(-1),sigma_array2, rho,)  # noqa: E501
        opt_array2[opt_array2 <= B] = B
        
        self.x = np.zeros(shape =(sigma_array1.shape[0],))
        
        fval1 = fun1(opt_array1)
        fval2 = fun2(opt_array2)
        
        self.x[fval1 <=fval2] = opt_array1[fval1 <=fval2]
        self.x[fval1 > fval2] = opt_array2[fval1 > fval2]
        
        
        for i in range(sigma_array1.shape[0]):
            self.pools[i] = Block_cpt([i], [sigma_array1[i]],[sigma_array2[i]], [m_array[i]])  # noqa: E501
            # print("vector time: ", time.time()-t2)
        # print(res)

    def get_opt(self):
        # flag = True
        flag = True
        while flag:
            is_violate = False
            i = 0
            new_pools = []
            while i < len(self.pools) - 1:
                if self.x[i] <= self.x[i+1]:
                    new_pools.append(self.pools[i])
                else:
                    is_violate = True
                    cur_block = self.pools[i]
                    while i < len(self.pools) - 1 and self.x[i] > self.x[i+1]:
                        cur_block = merge_block_cpt(cur_block, self.pools[i+1])
                        i+=1
                    new_pools.append(cur_block)
                i+=1
            if len(self.pools) >=2 and self.x[len(self.pools) - 1] >= self.x[len(self.pools)-2]:
                new_pools.append(self.pools[-1])
            if not is_violate:
                flag = False
            else:
                self.pools = new_pools
                ms = []
                sigmas1 = []
                sigmas2 = []
                for b in self.pools:
                    s1,s2,m = b.param()
                    sigmas1.append(s1)
                    ms.append(m)
                    sigmas2.append(s2)
                m_array = np.array(ms)
                sigma_array1 = np.array(sigmas1)
                sigma_array2 = np.array(sigmas2)
                
                rf1= partial(rootfun_vec, sigma_array1.reshape(-1), self.rho, m_array.reshape(-1))  # noqa: E501
                jacrf1 = partial(inv_rootfun_grad, sigma_array1.reshape(-1), self.rho)
                fun1 = partial(func_value, sigma_array1.reshape(-1), self.rho, m_array.reshape(-1))  # noqa: E501
                # print("vector time: ", time.time()-t2)
                opt_array1 = newton_system(rf1, jacrf1, fun1, m_array.reshape(-1),sigma_array1, self.rho,)  # noqa: E501
                opt_array1[opt_array1 > self.B] = self.B
                fval1 = fun1(opt_array1)
                
                rf2 = partial(rootfun_vec, sigma_array2.reshape(-1), self.rho, m_array.reshape(-1))  # noqa: E501
                jacrf2 = partial(inv_rootfun_grad, sigma_array2.reshape(-1), self.rho)
                fun2 = partial(func_value, sigma_array2.reshape(-1), self.rho, m_array.reshape(-1))  # noqa: E501
                # print("vector time: ", time.time()-t2)
                opt_array2 = newton_system(rf2, jacrf2, fun2, m_array.reshape(-1),sigma_array2, self.rho,)  # noqa: E501
                opt_array2[opt_array2 <= self.B] = self.B
                fval2 = fun2(opt_array2)
                
                self.x = np.zeros(shape = (m_array.shape[0],))
                self.x[fval1 <= fval2] = opt_array1[fval1 <= fval2]
                self.x[fval1 > fval2] = opt_array2[fval1 > fval2]
                
        opt_res = []
        for i in range(len(self.pools)):
            opt_res += [self.x[i]] * len(self.pools[i].index_array)
        return np.array(opt_res)

        # print(opt_res0)

        # while flag:
        #     is_violate = False
        #     for i in range(len(self.pools)-1):
        #         if self.pools[i].xopt > self.pools[i+1].xopt:
        #             is_violate = True
        #             break

        #     if is_violate:
        #         new_pools = self.pools[:i] + [merge(self.pools[i], self.pools[i+1])] + self.pools[i+2:]  # noqa: E501
        #         self.pools = new_pools
        #     else:
        #         flag = False

        # opt_res = []
        # for pool in self.pools:
        #     opt_res += [pool.xopt] * len(pool.index_array)

        # return np.array(opt_res)
