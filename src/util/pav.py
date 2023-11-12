# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:05:23 2023

@author: XRF YYF
"""

import numpy as np
from src.util.individual_solver import individual_solver
from src.util.individual_solver import logistic_vec_fun
import time

class Block(object):
    def __init__(self, index_array: list, sigma_array: list, m_array: list, sigma_param, m_param):
        self.index_array = index_array
        self.sigma_array = sigma_array
        self.m_array = m_array
        self.sigma_param = sigma_param  
        self.m_param = m_param
        

def merge_block(pool1: Block, pool2: Block):
    new_index_array = pool1.index_array + pool2.index_array
    new_sigma_array = pool1.sigma_array + pool2.sigma_array
    new_m_array = pool1.m_array + pool2.m_array
    new_sigma_param = pool1.sigma_param + pool2.sigma_param
    new_m_param = pool1.m_param + pool2.m_param

    return Block(new_index_array,new_sigma_array,new_m_array,new_sigma_param,new_m_param)

class Block_cpt(object):
    def __init__(self, index_array: list, sigma_array1: list,sigma_array2: list, m_array: list, sigma_param1, sigma_param2, m_param):
        self.index_array = index_array
        self.sigma_array1 = sigma_array1
        self.sigma_array2 = sigma_array2
        self.m_array = m_array
        self.sigma_param1 = sigma_param1
        self.sigma_param2 = sigma_param2
        self.m_param = m_param
        

def merge_block_cpt(pool1: Block_cpt, pool2: Block_cpt):
    new_index_array = pool1.index_array + pool2.index_array
    new_sigma_array1 = pool1.sigma_array1 + pool2.sigma_array1
    new_sigma_array2 = pool1.sigma_array2 + pool2.sigma_array2
    new_m_array = pool1.m_array + pool2.m_array
    new_sigma_param1 = pool1.sigma_param1 + pool2.sigma_param1
    new_sigma_param2 = pool1.sigma_param2 + pool2.sigma_param2
    new_m_param = pool1.m_param + pool2.m_param

    return Block_cpt(new_index_array,new_sigma_array1,new_sigma_array2,new_m_array,new_sigma_param1,new_sigma_param2,new_m_param)

    
class PAV_solver(object):
    def __init__(self, sigma_array: np.ndarray, m_array: np.ndarray, rho, loss="binary_cross_entropy", B=None, sigma_array2=None): # noqa: E501
        self.pools = [None for i in range(sigma_array.shape[0])]
        self.rho = rho
        self.loss = loss
        self.time = time.time()
        self.B = B
        self.n = sigma_array.shape[0]
        if B is None:
            opt_array = individual_solver(loss, sigma_array, rho, m_array)
            if opt_array is None:
                raise ValueError("The individual solver has some problems.")
            self.x = opt_array
            for i in range(sigma_array.shape[0]):
                self.pools[i] = Block([i], [sigma_array[i]], [m_array[i]], sigma_array[i], m_array[i])
        else:
            if loss != "binary_cross_entropy":
                raise ValueError("ehrm only can be with binary_cross_entropy.")
            opt_array1 = individual_solver(loss, sigma_array, rho, m_array)
            opt_array1[opt_array1 > B] = B
            opt_array2 = individual_solver(loss, sigma_array2, rho, m_array)   
            opt_array2[opt_array2 <= B] = B
            if opt_array1 is None or opt_array2 is None:
                raise ValueError("The individual solver has some problems.")
                
            self.x = np.zeros(shape =(opt_array1.shape[0],))
        
            fval1 = logistic_vec_fun(opt_array1, sigma_array, rho, m_array)
            fval2 = logistic_vec_fun(opt_array2, sigma_array2, rho, m_array)
            
            self.x[fval1 <=fval2] = opt_array1[fval1 <=fval2]
            self.x[fval1 > fval2] = opt_array2[fval1 > fval2]
        
        
            for i in range(sigma_array.shape[0]):
                self.pools[i] = Block_cpt([i], [sigma_array[i]],[sigma_array2[i]], [m_array[i]],
                                          sigma_array[i], sigma_array2[i],m_array[i])
                
        
    def get_opt(self, maxiter=10000):
        flag = True
        count = 0
        while flag:
            count += 1
            is_violate = False
            i = 0
            new_pools = []
            violate_list = [] # need to re-calculate
            violate_index = 0
            new_x = []
            while i < len(self.x) - 1:
                if self.x[i] <= self.x[i+1]:
                    new_pools.append(self.pools[i])
                    new_x.append(self.x[i])
                    violate_index += 1
                    i += 1
                else:
                    is_violate = True
                    violate_list.append(violate_index)
                    cur_block = self.pools[i]
                    while i < len(self.x) - 1 and self.x[i] > self.x[i+1]:
                        if self.B is None:
                            cur_block = merge_block(cur_block, self.pools[i+1])
                        else:
                            cur_block = merge_block_cpt(cur_block, self.pools[i+1])
                        i+=1
                    i+=1
                    new_pools.append(cur_block)
                    new_x.append(0)
                    violate_index += 1
            if len(self.x) >=2 and self.x[-1] >= self.x[-2]:
                new_pools.append(self.pools[-1])
                new_x.append(self.x[-1])
            if not is_violate or count >= maxiter:
                flag = False
            else:
                self.pools = new_pools
                ms = []
                if self.B is None:
                    sigmas = []
                    for i in violate_list:
                        index_len=len(new_pools[i].index_array)
                        sigmas.append(new_pools[i].sigma_param/index_len)
                        ms.append(new_pools[i].m_param/index_len)
                    m_array = np.array(ms).reshape(-1,1)         
                    sigma_array = np.array(sigmas).reshape(-1,1)
                    opt_array = individual_solver(self.loss, sigma_array, self.rho, m_array)
                    if opt_array is None:
                        raise ValueError("The individual solver has some problems.")
                    new_x = np.array(new_x,dtype=opt_array.dtype).reshape(-1)
                    violate_list = np.array(violate_list)
                    new_x[violate_list] = opt_array.reshape(-1)
                    self.x = new_x
                else:
                    sigmas1 = []
                    sigmas2 = []
                    for i in violate_list:
                        index_len=len(new_pools[i].index_array)
                        sigmas1.append(new_pools[i].sigma_param1/index_len)
                        sigmas2.append(new_pools[i].sigma_param2/index_len)
                        ms.append(new_pools[i].m_param/index_len)
                    m_array = np.array(ms).reshape(-1,1)
                    sigma_array1 = np.array(sigmas1).reshape(-1,1)
                    sigma_array2 = np.array(sigmas2).reshape(-1,1)
                    opt_array1 = individual_solver(self.loss, sigma_array1, self.rho, m_array)
                    opt_array1[opt_array1 > self.B] = self.B
                    opt_array2 = individual_solver(self.loss, sigma_array2, self.rho, m_array)
                    opt_array2[opt_array2 <= self.B] = self.B
                    fval1 = logistic_vec_fun(opt_array1, sigma_array1, self.rho, m_array)
                    fval2 = logistic_vec_fun(opt_array2, sigma_array2, self.rho, m_array)
                    if opt_array1 is None or opt_array2 is None:
                        raise ValueError("The individual solver has some problems.")
                    new_x = np.array(new_x,dtype=opt_array1.dtype).reshape(-1)
                    res_x = np.zeros(shape = opt_array1.shape)
                    res_x[fval1 <= fval2] = opt_array1[fval1 <= fval2]
                    res_x[fval1 > fval2] = opt_array2[fval1 > fval2]
                    violate_list = np.array(violate_list)
                    new_x[violate_list] = res_x.reshape(-1)
                    self.x = new_x
        res = np.zeros(self.n)
        for i in range(len(self.x)):
            res[self.pools[i].index_array] = self.x[i]
        self.time=time.time()-self.time
        # print("PAV iter: ", count)
        return res
                