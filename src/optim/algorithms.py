# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:40:54 2023

@author: XRF YYF
"""

import numpy as np
from src.optim.objective import rankbasedObjective
from src.util.pav import PAV_solver
from src.util.w_LBFGS import w_solver
from src.util.fast_lasso import FISTA
from sklearn.linear_model import Lasso
import time
import torch


class Optimizer:
    def __init__(self,X,y,weight_function="erm", loss="binary_cross_entropy",l2_reg=None,l1_reg=None,
                 B=None, n_class=None, args=None, w0=None, max_iter=200, tol=1e-4):
        self.objective = rankbasedObjective(torch.from_numpy(X.copy()), torch.from_numpy(y.copy()),weight_function,loss,l2_reg,l1_reg,B,n_class,args)
        self.D = -y * X
        self.DTD = self.D.T @ self.D
        
        self.num_row = X.shape[0]
        self.num_feature = X.shape[1]
        
        # regularization
        self.reg = l1_reg or l2_reg
        # lagriangian multiplier
        self.lagrangian = 0.1*self.reg/self.num_row*np.ones(shape=(self.num_row, 1))  # noqa: E501
        # intermediate variable
        self.z = 0.1*self.reg/self.num_row*np.ones(shape=(self.num_row, 1))
        # loss
        self.loss = loss
        
        # linear weights to optimize
        if w0 is not None:
            self.w = w0.reshape(-1, 1)
        else:
            self.w = 0.001*self.reg/self.num_feature/self.num_row*np.ones(shape=(self.num_feature, 1))
            
        self.tol = tol
        self.max_iter = max_iter
        
        if weight_function == 'ehrm':
            self.rho = 0.0001
        elif weight_function == 'aorr' or weight_function == 'aorr_dc':
            self.rho = 2e-7
        else:
            self.rho = 1e-5
        
        self.regI = self.reg * np.diag(np.ones(self.num_feature))
        if l1_reg is None and l2_reg is None:
            self.inv_w_matrix = np.linalg.pinv(self.DTD) @ self.D.T
        elif l1_reg is not None:
            self.w_flag = 1
        elif l2_reg is not None:
            self.w_flag = 2
        else:
            raise ValueError("More arguments: l1_reg or l2_reg not l1_reg and l2_reg!")
            
        self.B = B
        if B is not None and weight_function != "ehrm":
            raise ValueError(
                f"Unrecognized weight_function '{weight_function}'! Options: ['ehrm']"
            )
        self.w_tol = 7e-5
        self.z_maxiter = self.num_row
        self.store = False

    def start_store(self,X,y,weight_function="erm", loss="binary_cross_entropy",
                 B=None,l2_reg=None,l1_reg=None, n_class=None, args=None):
        # X, y both are test set.
        self.test_objective = rankbasedObjective(torch.from_numpy(X.copy()), torch.from_numpy(y.copy()),weight_function,loss,l2_reg,l1_reg,B,n_class,args)
        self.w_time = [0]
        self.z_time = [0]
        self.train_losses = [self.objective.get_arrogate_loss(torch.from_numpy(self.w).double())]
        self.test_losses = [self.test_objective.get_arrogate_loss(torch.from_numpy(self.w).double())]
        self.time_array = [0]
        self.store = True
        
    def z_subproblem(self):
        m = self.D @ self.w - self.lagrangian / self.rho
        m = m.reshape(-1)

        sort_index = np.argsort(m)
        sorted_m = np.sort(m)
        
        solver = PAV_solver(self.objective.alphas.numpy().reshape(-1),sorted_m, 
                            self.rho, self.loss, self.B,self.objective.betas.numpy().reshape(-1))
        
        pav_res = solver.get_opt(maxiter=self.z_maxiter)
        new_z = np.zeros(shape=self.z.shape)
        new_z[sort_index] = pav_res.reshape(self.z.shape)
        
        return new_z
    
    
    def w_subproblem(self):
        if self.w_flag == 0:
            b = self.z + self.lagrangian / self.rho
            w = self.inv_w_matrix @ b
        elif self.w_flag == 2:
            w = w_solver(self.w_flag, self.w, self.z, self.lagrangian, self.rho, 
                              self.DTD, self.D, self.reg)
        return w
        

    def main_loop(self, i, t_start, verbose):
        t1 = time.time()
        self.z = self._z_subproblem()
        t2 = time.time()
        if self.store:
            self.z_time.append(t2 - t1 + self.z_time[i])
        
        pre_w = self.w.copy()
        self.w = self._w_subproblem()
        if self.store:
            self.w_time.append(time.time() - t2 + self.w_time[i])
        
        # Lagrange multiplier update
        self.lagrangian = self.lagrangian + self.rho * (self.z - self.D @ self.w)
        
        # stopping criterion
        primal_feasibility = np.linalg.norm(self.z - self.D @ self.w)
        dual_feasibility = np.linalg.norm(self.w - pre_w)
        if primal_feasibility < self.tol and dual_feasibility < self.tol:
            print('algorithm converges within tolerance')
            print('iter_num=', i, 'primal_feasibility: ', primal_feasibility, 'dual_feasibility: ', dual_feasibility)  
            print('loss=', self.objective.get_arrogate_loss(torch.from_numpy(self.w).double()))
            return True
        if verbose:
            if i % 10 == 0:
                print('iter_num=', i, 'primal_feasibility: ', primal_feasibility, 'dual_feasibility: ', dual_feasibility) 
                print('loss=', self.objective.get_arrogate_loss(torch.from_numpy(self.w).double()))
                
        # ALM penalty update
        if self.loss == 'ehrm' or self.loss == 'aorr':
            self.rho = np.min((self.rho * 1.2, 17 * self.num_feature))
        elif self.loss == 'aorr_dc':
            if i >= 7 and i % 3 == 0:
                self.rho = np.min((self.rho * 5, 17 * self.num_feature))
        else:
            if primal_feasibility > 1e-2:
                self.rho = np.min((self.rho * 1.02, 217 * self.num_feature))
            else:
                self.rho = np.min((self.rho * 1.07, 217 * self.num_feature))
        
        if self.store:
            self.train_losses.append(self.objective.get_arrogate_loss(torch.from_numpy(self.w).double()))
            self.test_losses.append(self.test_objective.get_arrogate_loss(torch.from_numpy(self.w).double()))
            self.time_array.append(time.time()-t_start)
        
        return False

    def final_res(self):
        if self.store:
            return self.w, self.time_array, self.train_losses, self.test_losses
        else:
            raise ValueError("Data was not saved.")


class ADMMmethod(Optimizer):
    def __init__(self, X, y, weight_function="erm", loss="binary_cross_entropy",
                 l2_reg=None,l1_reg=None,B=None, n_class=None, args=None, w0=None, max_iter=200, tol=1e-4):
        super(ADMMmethod, self).__init__(X, y, weight_function, loss, l2_reg,l1_reg, B, n_class, 
                                         args, w0, max_iter, tol)
        
        
    def start_store(self,X,y,weight_function="erm", loss="binary_cross_entropy",
                 B=None,l2_reg=None,l1_reg=None, n_class=None, args=None):
        super(ADMMmethod, self).start_store(X, y,weight_function, loss, B, 
                                       l2_reg,l1_reg, n_class, args)
        
        
    def _z_subproblem(self):
        return super(ADMMmethod, self).z_subproblem()
    
    
    def _w_subproblem(self):
        if self.w_flag == 1:
            const_y = self.z + self.lagrangian / self.rho
            alpha = self.reg / (2 * self.rho * self.num_row)
            if self.num_row <= 500 and self.num_feature <= 60:
                lasso_model = Lasso(alpha=alpha, tol=1e-8, fit_intercept=False, max_iter=50000, warm_start=True)  # noqa: E501
                lasso_model.fit(X=self.D, y=const_y)
                w = lasso_model.coef_.reshape(-1, 1)
            else:
                L = np.float32(17)
                eta = np.float32(2.5)
                w = FISTA(beta=self.w.reshape(-1,), X=self.D, y=const_y.reshape(-1,), lam=alpha*self.num_row, L=L, eta=eta, tol=self.w_tol, max_iter=5000, dtype=torch.float32)  # noqa: E501
                w = w.reshape(-1, 1)
        elif self.w_flag == 0 or self.w_flag == 2:
            w= super(ADMMmethod, self).w_subproblem()
        else:
            raise ValueError("w_flag can only be 0, 1 or 2.")
        return w

    def main_loop(self, verbose=True):
        t_start = time.time()
        
        for i in range(self.max_iter):
            if super(ADMMmethod, self).main_loop(i, t_start, verbose):
                break
            
        return self.w


    def final_res(self):
        return super(ADMMmethod, self).final_res()
        

class smoothADMMmethod(Optimizer):
    def __init__(self, X, y, weight_function="erm", loss="binary_cross_entropy",
                 B=None,l2_reg=None,l1_reg=None, n_class=None, args=None, w0=None, t=1, max_iter=200, tol=1e-4):
        super(smoothADMMmethod, self).__init__(X, y, weight_function, loss, l2_reg,l1_reg, B, n_class, 
                                         args, w0, max_iter, tol)
        self.t = t
        
    def start_store(self,X,y,weight_function="erm", loss="binary_cross_entropy",
                 B=None,l2_reg=None,l1_reg=None, n_class=None, args=None):
        super(smoothADMMmethod, self).start_store(X, y,weight_function, loss, B, 
                                       l2_reg,l1_reg, n_class, args)
    
    def _z_subproblem(self):
        return super(smoothADMMmethod, self).z_subproblem()
    
    def _w_subproblem(self):
        if self.w_flag == 1:
            w = w_solver(self.w_flag, self.w, self.z, self.lagrangian, self.rho, 
                              self.DTD, self.D, self.reg, self.t)
        elif self.w_flag == 0 or self.w_flag == 2:
            w= super(smoothADMMmethod, self).w_subproblem()
        else:
            raise ValueError("w_flag can only be 0, 1 or 2.")
        return w

    def main_loop(self, verbose=True):
        t_start = time.time()
        
        for i in range(self.max_iter):
            if super(smoothADMMmethod, self).main_loop(i, t_start, verbose):
                break
            if i >= 17:
                self.t = max(self.t * 0.9, 1e-9) % np.power(self.rho, -0.1)  * np.power(i, -0.1)
         
        if self.w_flag == 1:
            self.w = np.sign(self.w)*np.where((np.abs(self.w)-self.t)>0,np.abs(self.w)-self.t,0)
            print('final true loss=', self.objective.get_arrogate_loss(torch.from_numpy(self.w).double()))
        return self.w

    def final_res(self):
        return super(smoothADMMmethod, self).final_res()
