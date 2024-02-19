# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 19:17:17 2023

@author: XRF
"""
from src.optim.algorithms import ADMMmethod
from src.util.load_data import get_data
from src.util.calculate_acc import calculate_accuracy
from sklearn.model_selection import train_test_split
from SGD_solver import SGDmethod
from LSVRG_solver import LSVRGmethod
from DCA_solver import DCAmethod
import math
from openpyxl import Workbook
import os
import numpy as np
pwd = os.path.abspath(os.path.dirname(__file__))

need_log = False # True for saving table

num_row = 1000
num_feature = 1000
# num_row and num_feature only for synthetic data
seed = 17
dataname = "synthetic" # Options: ["synthetic"]
X, label = get_data(dataname,num_row=num_row,num_feature=num_feature,seed=seed)

X_train, X_test, y_train, y_test= train_test_split(X, label,test_size=0.5, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=seed)

weight_function = "aorr" # Options: ["aorr"] 
loss = "hinge" # Options: ["binary_cross_entropy","hinge"]
l2_reg = 0.0001
l1_reg = None
verbose = True
args = [0.2,0.8]

# add intercept
X_train_other = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
X_test_other = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

admm_solver = ADMMmethod(X_train_other,y_train,weight_function,loss,l2_reg=l2_reg,l1_reg=l1_reg,args=args)
admm_solver.start_store(X_test_other, y_test,weight_function,loss,l2_reg=l2_reg,l1_reg=l1_reg,args=args)
admm_solver.main_loop(verbose=verbose) # verbose=True for printing the loss and time at each ten iteration
admm_w, admm_time_array, admm_train_losses, admm_test_losses=admm_solver.final_res()
admm_test_acc = calculate_accuracy(admm_w.reshape(-1, 1),X_test_other, y_test, threshold=0.5,loss=loss)

l2_reg = l2_reg*X_train.shape[0] 
l1_reg = None 

sgd_w, sgd_train_losses, sgd_test_losses, sgd_time_array = SGDmethod(X_train_other,y_train,weight_function,loss,
                l2_reg=l2_reg,l1_reg=l1_reg,max_iter = 2000,batch_size=64, lr = 0.0001,
                train_loss=admm_solver.objective.get_arrogate_loss, test_loss=admm_solver.test_objective.get_arrogate_loss,
                verbose=verbose,args=args)
sgd_test_acc = calculate_accuracy(sgd_w.reshape(-1, 1),X_test_other, y_test, threshold=0.5,loss=loss)

lsvrg_nu_w, lsvrg_nu_train_losses, lsvrg_nu_test_losses, lsvrg_nu_time_array = LSVRGmethod(X_train_other,y_train,weight_function,loss,
                l2_reg=l2_reg,l1_reg=l1_reg,max_iter = 200, lr = 0.0001,
                train_loss=admm_solver.objective.get_arrogate_loss, test_loss=admm_solver.test_objective.get_arrogate_loss,
                uniform=None,verbose=verbose,args=args)
lsvrg_nu_acc = calculate_accuracy(lsvrg_nu_w.reshape(-1, 1),X_test_other, y_test, threshold=0.5,loss=loss)

lsvrg_u_w, lsvrg_u_train_losses, lsvrg_u_test_losses, lsvrg_u_time_array = LSVRGmethod(X_train_other,y_train,weight_function,loss,
                l2_reg=l2_reg,l1_reg=l1_reg,max_iter = 200, lr = 0.0001,
                train_loss=admm_solver.objective.get_arrogate_loss, test_loss=admm_solver.test_objective.get_arrogate_loss,
                uniform=True,verbose=verbose,args=args)
lsvrg_u_acc = calculate_accuracy(lsvrg_u_w.reshape(-1, 1),X_test_other, y_test, threshold=0.5,loss=loss)

dca_w, dca_time, dca_train_loss, dca_test_loss=DCAmethod(X_train,y_train,X_val,y_val,X_test,y_test,loss, sigma=admm_solver.objective.alphas.numpy().reshape(-1), \
                reg=admm_solver.reg, train_loss=admm_solver.objective.get_arrogate_loss, test_loss=admm_solver.test_objective.get_arrogate_loss, \
                    kvalue=math.floor(X_train.shape[0]*args[1]),kvalue2=math.ceil(X_train.shape[0]*args[0]),dataname=dataname)
dca_test_acc = calculate_accuracy(dca_w.reshape(-1, 1),X_test_other, y_test, threshold=0.5,loss=loss)

print("admm train loss:", admm_train_losses[-1])
print("admm test loss:", admm_test_losses[-1])
print("admm time:", admm_time_array[-1])
print("admm test acc:", admm_test_acc)

print("sgd train loss:", sgd_train_losses[-1])
print("sgd test loss:", sgd_test_losses[-1])
print("sgd time:", sgd_time_array[-1])
print("sgd test acc:", sgd_test_acc)

print("lsvrg_u train loss:", lsvrg_u_train_losses[-1])
print("lsvrg_u test loss:", lsvrg_u_test_losses[-1])
print("lsvrg_u time:", lsvrg_u_time_array[-1])
print("lsvrg_u test acc:", lsvrg_u_acc)

print("lsvrg_nu train loss:", lsvrg_nu_train_losses[-1])
print("lsvrg_nu test loss:", lsvrg_nu_test_losses[-1])
print("lsvrg_nu time:", lsvrg_nu_time_array[-1])
print("lsvrg_nu test acc:", lsvrg_nu_acc)

print("dca train loss:", dca_train_loss)
print("dca test loss:", dca_test_loss)
print("dca time:", dca_time)
print("dca test acc:", dca_test_acc)

if need_log:
    workbook = Workbook()
    save_file = os.path.join(pwd, "table", f"{weight_function}_{dataname}_{admm_solver.num_row}x{admm_solver.num_feature}_l2_{loss}.xlsx")
    worksheet = workbook.active
    worksheet.title = "Sheet1"

    worksheet.append(admm_train_losses)
    worksheet.append(admm_time_array)
    worksheet.append([admm_test_acc])

    worksheet.append(sgd_train_losses)
    worksheet.append(sgd_time_array)
    worksheet.append([sgd_test_acc])

    worksheet.append(lsvrg_nu_train_losses)
    worksheet.append(lsvrg_nu_time_array)
    worksheet.append([lsvrg_nu_acc])

    worksheet.append(lsvrg_u_train_losses)
    worksheet.append(lsvrg_u_time_array)
    worksheet.append([lsvrg_u_acc])

    workbook.save(filename=save_file)
