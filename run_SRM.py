# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 16:17:15 2023

@author: XRF
"""
from sklearn.model_selection import train_test_split
from src.optim.algorithms import ADMMmethod, smoothADMMmethod
from src.util.load_data import get_data
from src.util.calculate_acc import calculate_accuracy
from SGD_solver import SGDmethod
from LSVRG_solver import LSVRGmethod
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook
import os
pwd = os.path.abspath(os.path.dirname(__file__))

need_log = True # True for saving the figure and table

num_row = 10000
num_feature = 1000
# num_row and num_feature only for synthetic data
seed = 17
dataname = "synthetic" # Options: ["synthetic","ad","svmguide1"]
X, label= get_data(dataname,num_row=num_row,num_feature=num_feature,seed=seed)

X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.4, random_state=seed)

weight_function = "erm" # Options: ["erm","extremile","superquantile","esrm"] 
loss = "binary_cross_entropy" # Options: ["binary_cross_entropy","hinge"]
l2_reg = None
l1_reg = 0.01
verbose = True
# at most one of l2_reg and l1_reg can not be None
args = [0.2, 0.8] 
# Options: [r] for "extremile" [q] for "superquantile" [rho] for "esrm" [qlow,qup]

admm_solver = ADMMmethod(X_train,y_train,weight_function,loss,l2_reg=l2_reg,l1_reg=l1_reg,args=args)
admm_solver.start_store(X_test, y_test,weight_function,loss,l2_reg=l2_reg,l1_reg=l1_reg,args=args)
admm_solver.main_loop(verbose=verbose) # verbose=True for printing the loss and time at each ten iteration
admm_w, admm_time_array, admm_train_losses, admm_test_losses=admm_solver.final_res()
admm_test_acc = calculate_accuracy(admm_w.reshape(-1, 1),X_test, y_test, threshold=0.5,loss=loss)

if l1_reg is not None:
    sadmm_solver = smoothADMMmethod(X_train,y_train,weight_function,loss,l2_reg=l2_reg,l1_reg=l1_reg,args=args)
    sadmm_solver.start_store(X_test, y_test,weight_function,loss,l2_reg=l2_reg,l1_reg=l1_reg,args=args)
    sadmm_solver.main_loop(verbose=verbose) # verbose=True for printing the loss and time at each ten iteration
    sadmm_w, sadmm_time_array, sadmm_train_losses, sadmm_test_losses=sadmm_solver.final_res()
    sadmm_test_acc = calculate_accuracy(admm_w.reshape(-1, 1),X_test, y_test, threshold=0.5,loss=loss)

sgd_w, sgd_train_losses, sgd_test_losses, sgd_time_array = SGDmethod(X_train,y_train,weight_function,loss,
                l2_reg=l2_reg,l1_reg=l1_reg,max_iter = 1000,batch_size=64, lr = 1e-5,
                train_loss=admm_solver.objective.get_arrogate_loss, test_loss=admm_solver.test_objective.get_arrogate_loss,
                verbose=verbose,args=args)
sgd_test_acc = calculate_accuracy(sgd_w.reshape(-1, 1),X_test, y_test, threshold=0.5,loss=loss)

lsvrg_nu_w, lsvrg_nu_train_losses, lsvrg_nu_test_losses, lsvrg_nu_time_array = LSVRGmethod(X_train,y_train,weight_function,loss,
                l2_reg=l2_reg,l1_reg=l1_reg,max_iter = 1000, lr = 1,
                train_loss=admm_solver.objective.get_arrogate_loss, test_loss=admm_solver.test_objective.get_arrogate_loss,
                uniform=None,verbose=verbose,args=args)
lsvrg_nu_acc = calculate_accuracy(lsvrg_nu_w.reshape(-1, 1),X_test, y_test, threshold=0.5,loss=loss)

lsvrg_u_w, lsvrg_u_train_losses, lsvrg_u_test_losses, lsvrg_u_time_array = LSVRGmethod(X_train,y_train,weight_function,loss,
                l2_reg=l2_reg,l1_reg=l1_reg,max_iter = 1000, lr = 1,
                train_loss=admm_solver.objective.get_arrogate_loss, test_loss=admm_solver.test_objective.get_arrogate_loss,
                uniform=True,verbose=verbose,args=args)
lsvrg_u_acc = calculate_accuracy(lsvrg_u_w.reshape(-1, 1),X_test, y_test, threshold=0.5,loss=loss)

print("admm train loss:", admm_train_losses[-1])
print("admm test loss:", admm_test_losses[-1])
print("admm time:", admm_time_array[-1])
print("admm test acc:", admm_test_acc)

if l1_reg is not None:
    print("sadmm train loss:", sadmm_train_losses[-1])
    print("sadmm test loss:", sadmm_test_losses[-1])
    print("sadmm time:", sadmm_time_array[-1])
    print("sadmm test acc:", sadmm_test_acc)

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

if need_log:
    train_loss_best = min(np.min(admm_train_losses),np.min(sadmm_train_losses),np.min(sgd_train_losses),np.min(lsvrg_nu_train_losses),np.min(lsvrg_u_train_losses))

    plt.semilogy(np.array(admm_time_array), admm_train_losses-train_loss_best, label="ADMM")
    if l1_reg is not None:
        plt.semilogy(np.array(sadmm_time_array), sadmm_train_losses-train_loss_best, label="sADMM")
    plt.semilogy(np.array(sgd_time_array), sgd_train_losses-train_loss_best, label="SGD")
    plt.semilogy(np.array(lsvrg_nu_time_array), lsvrg_nu_train_losses-train_loss_best, label="LSVRG(NU)")
    plt.semilogy(np.array(lsvrg_u_time_array), lsvrg_u_train_losses-train_loss_best, label="LSVRG(U)")

    plt.xlabel(r'CPU time (secs.)', fontsize=15)
    plt.xlim((0,20))
    plt.ylabel(r'Sub-optimality gap $log_{10}(F^k-F^*)$', fontsize=15)
    plt.ylim((1e-8,1e1))

    plt.legend()

    plt.title(f'num_sample: {admm_solver.num_row} num_feature: {admm_solver.num_feature}', fontsize=17)

    if l1_reg is not None:
        save_fig = os.path.join(pwd, "figure", f"{weight_function}_{dataname}_{admm_solver.num_row}x{admm_solver.num_feature}_l1_{loss}.pdf")
    else:
        save_fig = os.path.join(pwd, "figure", f"{weight_function}_{dataname}_{admm_solver.num_row}x{admm_solver.num_feature}_l2_{loss}.pdf")
    plt.savefig(save_fig)

    workbook = Workbook()
    if l1_reg is not None:
        save_file = os.path.join(pwd, "table", f"{weight_function}_{dataname}_{admm_solver.num_row}x{admm_solver.num_feature}_l1_{loss}.xlsx")
    else:
        save_file = os.path.join(pwd, "table", f"{weight_function}_{dataname}_{admm_solver.num_row}x{admm_solver.num_feature}_l2_{loss}.xlsx")
    worksheet = workbook.active
    worksheet.title = "Sheet1"

    worksheet.append(admm_train_losses)
    worksheet.append(admm_time_array)
    worksheet.append([admm_test_acc])

    if l1_reg is not None:
        worksheet.append(sadmm_train_losses)
        worksheet.append(sadmm_time_array)
        worksheet.append([sadmm_test_acc])

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
