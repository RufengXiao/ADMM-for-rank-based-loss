# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 19:17:17 2023

@author: XRF
"""
from src.optim.algorithms import ADMMmethod
from src.util.load_data import get_data
from src.util.calculate_acc import calculate_accuracy
from src.util.split_group import train_test_split_group
from src.util.fair_metric import calculate_statistics
from SGD_solver import SGDmethod
from LSVRG_solver import LSVRGmethod
import numpy as np
from openpyxl import Workbook
import os
pwd = os.path.abspath(os.path.dirname(__file__))

need_log = False # True for saving table

seed = 17
dataname = "UTKFace" # Options: ["UTKFace"]
X, label, group = get_data(dataname,seed=seed)

X_train, X_test, y_train, y_test, group_train, group_test= train_test_split_group(X, label, group,test_size=0.4, random_state=seed)

weight_function = "ehrm" # Options: ["ehrm"] 
loss = "binary_cross_entropy" # Options: ["binary_cross_entropy"]
l2_reg = 0.01
l1_reg = None
verbose = True
args = None
B = np.log(1 + np.exp(-5))

admm_solver = ADMMmethod(X_train,y_train,weight_function,loss,l2_reg=l2_reg,l1_reg=l1_reg,B=B,args=args)
admm_solver.start_store(X_test, y_test,weight_function,loss,l2_reg=l2_reg,l1_reg=l1_reg,args=args)
admm_solver.main_loop(verbose=verbose) # verbose=True for printing the loss and time at each ten iteration
admm_w, admm_time_array, admm_train_losses, admm_test_losses=admm_solver.final_res()
admm_test_acc = calculate_accuracy(admm_w.reshape(-1, 1),X_test, y_test, threshold=0.5,loss=loss)
admm_SPD, admm_DI, admm_EOD, admm_AOD, admm_TI ,admm_FNRD = calculate_statistics(admm_w.reshape(-1,1), X_test, y_test, group_test, threshold=0.5)

l2_reg = l2_reg*X_train.shape[0] 
l1_reg = None 

sgd_w, sgd_train_losses, sgd_test_losses, sgd_time_array = SGDmethod(X_train,y_train,weight_function,loss,
                l2_reg=l2_reg,l1_reg=l1_reg,lossB=B,max_iter = 2000,batch_size=64, lr = 0.0001,
                train_loss=admm_solver.objective.get_arrogate_loss, test_loss=admm_solver.test_objective.get_arrogate_loss,
                verbose=verbose,args=args)
sgd_test_acc = calculate_accuracy(sgd_w.reshape(-1, 1),X_test, y_test, threshold=0.5,loss=loss)
sgd_SPD, sgd_DI, sgd_EOD, sgd_AOD, sgd_TI ,sgd_FNRD = calculate_statistics(sgd_w.reshape(-1,1), X_test, y_test, group_test, threshold=0.5)

lsvrg_nu_w, lsvrg_nu_train_losses, lsvrg_nu_test_losses, lsvrg_nu_time_array = LSVRGmethod(X_train,y_train,weight_function,loss,
                l2_reg=l2_reg,l1_reg=l1_reg,lossB=B,max_iter = 200, lr = 0.0001,
                train_loss=admm_solver.objective.get_arrogate_loss, test_loss=admm_solver.test_objective.get_arrogate_loss,
                uniform=None,verbose=verbose,args=args)
lsvrg_nu_acc = calculate_accuracy(lsvrg_nu_w.reshape(-1, 1),X_test, y_test, threshold=0.5,loss=loss)
lsvrg_nu_SPD, lsvrg_nu_DI, lsvrg_nu_EOD, lsvrg_nu_AOD, lsvrg_nu_TI ,lsvrg_nu_FNRD = calculate_statistics(lsvrg_nu_w.reshape(-1,1), X_test, y_test, group_test, threshold=0.5)

lsvrg_u_w, lsvrg_u_train_losses, lsvrg_u_test_losses, lsvrg_u_time_array = LSVRGmethod(X_train,y_train,weight_function,loss,
                l2_reg=l2_reg,l1_reg=l1_reg,lossB=B,max_iter = 200, lr = 0.0001,
                train_loss=admm_solver.objective.get_arrogate_loss, test_loss=admm_solver.test_objective.get_arrogate_loss,
                uniform=True,verbose=verbose,args=args)
lsvrg_u_acc = calculate_accuracy(lsvrg_u_w.reshape(-1, 1),X_test, y_test, threshold=0.5,loss=loss)
lsvrg_u_SPD, lsvrg_u_DI, lsvrg_u_EOD, lsvrg_u_AOD, lsvrg_u_TI ,lsvrg_u_FNRD = calculate_statistics(lsvrg_u_w.reshape(-1,1), X_test, y_test, group_test, threshold=0.5)

print("admm train loss:", admm_train_losses[-1])
print("admm test loss:", admm_test_losses[-1])
print("admm time:", admm_time_array[-1])
print("admm test acc:", admm_test_acc)
print("admm SPD:", admm_SPD, "admm DI:", admm_DI, "admm EOD:", admm_EOD, "admm AOD:", admm_AOD, "admm TI:", admm_TI, "admm FNRD:", admm_FNRD)

print("sgd train loss:", sgd_train_losses[-1])
print("sgd test loss:", sgd_test_losses[-1])
print("sgd time:", sgd_time_array[-1])
print("sgd test acc:", sgd_test_acc)
print("sgd SPD:", sgd_SPD, "sgd DI:", sgd_DI, "sgd EOD:", sgd_EOD, "sgd AOD:", sgd_AOD, "sgd TI:", sgd_TI, "sgd FNRD:", sgd_FNRD)

print("lsvrg_u train loss:", lsvrg_u_train_losses[-1])
print("lsvrg_u test loss:", lsvrg_u_test_losses[-1])
print("lsvrg_u time:", lsvrg_u_time_array[-1])
print("lsvrg_u test acc:", lsvrg_u_acc)
print("lsvrg_u SPD:", lsvrg_u_SPD, "lsvrg_u DI:", lsvrg_u_DI, "lsvrg_u EOD:", lsvrg_u_EOD, "lsvrg_u AOD:", lsvrg_u_AOD, "lsvrg_u TI:", lsvrg_u_TI, "lsvrg_u FNRD:", lsvrg_u_FNRD)

print("lsvrg_nu train loss:", lsvrg_nu_train_losses[-1])
print("lsvrg_nu test loss:", lsvrg_nu_test_losses[-1])
print("lsvrg_nu time:", lsvrg_nu_time_array[-1])
print("lsvrg_nu test acc:", lsvrg_nu_acc)
print("lsvrg_nu SPD:", lsvrg_nu_SPD, "lsvrg_nu DI:", lsvrg_nu_DI, "lsvrg_nu EOD:", lsvrg_nu_EOD, "lsvrg_nu AOD:", lsvrg_nu_AOD, "lsvrg_nu TI:", lsvrg_nu_TI, "lsvrg_nu FNRD:", lsvrg_nu_FNRD)

if need_log:
    workbook = Workbook()
    save_file = os.path.join(pwd, "table", f"{weight_function}_{dataname}_{admm_solver.num_row}x{admm_solver.num_feature}_l2_{loss}.xlsx")
    worksheet = workbook.active
    worksheet.title = "Sheet1"

    worksheet.append(admm_train_losses)
    worksheet.append(admm_time_array)
    worksheet.append([admm_test_acc,admm_SPD, admm_DI, admm_EOD, admm_AOD, admm_TI ,admm_FNRD])

    worksheet.append(sgd_train_losses)
    worksheet.append(sgd_time_array)
    worksheet.append([sgd_test_acc,sgd_SPD, sgd_DI, sgd_EOD, sgd_AOD, sgd_TI ,sgd_FNRD])

    worksheet.append(lsvrg_nu_train_losses)
    worksheet.append(lsvrg_nu_time_array)
    worksheet.append([lsvrg_nu_acc,lsvrg_nu_SPD, lsvrg_nu_DI, lsvrg_nu_EOD, lsvrg_nu_AOD, lsvrg_nu_TI ,lsvrg_nu_FNRD])

    worksheet.append(lsvrg_u_train_losses)
    worksheet.append(lsvrg_u_time_array)
    worksheet.append([lsvrg_u_acc,lsvrg_u_SPD, lsvrg_u_DI, lsvrg_u_EOD, lsvrg_u_AOD, lsvrg_u_TI ,lsvrg_u_FNRD])

    workbook.save(filename=save_file)
