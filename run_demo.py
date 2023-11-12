# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 12:31:50 2023

@author: XRF
"""
from sklearn.model_selection import train_test_split
from src.optim.algorithms import ADMMmethod
from src.util.load_data import get_data
from src.util.split_group import train_test_split_group

num_row = 1000
num_feature = 500
# num_row and num_feature only for synthetic data
seed = 17
dataname = "ad" # Options: ["synthetic","ad","monks-3","splice","svmguide1","australian", "phoneme","titanic","UTKFace"]
X, label, group= get_data(dataname,num_row=num_row,num_feature=num_feature,seed=seed)

if dataname != 'UTKFace':
    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.4, random_state=seed)
else:
    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split_group(X, label, group, test_size=0.2, random_state=seed)

weight_function = "erm" # Options: ["erm","extremile","superquantile","esrm","aorr","aorr_dc","ehrm"] 
loss = "binary_cross_entropy" # Options: ["binary_cross_entropy","hinge"]
l2_reg = 0.0001
l1_reg = None
# at most one of l2_reg and l1_reg can not be None
args = [0.2, 0.8] 
# Options: [r] for "extremile" [q] for "superquantile" [rho] for "esrm" [qlow,qup] for "aorr" [k,m] for "aorr_dc
# "aorr_dc" is for the fixed upper and lower numbers in the section 5.3 of the paper
# "aorr" is for the ratio of upper and lower numbers in the appendix C.3 of the paper 
admm_solver = ADMMmethod(X_train,y_train,weight_function,loss,l2_reg=l2_reg,l1_reg=l1_reg,args=args)
admm_solver.start_store(X_test, y_test,weight_function,loss,l2_reg=l2_reg,l1_reg=l1_reg,args=args)
admm_solver.main_loop(verbose=True) # verbose=True for printing the loss and time at each ten iteration
admm_w, admm_time_array, admm_train_losses, admm_test_losses=admm_solver.final_res()

print("admm train loss:", admm_train_losses[-1])
print("admm test loss:", admm_test_losses[-1])
print("admm time:", admm_time_array[-1])