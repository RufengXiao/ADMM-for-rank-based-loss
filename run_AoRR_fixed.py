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
import sklearn.metrics as metrics
from openpyxl import Workbook
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import csv
pwd = os.path.abspath(os.path.dirname(__file__))

def sample_per_class(random_state, labels, size_ratio, forbidden_indices=None):
    uniqueValues, occurCount = np.unique(labels, return_counts=True)
    num_samples = len(labels)
    num_classes = len(uniqueValues)
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], int(len(sample_indices_per_class[class_index])*size_ratio), replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])

need_log = False # True for saving table

seed = 17
dataname = "australian" # Options: ["monks-3","splice","australian", "phoneme","titanic"]

train_size = 0.5
val_size = 0.25

if dataname == "monks-3":
    dataset = pd.read_csv(pwd+'/dataset/{}.test'.format(dataname),header=None, delimiter=' ')
    data = dataset.values[:,2:8].astype(float)
    data = preprocessing.scale(data)
    label = dataset.values[:,1].astype(np.int32)
elif dataname == 'splice':
    file_path = pwd+"/dataset/splice.data"
    data_array = np.fromfile(file_path, dtype=np.float32)

    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        dataset = np.array([row for row in csv_reader])
    label = dataset[:, 0].copy()
    label[dataset[:, 0] == 'N'] = 0
    label[dataset[:, 0] != 'N'] = 1
    label = label.astype(np.int32)
    X = dataset[:, -1].copy()
    letter_to_number = {
        'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10,
        'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19,
        't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26
    }
    result = []
    for input_str in X:
        input_str = input_str.replace(" ", "").lower()
        result.append([letter_to_number[char] for char in input_str])

    data = np.array(result).astype(float)
    data = preprocessing.scale(data)
else:
    dataset = pd.read_csv(pwd+'/dataset/{}.csv'.format(dataname),header=None, delimiter=',')
    data = dataset.values[:,:-1].astype(float)
    data = preprocessing.scale(data)
    label = dataset.values[:,-1].astype(np.int32)
    label[label==-1] = 0

random_state = np.random.RandomState(seed)
uniqueValues, occurCount = np.unique(label, return_counts=True)
remaining_indices = list(range(len(label)))
train_indices = sample_per_class(random_state, label, train_size)
val_indices = sample_per_class(random_state, label, val_size*2, forbidden_indices=train_indices)
forbidden_indices = np.concatenate((train_indices, val_indices))
test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

X_train = data[train_indices,:]
y_train = label[train_indices]

X_val = data[val_indices,:]
y_val = label[val_indices]

X_test = data[test_indices, :]
y_test = label[test_indices]

y_train[y_train==0] = -1
y_val[y_val==0] = -1
y_test[y_test==0] = -1
y_train = y_train.reshape(-1,1)
y_val = y_val.reshape(-1,1)
y_test = y_test.reshape(-1,1)

weight_function = "aorr_dc" # Options: ["aorr_dc"] 
loss = "hinge" # Options: ["binary_cross_entropy","hinge"]
l2_reg = 0.0001
l1_reg = None
verbose = True
if loss == "binary_cross_entropy":
    if dataname == "monks-3":
        kvalue = 70
        kvalue2 = 20
    elif dataname == "australian":
        kvalue = 80
        kvalue2 = 3
    elif dataname == "phoneme":
        kvalue = 1400
        kvalue2 = 100
    elif dataname == "titanic":
        kvalue = 500
        kvalue2 = 10
    elif dataname == "splice":
        kvalue = 450
        kvalue2 = 50
else:
    if dataname == "monks-3":
        kvalue = 70
        kvalue2 = 45
    elif dataname == "australian":
        kvalue = 80
        kvalue2 = 3
    elif dataname == "phoneme":
        kvalue = 1400
        kvalue2 = 410
    elif dataname == "titanic":
        kvalue = 500
        kvalue2 = 10
    elif dataname == "splice":
        kvalue = 450
        kvalue2 = 50
args = [kvalue,kvalue2]

# add intercept
X_train_other = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
X_test_other = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

admm_solver = ADMMmethod(X_train_other,y_train,weight_function,loss,l2_reg=l2_reg,l1_reg=l1_reg,args=args)
admm_solver.start_store(X_test_other, y_test,weight_function,loss,l2_reg=l2_reg,l1_reg=l1_reg,args=[1,0])
admm_solver.main_loop(verbose=verbose) # verbose=True for printing the loss and time at each ten iteration
admm_w, admm_time_array, admm_train_losses, admm_test_losses=admm_solver.final_res()
admm_test_acc = calculate_accuracy(admm_w.reshape(-1, 1),X_test_other, y_test, threshold=0.5,loss=loss)

l2_reg = l2_reg*X_train.shape[0] 
l1_reg = None 

sgd_w, sgd_train_losses, sgd_test_losses, sgd_time_array = SGDmethod(X_train_other,y_train,weight_function,loss,
                l2_reg=l2_reg,l1_reg=l1_reg,max_iter = 2000,batch_size=64, lr = 0.01,
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


if loss == "binary_cross_entropy":
    y_train = y_train.reshape(-1)
    y_val = y_val.reshape(-1)
    y_test = y_test.reshape(-1)
    y_train[y_train==-1] = 0
    y_val[y_val==-1] = 0
    y_test[y_test==-1] = 0
else:
    y_train = y_train.reshape(-1)
    y_val = y_val.reshape(-1)
    y_test = y_test.reshape(-1)

dca_w, dca_time, dca_train_loss, dca_test_loss, predicted=DCAmethod(X_train,y_train,X_val,y_val,X_test,y_test,loss,  sigma=admm_solver.objective.alphas.numpy().reshape(-1), \
                reg=admm_solver.reg, train_loss=admm_solver.objective.get_arrogate_loss, test_loss=admm_solver.test_objective.get_arrogate_loss, \
                    kvalue=kvalue,kvalue2=kvalue2,dataname=dataname)
dca_test_acc = metrics.classification_report(y_test, predicted, digits=3, output_dict=True)['accuracy']

print("admm train loss:", admm_train_losses[-1])
print("admm time:", admm_time_array[-1])
print("admm test acc:", admm_test_acc)

print("sgd train loss:", sgd_train_losses[-1])
print("sgd time:", sgd_time_array[-1])
print("sgd test acc:", sgd_test_acc)

print("lsvrg_u train loss:", lsvrg_u_train_losses[-1])
print("lsvrg_u time:", lsvrg_u_time_array[-1])
print("lsvrg_u test acc:", lsvrg_u_acc)

print("lsvrg_nu train loss:", lsvrg_nu_train_losses[-1])
print("lsvrg_nu time:", lsvrg_nu_time_array[-1])
print("lsvrg_nu test acc:", lsvrg_nu_acc)

print("dca train loss:", dca_train_loss)
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
