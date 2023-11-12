import numpy as np
import csv
from sklearn import preprocessing
from sklearn.datasets import make_classification
import pandas as pd
from scipy.io import loadmat
import os
pwd = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def get_data(dataname, num_row=None, num_feature=None, seed=17):
    if dataname == "ad":
        file_path = pwd+"/dataset/ad.data"
        with open(file_path, 'r') as file:
            # creative CSV reader
            csv_reader = csv.reader(file)
            data = np.array([row for row in csv_reader])

        condition = np.any(data == "?", axis=1)
        data = data[~condition]
        condition = np.any(data == "   ?", axis=1)
        data = data[~condition]
        condition = np.any(data == "     ?", axis=1)
        data = data[~condition]

        data[data == 'ad.'] = 1
        data[data == 'nonad.'] = -1

        data = data.astype(float)
        label = data[:, -1]
        X = data[:, :-1]
    elif dataname == "svmguide1":
        from libsvmdata import fetch_libsvm 
        X, label = fetch_libsvm("svmguide1")
        label[label == 0] = -1
    elif dataname == "monks-3":
        file_path = pwd+'/dataset/{}.test'.format(dataname)
        dataset = pd.read_csv(file_path, header=None, delimiter=' ')
        X = dataset.values[:,2:8].astype(float)
        label = dataset.values[:,1].astype(np.int32)
    elif dataname == 'splice':
        file_path = pwd+"/dataset/splice.data"
        with open(file_path, 'r') as file:
            # creative CSV reader
            csv_reader = csv.reader(file)
            dataset = np.array([row for row in csv_reader])
        label = dataset[:, 0].copy()
        label[dataset[:, 0] == 'N'] = -1
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

        X = np.array(result).astype(float)
    elif dataname in ["australian", "phoneme","titanic"]:
        file_path = pwd+'/dataset/{}.csv'.format(dataname)
        dataset = pd.read_csv(file_path,header=None, delimiter=',')
        X = dataset.values[:,:-1].astype(float)
        label = dataset.values[:,-1].astype(np.int32)
        label[label==0] = -1
    elif dataname == "UTKFace":
        data = []
        file_path = pwd+"/dataset/UTKFace/landmark_list_part1.txt"
        with open(file_path,"r") as f:
            for line in f.readlines():
                line = line.strip("\n")
                line = line.split()
                data.append(line)   
        data = np.array(data)
        data2 = data[:,0].reshape(-1)
        label = []
        group = []
        index = []
        count = 0
        for line in data2:
            if len(line.split("_")) != 4:
                index.append(count)
            count += 1
        data = np.delete(data, index, 0)
        data2 = data[:,0].reshape(-1)
        for line in data2:
            label.append(line.split("_")[1])
            group.append(line.split("_")[2])
        label = np.array(label)
        group = np.array(group)
        label = label.astype(float)
        group = group.astype(float)
        label[label == 0] = -1
        group[group != 0] = 1
        X = data[:,1:]
        X = X.astype(float)
        label = label.reshape((-1, 1))
        return X,label,group
    elif dataname == "synthetic":
        if num_row is None or num_feature is None:
            raise ValueError("Number of samples and features should be specified for synthetic data!")
        else:
            (X, label) = make_classification(n_samples=num_row, n_features=num_feature, n_classes=2, 
                                            random_state=seed)  
            label[label == 0] = -1
            label = label.reshape((-1, 1))
    else:
        raise ValueError(
            f"Unrecognized data '{dataname}'! Options: ['ad', 'svmguide1','UTKFace','synthetic']"
        )

    label = label.reshape((-1, 1))
    X = preprocessing.scale(X)
    return X, label

if __name__ == "__main__" :
    X, label= get_data("ad")
    ratio = np.sum(np.where(label==1,1,0))/np.sum(np.where(label==-1,1,0))
    input()