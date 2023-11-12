import numpy as np

def calculate_statistics(w, X_test, label_test, group_test, threshold=0.5):
    predictions = np.dot(X_test, w)
    class_probs = np.zeros((predictions.shape))
    class_probs[predictions >= 0] = 1 / (1 + np.exp(-predictions[predictions >= 0]))
    class_probs[predictions < 0] = np.exp(predictions[predictions < 0]) / (np.exp(predictions[predictions < 0]) + 1)
    binary_preds = (class_probs >= threshold).astype(int)
    y_test = label_test.copy()
    y_test[y_test == -1] = 0
    G1preP = np.sum(np.where(binary_preds[group_test==0]==1,1,0))
    G2preP = np.sum(np.where(binary_preds[group_test==1]==1,1,0))
    G1P = G1preP / len(binary_preds[group_test==0])  # white
    G2P = G2preP / len(binary_preds[group_test==1])  # others
    SPD =  G2P - G1P
    G1TP = np.sum(np.where(group_test[(binary_preds==1).reshape(-1)][(binary_preds[binary_preds==1] == y_test[binary_preds==1]).reshape(-1)] == 0,1,0))
    G2TP = np.sum(np.where(group_test[(binary_preds==1).reshape(-1)][(binary_preds[binary_preds==1] == y_test[binary_preds==1]).reshape(-1)] == 1,1,0))
    G1FN = np.sum(np.where(group_test[(binary_preds==0).reshape(-1)][(binary_preds[binary_preds==0] != y_test[binary_preds==0]).reshape(-1)] == 0,1,0))
    G2FN = np.sum(np.where(group_test[(binary_preds==0).reshape(-1)][(binary_preds[binary_preds==0] != y_test[binary_preds==0]).reshape(-1)] == 1,1,0))
    G1TN = np.sum(np.where(group_test[(binary_preds==0).reshape(-1)][(binary_preds[binary_preds==0] == y_test[binary_preds==0]).reshape(-1)] == 0,1,0))
    G2TN = np.sum(np.where(group_test[(binary_preds==0).reshape(-1)][(binary_preds[binary_preds==0] == y_test[binary_preds==0]).reshape(-1)] == 1,1,0))
    G1FP = np.sum(np.where(group_test[(binary_preds==1).reshape(-1)][(binary_preds[binary_preds==1] != y_test[binary_preds==1]).reshape(-1)] == 0,1,0))
    G2FP = np.sum(np.where(group_test[(binary_preds==1).reshape(-1)][(binary_preds[binary_preds==1] != y_test[binary_preds==1]).reshape(-1)] == 1,1,0))
    if G1P == 0:
        DI = np.inf
    else:
        DI = G2P/G1P
    TPRG1 = G1TP / (G1TP + G1FN)
    TPRG2 = G2TP / (G2TP + G2FN)
    FPRG1 = G1FP / (G1FP + G1TN)
    FPRG2 = G2FP / (G2FP + G2TN)
    FNRG1 = G1FN / (G1TP + G1FN)
    FNRG2 = G2FN / (G2TP + G2FN)
    EOD = TPRG2 - TPRG1
    AOD = 0.5 * (FPRG2-FPRG1 + EOD)
    b = class_probs - y_test + 1
    mu = np.sum(b.reshape(-1)) / len(b)
    a = (b/mu) * (np.log(b/mu))
    TI = np.sum(a.reshape(-1)) / len(b)
    FNRD = FNRG2 - FNRG1
    return SPD, DI, EOD, AOD, TI ,FNRD