import pandas as pd
import numpy as np
import random
random.seed(2020)

train_user = pd.read_excel('../assets/train_user.xlsx')
train_uids = list(train_user["uid"].values)
train_user_ids = list(train_user["user_id"].values)
val_user = pd.read_excel('../assets/val_user.xlsx')
val_uids = list(val_user["uid"].values)
val_user_ids = list(val_user["user_id"].values)

train_uids.extend(val_uids)
train_user_ids.extend(val_user_ids)

test_user = pd.read_excel('../assets/test_user.xlsx')
test_uids = list(test_user["uid"].values)
test_user_ids = list(test_user["user_id"].values)

userinf = pd.read_excel('../assets/userinf_new.xlsx')
userinf_user_ids = list(userinf["user_id"].values)

sentence_view_features = np.load("sentence_view_features.npy", allow_pickle=True).item()

for i in range(len(train_user_ids)):
    svf_i = sentence_view_features[train_user_ids[i]]

# # fine-tune
max_K, max_mu = 0, 0
K_list = range(5, 151, 5)
mu_list = [0.01 * mu for mu in range(1, 21, 1)]

max_g = 0.0

precision_neg_matrix, recall_pos_matrix, f1_matrix, g_means_matrix = [], [], [], []
for K in K_list:
    precision_neg_list, recall_pos_list, f1_list, g_means_list = [], [], [], []

    for mu in mu_list:
        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(len(train_user_ids)):
            svf_i = sentence_view_features[train_user_ids[i]]
            svf_i = svf_i[max(len(svf_i) - K, 0): len(svf_i)]

            if sum(svf_i) / len(svf_i) >= mu:
                if int(userinf.iloc[userinf_user_ids.index(train_user_ids[i])]["XQ"]) == 1:
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:
                if int(userinf.iloc[userinf_user_ids.index(train_user_ids[i])]["XQ"]) == 1:
                    fn = fn + 1
                else:
                    tn = tn + 1

        precision_pos = 0 if tp + fp == 0 else tp / (tp + fp)
        precision_neg = 0 if tn + fn == 0 else tn / (tn + fn)
        recall_pos = 0 if tp + fn == 0 else tp / (tp + fn)
        recall_neg = 0 if tn + fp == 0 else tn / (tn + fp)

        precision = (precision_pos + precision_neg) / 2
        recall = (recall_pos + recall_neg) / 2
        f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        g_means = (recall_pos * recall_neg) ** 0.5

        if g_means > max_g:
            max_g = g_means
            max_K = K
            max_mu = mu

            print("K: " + str(K) + ", mu: " + str(mu))
            print(str(tp) + ' ' + str(fp))
            print(str(fn) + ' ' + str(tn))
            print("neg_precision: " + str(precision_neg))
            print("pos_recall: " + str(recall_pos))
            print("f1: " + str(f1))
            print("g_means: " + str(g_means))

        precision_neg_list.append(precision_neg)
        recall_pos_list.append(recall_pos)
        f1_list.append(f1)
        g_means_list.append(g_means)

    precision_neg_matrix.append(precision_neg_list)
    recall_pos_matrix.append(recall_pos_list)
    f1_matrix.append(f1_list)
    g_means_matrix.append(g_means_list)

print(max_K)
print(max_mu)


K = max_K  # 80
mu = max_mu  # 0.01
tp, fp, tn, fn = 0, 0, 0, 0

test_Y = []
predict_Y = []

for i in range(len(test_user_ids)):
    svf_i = sentence_view_features[test_user_ids[i]]
    svf_i = svf_i[max(len(svf_i) - K, 0): len(svf_i)]

    test_Y.append(int(userinf.iloc[userinf_user_ids.index(test_user_ids[i])]["XQ"]))
    predict_Y.append(1 if sum(svf_i) / len(svf_i) >= mu else 0)

    if sum(svf_i) / len(svf_i) >= mu:
        if int(userinf.iloc[userinf_user_ids.index(test_user_ids[i])]["XQ"]) == 1:
            tp = tp + 1
        else:
            fp = fp + 1
    else:
        if int(userinf.iloc[userinf_user_ids.index(test_user_ids[i])]["XQ"]) == 1:
            fn = fn + 1
        else:
            tn = tn + 1

precision_pos = 0 if tp + fp == 0 else tp / (tp + fp)
precision_neg = 0 if tn + fn == 0 else tn / (tn + fn)
recall_pos = 0 if tp + fn == 0 else tp / (tp + fn)
recall_neg = 0 if tn + fp == 0 else tn / (tn + fp)

precision = (precision_pos + precision_neg) / 2
recall = (recall_pos + recall_neg) / 2
f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
g_means = (recall_pos * recall_neg) ** 0.5

print(str(tp) + ' ' + str(fp))
print(str(fn) + ' ' + str(tn))
print("neg_precision: " + str(precision_neg))
print("pos_recall: " + str(recall_pos))
print("f1: " + str(f1))
print("g_means: " + str(g_means))

# sentence_view_prediction = []
# for i in range(len(test_user_ids)):
#     sentence_view_prediction.append((test_user_ids[i], test_Y[i], predict_Y[i]))
# np.save("sentence_view_prediction.npy", sentence_view_prediction)
