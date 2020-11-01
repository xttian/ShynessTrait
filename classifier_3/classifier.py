import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.linear_model import LogisticRegression

train_user = pd.read_excel('../assets/train_user.xlsx')
train_uids = list(train_user["uid"].values)
train_user_ids = list(train_user["user_id"].values)
val_user = pd.read_excel('../assets/val_user.xlsx')
val_uids = list(val_user["uid"].values)
val_user_ids = list(val_user["user_id"].values)
test_user = pd.read_excel('../assets/test_user.xlsx')
test_uids = list(test_user["uid"].values)
test_user_ids = list(test_user["user_id"].values)

userinf = pd.read_excel('../assets/userinf_new.xlsx')
userinf_user_ids = list(userinf["user_id"].values)
labels = list(userinf["XQ"].values)
labels_dict = dict()
for i in range(len(labels)):
    labels_dict[userinf_user_ids[i]] = labels[i]

user_mean_ip = np.load("user_mean_ip.npy", allow_pickle=True).item()
user_mean_wn = np.load("user_mean_wn.npy", allow_pickle=True).item()
user_var_wn = np.load("user_var_wn.npy", allow_pickle=True).item()
user_mean_sn = np.load("user_mean_sn.npy", allow_pickle=True).item()
user_var_sn = np.load("user_var_sn.npy", allow_pickle=True).item()

# data
train_X = []
train_Y = []
for i in range(len(train_user_ids)):
    x = [user_mean_ip[train_user_ids[i]], user_mean_wn[train_user_ids[i]], user_var_wn[train_user_ids[i]],
         user_mean_sn[train_user_ids[i]], user_var_sn[train_user_ids[i]]]
    train_X.append(x)
    y = labels_dict[train_user_ids[i]]
    train_Y.append(y)
train_X = np.array(train_X)
train_Y = np.array(train_Y)

sample_weight = np.zeros(len(train_Y))
for i in range(0, len(sample_weight)):
    if train_Y[i] == 1:
        sample_weight[i] = 7
    else:
        sample_weight[i] = 1

val_X = []
val_Y = []
for i in range(len(val_user_ids)):
    x = [user_mean_ip[val_user_ids[i]], user_mean_wn[val_user_ids[i]], user_var_wn[val_user_ids[i]],
         user_mean_sn[val_user_ids[i]], user_var_sn[val_user_ids[i]]]
    val_X.append(x)
    y = labels_dict[val_user_ids[i]]
    val_Y.append(y)
val_X = np.array(val_X)
val_Y = np.array(val_Y)

test_X = []
test_Y = []
for i in range(len(test_user_ids)):
    x = [user_mean_ip[test_user_ids[i]], user_mean_wn[test_user_ids[i]], user_var_wn[test_user_ids[i]],
         user_mean_sn[test_user_ids[i]], user_var_sn[test_user_ids[i]]]
    test_X.append(x)
    y = labels_dict[test_user_ids[i]]
    test_Y.append(y)
test_X = np.array(test_X)
test_Y = np.array(test_Y)


# evaluation
def evaluation(test_Y, predict_Y):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(0, len(test_Y)):
        if test_Y[i] == 1:
            if predict_Y[i] >= 0.5:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if predict_Y[i] >= 0.5:
                fp = fp + 1
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

    return round(precision_neg, 6), round(recall_pos, 6), round(f1, 6), round(g_means, 6)


epochs = 100
max_epoch = 0
max_val_f1 = 0.0
max_test_precision_neg, max_test_recall_pos, max_test_f1, max_test_g_means = 0.0, 0.0, 0.0, 0.0
for epoch in range(epochs):
    model = LogisticRegression(max_iter=epoch+1, solver='liblinear')
    model.fit(train_X, train_Y, sample_weight=sample_weight)

    # print("Train set: ")
    evaluation(train_Y, model.predict(train_X))

    # print("Val set:")
    val_predict_Y = model.predict(val_X)
    val_precision_neg, val_recall_pos, val_f1, val_g_means = evaluation(val_Y, val_predict_Y)

    # print("Test set:")
    test_predict_Y = model.predict(test_X)
    test_precision_neg, test_recall_pos, test_f1, test_g_means = evaluation(test_Y, test_predict_Y)

    if val_f1 > max_val_f1:
        max_epoch = epoch
        max_val_f1 = val_f1
        max_test_precision_neg, max_test_recall_pos, max_test_f1, max_test_g_means = test_precision_neg, test_recall_pos, test_f1, test_g_means

        # temporality_view_prediction = []
        # for i in range(len(test_user_ids)):
        #     predict_Y = model.predict(test_X)
        #     temporality_view_prediction.append((test_user_ids[i], test_Y[i], predict_Y[i]))
        # np.save("temporality_view_prediction.npy", temporality_view_prediction)

print(max_epoch)
print(max_val_f1)
print(max_test_precision_neg, max_test_recall_pos, max_test_f1, max_test_g_means)
