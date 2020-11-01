import pandas as pd
import numpy as np

test_user = pd.read_excel('assets/test_user.xlsx')
test_uids = list(test_user["uid"].values)
test_user_ids = list(test_user["user_id"].values)

sentence_view_features = np.load("./classifier_2/sentence_view_features.npy", allow_pickle=True).item()

# test_group_1, test_group_2, test_group_3 = [], [], []
# for i in range(len(test_uids)):
#     uid = test_uids[i]
#     user_id = test_user_ids[i]
#
#     svf_i = sentence_view_features[user_id]
#     svf_len = len(svf_i)
#
#     if svf_len > 100:
#         test_group_1.append(i)
#     elif svf_len > 20:
#         test_group_2.append(i)
#     else:
#         test_group_3.append(i)

test_group = range(len(test_uids))
# test_group = test_group_1
# test_group = test_group_2
# test_group = test_group_3

sentence_view_prediction = np.load("./classifier_2/sentence_view_prediction.npy", allow_pickle=True)

temporality_view_prediction = np.load("./classifier_3/temporality_view_prediction.npy", allow_pickle=True)

precision_neg_list = []
recall_pos_list = []
f1_list = []
g_means_list = []

for i in range(1, 6):
    document_view_prediction = np.load("./classifier_1/document_view_prediction_" + str(i) + ".npy", allow_pickle=True)

    tp, fp, fn, tn = 0, 0, 0, 0

    for index in range(len(test_user_ids)):
        if index not in test_group:
            continue

        y_real = document_view_prediction[index][1]

        y1 = 1 if document_view_prediction[index][2] >= 0.5 else 0
        y2 = sentence_view_prediction[index][2]
        y3 = temporality_view_prediction[index][2]

        # y_pred = y1
        # y_pred = float(y2)
        # y_pred = float(y3)
        # y_pred = 0.5 * y1 + 0.5 * float(y2)
        # y_pred = 0.5 * y1 + 0.5 * float(y3)
        # y_pred = 0.5 * float(y2) + 0.5 * float(y3)
        # y_pred = max(int(y1), int(y2), int(y3))
        # y_pred = min(int(y1), int(y2), int(y3))
        # y_pred = 0.34 * int(y1) + 0.34 * int(y2) + 0.34 * int(y3)
        # y_pred = 0.5 * int(y1) + 0.25 * int(y2) + 0.25 * int(y3)
        # y_pred = 0.25 * int(y1) + 0.5 * int(y2) + 0.25 * int(y3)
        y_pred = 0.25 * int(y1) + 0.25 * int(y2) + 0.5 * int(y3)

        if y_real == 1:
            if y_pred >= 0.5:
                tp += 1
            else:
                fn += 1
        else:
            if y_pred >= 0.5:
                fp += 1
            else:
                tn += 1

    # print(tp, fp)
    # print(fn, tn)

    precision_pos = 0 if tp + fp == 0 else tp / (tp + fp)
    precision_neg = 0 if tn + fn == 0 else tn / (tn + fn)
    recall_pos = 0 if tp + fn == 0 else tp / (tp + fn)
    recall_neg = 0 if tn + fp == 0 else tn / (tn + fp)

    precision = (precision_pos + precision_neg) / 2
    recall = (recall_pos + recall_neg) / 2
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    g_means = (recall_pos * recall_neg) ** 0.5

    precision_neg_list.append(precision_neg)
    recall_pos_list.append(recall_pos)
    f1_list.append(f1)
    g_means_list.append(g_means)

print(precision_neg_list)
print(recall_pos_list)
print(f1_list)
print(g_means_list)

print(np.mean(precision_neg_list), np.std(precision_neg_list, ddof=1))
print(np.mean(recall_pos_list), np.std(recall_pos_list, ddof=1))
print(np.mean(f1_list), np.std(f1_list, ddof=1))
print(np.mean(g_means_list), np.std(g_means_list, ddof=1))

print(str(round(float(np.mean(precision_neg_list)), 4)) + "(" + str(round(float(np.std(precision_neg_list)), 4)) + ")")
print(str(round(float(np.mean(recall_pos_list)), 4)) + "(" + str(round(float(np.std(recall_pos_list)), 4)) + ")")
print(str(round(float(np.mean(f1_list)), 4)) + "(" + str(round(float(np.std(f1_list)), 4)) + ")")
print(str(round(float(np.mean(g_means_list)), 4)) + "(" + str(round(float(np.std(g_means_list)), 4)) + ")")
