import pandas as pd
import numpy as np

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
userinf_uids = list(userinf["uid"].values)

dict_score = pd.read_excel('./dict_score_new.xlsx')
dict_score_uid = list(dict_score["uid"].values)
bert_features = np.load('./user_bert_features.npy', allow_pickle=True).item()

# train
train_X1 = []
train_X2 = []
train_Y = []
for i in range(len(train_uids)):
    uid = train_uids[i]
    user_id = train_user_ids[i]
    dvf_d_i = list(dict_score.iloc[dict_score_uid.index(uid), ])[0: -1]
    dvf_b_i = bert_features[user_id]
    y_i = int(userinf.iloc[userinf_uids.index(uid)]["XQ"])
    train_X1.append(dvf_d_i)
    train_X2.append(dvf_b_i)
    train_Y.append(y_i)

train_X1, train_X2 = np.array(train_X1), np.array(train_X2)
train_X = [train_X1, train_X2]
train_Y = np.array(train_Y)

val_X1 = []
val_X2 = []
val_Y = []
for i in range(len(val_uids)):
    uid = val_uids[i]
    user_id = val_user_ids[i]
    dvf_d_i = list(dict_score.iloc[dict_score_uid.index(uid), ])[0: -1]
    val_X1.append(dvf_d_i)
    dvf_b_i = bert_features[user_id]
    val_X2.append(dvf_b_i)
    y = int(userinf.iloc[userinf_uids.index(uid)]["XQ"])
    val_Y.append(y)
val_X1, val_X2 = np.array(val_X1), np.array(val_X2)
val_X = [val_X1, val_X2]
val_Y = np.array(val_Y)

test_X1 = []
test_X2 = []
test_Y = []
for i in range(len(test_uids)):
    uid = test_uids[i]
    user_id = test_user_ids[i]
    dvf_d_i = list(dict_score.iloc[dict_score_uid.index(uid), ])[0: -1]
    test_X1.append(dvf_d_i)
    dvf_b_i = bert_features[user_id]
    test_X2.append(dvf_b_i)
    y = int(userinf.iloc[userinf_uids.index(uid)]["XQ"])
    test_Y.append(y)
test_X1, test_X2 = np.array(test_X1), np.array(test_X2)
test_X = [test_X1, test_X2]
test_Y = np.array(test_Y)


# training
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

    print(str(tp) + ' ' + str(fp))
    print(str(fn) + ' ' + str(tn))
    # print("neg_precision: " + str(precision_neg))
    # print("neg_recall: " + str(recall_neg))
    # print("pos_precision: " + str(precision_pos))
    # print("pos_recall: " + str(recall_pos))
    # print("f1: " + str(f1))
    # print("g_means: " + str(g_means))

    return round(precision_neg, 6), round(recall_pos, 6), round(f1, 6), round(g_means, 6)


# model
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.metrics import *
from keras.losses import *
from keras import regularizers
import os

K.clear_session()
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

dvf_d_input = Input(shape=(118,))
dvf_b_input = Input(shape=(768,))

dvf_b_hidden_layer = Dense(32, activation='relu')(dvf_b_input)
dvf_b_hidden_layer = Dense(16, activation='relu')(dvf_b_hidden_layer)

dvf = Concatenate()([dvf_d_input, dvf_b_hidden_layer])

hidden_layer = Dense(10, activation='relu')(dvf)
prediction = Dense(1, activation='sigmoid')(hidden_layer)

model = Model([dvf_d_input, dvf_b_input], prediction)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-3),
    metrics=['accuracy']
)
model.summary()
# model end

val_precision_neg_list, val_recall_pos_list, val_f1_list, val_g_means_list = [0], [0], [0], [0]
test_precision_neg_list, test_recall_pos_list, test_f1_list, test_g_means_list = [0], [0], [0], [0]

epochs = 500
for epoch in range(epochs):
    print('Epoch ' + str(epoch) + '/' + str(epochs) + ": ")
    model.fit(train_X, train_Y, batch_size=10, epochs=1, class_weight={0: 1, 1: 2}, validation_data=[val_X, val_Y])

    print("Train set: ")
    evaluation(train_Y, model.predict(train_X))

    print("Val set:")
    val_predict_Y = model.predict(val_X)
    val_precision_neg, val_recall_pos, val_f1, val_g_means = evaluation(val_Y, val_predict_Y)

    print("Test set:")
    test_predict_Y = model.predict(test_X)
    test_precision_neg, test_recall_pos, test_f1, test_g_means = evaluation(test_Y, test_predict_Y)

    if val_f1 > max( val_f1_list):
        print("neg_precision: " + str(test_precision_neg))
        print("pos_recall: " + str(test_recall_pos))
        print("f1: " + str(test_f1))
        print("g_means: " + str(test_g_means))

        # document_view_prediction = []
        # for i in range(len(test_user_ids)):
        #     document_view_prediction.append((test_user_ids[i], test_Y[i], test_predict_Y[i]))
        # np.save("document_view_prediction.npy", document_view_prediction)

    val_precision_neg_list.append(val_precision_neg)
    val_recall_pos_list.append(val_recall_pos)
    val_f1_list.append(val_f1)
    val_g_means_list.append(val_g_means)

    test_precision_neg_list.append(test_precision_neg)
    test_recall_pos_list.append(test_recall_pos)
    test_f1_list.append(test_f1)
    test_g_means_list.append(test_g_means)

max_f1 = max(val_f1_list)
max_index = val_f1_list.index(max_f1)

print(max_index)
print("neg_precision: " + str(test_precision_neg_list[max_index]))
print("pos_recall: " + str(test_recall_pos_list[max_index]))
print("f1: " + str(test_f1_list[max_index]))
print("g_means: " + str(test_g_means_list[max_index]))
