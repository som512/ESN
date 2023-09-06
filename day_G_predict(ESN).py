# -*- coding: utf-8 -*-
import sys, os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

import function as fc
from model import ESN, Tikhonov

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

with open('nadir_arr.pickle', mode='br') as fi:
  nadir_arr = pickle.load(fi)


df = pd.read_csv("df.csv", index_col=0)


def normalization(dev_60): # 正規化
    dev_arr = []
    min = np.min(dev_60)
    max = np.max(dev_60)
    for dev in dev_60:
        dev_arr.append((dev-min)/(max-min))
    return dev_arr

'''
1. データの準備
'''

maxlen = 60
df_inout = df[maxlen-1:len(df['Close']) - 1].reset_index(drop=True).loc[:,['Date','Close','Deviation','Increase Rate','Kyousi']]
df_inout['Date'] = pd.to_datetime(df_inout['Date'])

length_of_sequences = len(df_inout['Close'])
x = [] #入力データ、乖離率
t = [] #教師データ、上昇度



for i in range(length_of_sequences - maxlen + 1): #5650 - 60 + 1
    x.append(normalization(df_inout['Deviation'][i:i+maxlen]))
    t.append(df_inout['Increase Rate'][i+maxlen-1]) #自作教師
    #t.append(df_inout['Kyousi'][i+maxlen-1]) #先輩教師

t_copy = [None for i in range(maxlen -1)]
#上昇なら1、下降なら0にする
for i in range(len(t)):
    if t[i] != None:
        if t[i] >= 0.5:
            t_copy.append(1)
        elif t[i] < 0.5:
            t_copy.append(0)

df_inout['Kyousi'] = t_copy

x = np.array(x).reshape(-1, maxlen)
t = np.array(t).reshape(-1, 1)

#ここで指定した日にちを含めて、以降が予測期間になる
print("予測期間を設定します。")
num = len(df_inout) - df_inout[df_inout['Date']== pd.to_datetime('2014-01-06')].index.values[0] + 1
x_train, x_val, t_train, t_val = \
    train_test_split(x, t, test_size=num, shuffle=False)

'''
2. モデル学習
'''
train_U = x_train
train_D = t_train
test_U = x_val
test_D = t_val


# ESNモデル
N_x = 300  # リザバーのノード数
model = ESN(train_U.shape[1], train_D.shape[1], N_x, density=0.1, 
            input_scale=0.1, rho=0.9)

# 学習(リッジ回帰)
train_Y = model.train(train_U, train_D, 
                        Tikhonov(N_x, train_D.shape[1], 1e-3))

# モデル出力
train_Y = model.predict(train_U)
test_Y = model.predict(test_U)

'''
3. モデル評価
'''
correct_train = 0
failed_train = 0
correct_test = 0
failed_test = 0
for i in range(len(train_U)):
    if train_Y[i] >= 0.5:
        if train_D[i] == 1:
            correct_train = correct_train + 1
        else:
            failed_train = failed_train + 1
    else:
        if train_D[i] == 0:
            correct_train = correct_train + 1
        else:
            failed_train = failed_train + 1

for i in range(len(test_U)):
    if test_Y[i] >= 0.5:
        if test_D[i] == 1:
            correct_test = correct_test + 1
        else:
            failed_test = failed_test + 1
    else:
        if test_D[i] == 0:
            correct_test = correct_test + 1
        else:
            failed_test = failed_test + 1

print("訓練データの正解率[%] : ",int(correct_train/(correct_train+failed_train)*1000)/10,"   評価データの正解率[%] : ",int(correct_test/(correct_test+failed_test)*1000)/10)
