# -*- coding: utf-8 -*-
import sys
sys.path.append("C:/Users/daisa/プログラミング/Python/2023/卒研/(自作)株価予測2/nikkei_simulation")
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas_datareader.data as web
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
yf.pdr_override()
import itertools
import pickle

import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from callbacks import EarlyStopping


import function as fc
from day_Resorvoir.resorvoir.model import ESN, Tikhonov

import os
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


os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("..")



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

sys.exit()

# 訓練誤差評価（NRMSE）
RMSE = np.sqrt(((train_D/data_scale - train_Y/data_scale) ** 2)
                        .mean())
NRMSE = RMSE/np.sqrt(np.var(train_D/data_scale))
print(step, 'ステップ先予測')
print('訓練誤差：NRMSE =', NRMSE)

# 検証誤差評価（NRMSE）
RMSE = np.sqrt(((test_D/data_scale - test_Y/data_scale) ** 2)
                    .mean())
NRMSE = RMSE/np.sqrt(np.var(test_D/data_scale))
print(step, 'ステップ先予測')
print('検証誤差：NRMSE =', NRMSE)


'''
4. モデルの評価
'''
gen = [None] * (maxlen -1)  #ディジタル[0,1]

preds  = [None] * (maxlen*2 -1) #アナログ


print("予測値を計算します")
y = model(x)


correct_val = 0
mistake_val = 0
correct_train = 0
mistake_train = 0
y_val = model(x_val)
y_train = model(x_train)



for i in range(len(y_val)):
    if y_val[i][0] >= 0.5:
        y_v = 1
    else:
        y_v = 0

    if y_v == t_val[i][0]:
        correct_val = correct_val + 1
    else:
        mistake_val = mistake_val + 1

for i in range(len(y_train)):
    if y_train[i][0] >= 0.5:
        y_t = 1
    else:
        y_t = 0
    
    if y_t == t_train[i][0]:
        correct_train = correct_train + 1
    else:
        mistake_train = mistake_train + 1



#上昇なら1、下降なら0にする
for i in range(len(x)):
    if y[i][0] >= 0.5:
        gen.append(1)
        preds.append(y[i][0])
    else:
        gen.append(0)
        preds.append(y[i][0])
df_inout['Predict'] = gen


df_inout.to_csv('df_inout.csv')

'''
correct = 0
mistake = 0
for i in range(len(df_val['Date'])):
    if df_val['Predict'][i] != np.nan or df_val['Kyosi'][i] != np.nan:
        if df_val['Predict'][i] == df_val['Kyosi'][i]:
            correct = correct + 1
        elif df_val['Predict'][i] != df_val['Kyosi'][i]:
            mistake = mistake + 1
print("すべての正答率[%]", correct / (correct + mistake) * 100)
'''

print("0.5学習期間の正答率[%]", correct_train / (correct_train + mistake_train) * 100)
print("0.5検証期間の正答率[%]", correct_val / (correct_val + mistake_val) * 100)
print("最大正解率", max(accuracy_rate))




#プロット-----------------------------------------------------------------------------------------
os.chdir("./pic")


#日経平均株価、天井度、上昇度
train_val_term = df_inout.loc[df_inout.index==len(t_train), 'Date'].values[0]
fig = plt.figure()
plt.rc('font', family='serif')
plt.title("日経平均株価 天井と底あり", fontname="MS Gothic", fontsize=36)
plt.plot(pd.to_datetime(df["Date"]) , df['Close'],color='black', linewidth=0.5, label="日経平均株価")
for i in range(1,len(nadir_arr)):
    if nadir_arr[i][2] == 1:
        plt.plot(nadir_arr[i][0], nadir_arr[i][1], marker='o', color='red')
    else:
        plt.plot(nadir_arr[i][0], nadir_arr[i][1], marker='x', color='blue')
plt.axvline(x=train_val_term, linewidth=2.0)
plt.ylabel("[円]", fontsize=18, fontname="MS Gothic")
plt.tick_params(labelsize=18) #軸の目盛り
plt.savefig("fig1.png")

#正解率
fig = plt.figure()
plt.rc('font', family='serif')
plt.title("エポック毎の正解率", fontname="MS Gothic", fontsize=36)
plt.plot([i for i in range(1, len(accuracy_rate)+1)], accuracy_rate, color='black', label="正解率")
plt.legend(prop = {"family" : "MS Gothic","size" : "20"}) #図中のラベル名
plt.tick_params(labelsize=18) #軸の目盛り
plt.savefig("fig2.png")

#学習誤差と評価誤差
fig = plt.figure()
plt.rc('font', family='serif')
plt.title("学習誤差と評価誤差", fontname="MS Gothic", fontsize=36)
plt.plot(epoch_arr, train_loss_arr, color='black', label="学習誤差")
plt.plot(epoch_arr, val_loss_arr, color='black', label="評価誤差")
plt.legend(prop = {"family" : "MS Gothic","size" : "20"}) #図中のラベル名
plt.tick_params(labelsize=18) #軸の目盛り
plt.savefig("fig3.png")

'''
fig = plt.figure()
plt.rc('font', family='serif')
plt.title("上昇度と予測上昇度", fontname="MS Gothic", fontsize=36)
plt.axvline(x=train_val_term, linewidth=2.0)
plt.plot(date, increase_rate_arr, color='black', label="上昇度")
plt.plot(date, preds, color='r', label="予測上昇度")
plt.legend(prop = {"family" : "MS Gothic","size" : "20"}) #図中のラベル名
plt.tick_params(labelsize=18) #軸の目盛り
'''

'''
fig = plt.figure()
plt.rc('font', family='serif')

ax1 = fig.add_subplot(3, 1, 1)
ax1.set_title("日経平均株価 天井と底あり", fontname="MS Gothic")
ax1.plot(date, df['Close'],color='black', linewidth=0.5)
for a in nadir_arr:
    ax1.plot(pd.to_datetime(a[0]),a[1],'o',markersize=3,color='r')
ax1.axvline(x=train_val_term, linewidth=2.0)


ax2 = fig.add_subplot(3, 1, 2)
ax2.set_title("天井度", fontname="MS Gothic")
ax2.plot(date, ceiling_degree_arr, color='black')
ax2.axvline(x=train_val_term, linewidth=2.0)

ax3 = fig.add_subplot(3, 1, 3)
ax3.axhline(y=0.5,linestyle='dashed')
ax3.set_title("上昇度", fontname="MS Gothic")
ax3.plot(date, increase_rate_arr, color='black')
ax3.plot(date, preds, color='r')
ax3.axvline(x=train_val_term, linewidth=2.0)

'''


plt.show()