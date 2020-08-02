import os
import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt

from sklearn import preprocessing
from collections import deque


STOCK_TO_PREDICT = "NIFTY_F1"
FUTURE_TO_REDICT = 3 #days
SEQ_LENGTH = 30 #days
PREDICT_HOUR_DATA = False

#############################################  
# Functions and utilities start
#############################################  
def get_data_from_folder(path):
    if os.path.exists(f'{path}/{STOCK_TO_PREDICT}.txt'):
        _df = pd.read_csv(f'{path}/{STOCK_TO_PREDICT}.txt')
        _df.to_csv(f'./data/{STOCK_TO_PREDICT}.csv', mode = 'a', header = False, index = False)
    else:
        for dname in os.listdir(f'{path}'):
            if os.path.isdir(f'{path}/{dname}'):
                get_data_from_folder(f'{path}/{dname}')

def get_main_df():
    # if os.path.exists(f'./data/{STOCK_TO_PREDICT}.csv') == False:
    #     for yname in os.listdir('./data'):
    #         get_data_from_folder(f'./data/{yname}')
    # cols = ['name', 'date', 'time', 'open', 'high', 'low', 'close', 'qty'] # NIFTY_F1,20130305,12:32,5775.4,5775.9,5774.05,5775.9,3250
    main_df = pd.read_csv(f'./data/{STOCK_TO_PREDICT}.csv', names=cols)
    return main_df

def get_hour_data(data): 
    qty = data.qty.values.sum()
    open = data.open.values[0] #start of the hour
    close = data.close.values[-1] #end of the end
    low = data.low.values.min()
    high = data.high.values.max()

    return pd.Series([low, high, open, close, qty],index=['low', 'high', 'open','close', 'qty'])

def get_day_data(data): 
    qty = data.qty.values.sum()
    open = data.open.values[0] #start of the day
    close = data.close.values[-1] #end of the day
    low = data.low.values.min()
    high = data.high.values.max()

    return pd.Series([low, high, open, close, qty],index=['low', 'high', 'open','close', 'qty'])

def sequence_df(scaled_data):
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LENGTH)
    
    for i in scaled_data:
        prev_days.append([n for n in i[:-1]]) #considering last col as y
        if len(prev_days) == SEQ_LENGTH:
            sequential_data.append([np.array(prev_days), i[-1]])

    X = []
    y = []
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)
#############################################    
# Functions and utilities end
#############################################  


#############################################    
# Data preparation start
#############################################  
main_df = get_main_df()

if PREDICT_HOUR_DATA==True:
    FUTURE_TO_REDICT = FUTURE_TO_REDICT * 6
    SEQ_LENGTH = SEQ_LENGTH * 6

    time = main_df["time"].str.split(":", n = 1, expand = True)
    main_df["date"] =  main_df["date"].astype(str) + time[0]

    main_df = main_df.sort_values(['date', 'time']).groupby(['date']).apply(get_hour_data).reset_index('date')
else:
    main_df = main_df.sort_values(['date', 'time']).groupby(['date']).apply(get_day_data).reset_index('date')

main_df["past"] = main_df["close"].shift(+1)
main_df["future"] = main_df["close"].shift(-FUTURE_TO_REDICT)
main_df.dropna(inplace=True)

main_df = main_df.sort_index(ascending=True, axis=0)
main_df = main_df[["past", "open", "future"]]

main_df.dropna(inplace=True)

print(main_df.head())
print('\n Shape of the data:')
print(main_df.shape)

# plt.figure(figsize=(10,6))
# plt.plot(main_df['future'], label='Future Price history')
# plt.show()
#############################################    
# Data preparation end
#############################################  

#############################################    
# Prepare model start
#############################################  
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(main_df)

split_num = int(len(scaled_data) * 0.95)

train_x, train_y = sequence_df(scaled_data[0:split_num])
test_x, test_y = sequence_df(scaled_data[split_num:])

print(f"Train Data: {len(train_x)} Test Data: {len(test_x)}")
print(f"Train Shape: {train_x.shape} Test Shape: {test_x.shape}")
print(f"Train Shape: {train_y.shape} Test Shape: {test_y.shape}")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#gpu fix
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

#model
model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(train_x.shape[1:])))
model.add(LSTM(units=128, return_sequences=True))
model.add(LSTM(units=128))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(
    train_x, 
    train_y, 
    epochs=10, 
    batch_size=120, 
    verbose=1)

#predict
closing_price = model.predict(test_x)

#for plotting
plt.plot(test_y)
plt.plot(closing_price[:,0])
plt.show = plt.show()

#############################################    
# Prepare model end
#############################################  