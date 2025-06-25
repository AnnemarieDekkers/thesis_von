import numpy as np
import pandas as pd
import time
import pickle

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import deque

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers


# --- Load files --- #
file_path = '/home/adekkers'
df = pd.read_csv(f'{file_path}/all_data.csv')

features = ['piek','p_APX', 'kWh_real_zon', 'kWh_nom_zon', 'kWh_sur_zon', 'kWh_shor_zon', 'p_sur', 'p_shor', 'eur_tot_zon', 'eur_APX_zon', 'eur_sur_zon', 'eur_shor_zon', 'Uur', 'ISP', 'maand', 'DagM', 'jaar', 'Imbalance', 'Surplus', 'Shortage', 'Absolute']
target = 'Imbalance'

# input & labels aanmaken
duration = 12
future = 4
modulo = 1     # 1 = ISP, 4 = hour

input = []  # list of inputs (sequence of timesteps values)
labels = [] # list of labels (future timestep value)

dataset = []

ISP_counter = 0
for index, row in df.iterrows():
  if ISP_counter % modulo == 0:  # every modulo-th row is new hourly value
    dataset.append(row[features])
  ISP_counter += 1

timeseries = deque()

i = 0
for row in dataset:  # dataset is your sequence of p_APX values
    if i < duration+future:
        timeseries.append(row)
    else:
        input.append(list(timeseries)[:duration])  # current timeseries becomes input
        labels.append(row['Imbalance'])            # next value is the label
        timeseries.rotate(-1)
        timeseries[-1] = row
    i += 1


count = len(input)
aantal_features = len(features)
X = np.array(input)
X = X.reshape((count, duration, aantal_features))  # shape: (samples, timesteps, features)
y = np.array(labels).astype('float')               # shape: (samples,)
indices = np.arange(count)

X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(X, y, indices, test_size=0.1, shuffle=True, random_state=42)


def build_gru(neurons, lr):
    model = Sequential()
    for i, n in enumerate(neurons):
        # return_sequences for all layer, instead of last layer
        return_seq = i < len(neurons) - 1
        model.add(GRU(n, activation='tanh', return_sequences=return_seq))
    model.add(Dense(1))
    adam = optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def save_pickle(title, variable):
  with open(title, "wb") as file_pi:
    pickle.dump(variable, file_pi)
  
# test_train_split data opslaan
save_pickle("gru12_xtrain", X_train)
save_pickle("gru12_xtest", X_test)
save_pickle("gru12_ytrain", y_train)
save_pickle("gru12_ytest", y_test)
save_pickle("gru12_indtrain", ind_train)
save_pickle("gru12_indtest", ind_test)
save_pickle("gru12_dataset", dataset)

def open_pickle(file):
    with open(file, 'rb') as file_pi:
        return pickle.load(file_pi)

# X_train = open_pickle("gru12_xtrain")
# X_test = open_pickle("gru12_xtest")
# y_train = open_pickle("gru12_ytrain")
# y_test = open_pickle("gru12_ytest")
# ind_train = open_pickle("gru12_indtrain")
# ind_test = open_pickle("gru12_indtest")
# dataset = open_pickle("gru12_dataset")

epochs = 200

gru1 = build_gru([32], 0.0001)
start = time.time()
history_gru1 = gru1.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
gru1_time = end - start
gru1.save('12gru1.keras')
save_pickle('12gruHist1', history_gru1)
save_pickle('12gru1_time', gru1_time)

gru2 = build_gru([64, 32], 0.0001)
start = time.time()
history_gru2 = gru2.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
gru2_time = end - start
gru2.save('12gru2.keras')
save_pickle('12gruHist2', history_gru2)
save_pickle('12gru2_time', gru2_time)

gru3 = build_gru([128, 64, 32], 0.0001)
start = time.time()
history_gru3 = gru3.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
gru3_time = end - start
gru3.save('12gru3.keras')
save_pickle('12gruHist3', history_gru3)
save_pickle('12gru3_time', gru3_time)

gru4 = build_gru([256, 128, 64, 32], 0.0001)
start = time.time()
history_gru4 = gru4.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
gru4_time = end - start
gru4.save('12gru4.keras')
save_pickle('12gruHist4', history_gru4)
save_pickle('12gru4_time', gru4_time)

gru5 = build_gru([512, 256, 128, 64, 32], 0.0001)
start = time.time()
history_gru5 = gru5.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
gru5_time = end - start
gru5.save('12gru5.keras')
save_pickle('12gruHist5', history_gru5)
save_pickle('12gru5_time', gru5_time)

gru6 = build_gru([1024, 512, 256, 128, 64, 32], 0.0001)
start = time.time()
history_gru6 = gru6.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
gru6_time = end - start
gru6.save('12gru6.keras')
save_pickle('12gruHist6', history_gru6)
save_pickle('12gru6_time', gru6_time)
