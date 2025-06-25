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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers


# --- Load files --- #
file_path = '/home/adekkers'
df = pd.read_csv(f'{file_path}/all_data.csv')

features = ['piek','p_APX', 'kWh_real_zon', 'kWh_nom_zon', 'kWh_sur_zon', 'kWh_shor_zon', 'p_sur', 'p_shor', 'eur_tot_zon', 'eur_APX_zon', 'eur_sur_zon', 'eur_shor_zon', 'Uur', 'ISP', 'maand', 'DagM', 'jaar', 'Imbalance', 'Surplus', 'Shortage', 'Absolute']
target = 'Imbalance'

# input & labels aanmaken
duration = 24
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


def build_lstm(neurons, lr):
    model = Sequential()

    for i, n in enumerate(neurons):
        # return_sequences for all layer, instead of last layer
        return_seq = i < len(neurons) - 1
        model.add(LSTM(n, activation='tanh', return_sequences=return_seq))

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr))
    return model

def save_pickle(title, variable):
  with open(title, "wb") as file_pi:
    pickle.dump(variable, file_pi)

# test_train_split data opslaan
save_pickle("lstm24_xtrain", X_train)
save_pickle("lstm24_xtest", X_test)
save_pickle("lstm24_ytrain", y_train)
save_pickle("lstm24_ytest", y_test)
save_pickle("lstm24_indtrain", ind_train)
save_pickle("lstm24_indtest", ind_test)
save_pickle("lstm24_dataset", dataset)

def open_pickle(file):
    with open(file, 'rb') as file_pi:
        return pickle.load(file_pi)

# X_train = open_pickle("lstm24_xtrain")
# X_test = open_pickle("lstm24_xtest")
# y_train = open_pickle("lstm24_ytrain")
# y_test = open_pickle("lstm24_ytest")
# ind_train = open_pickle("lstm24_indtrain")
# ind_test = open_pickle("lstm24_indtest")
# dataset = open_pickle("lstm24_dataset")

epochs = 200

lstm1 = build_lstm([32], 0.0001)
start = time.time()
history_lstm1 = lstm1.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
lstm1_time = end - start
lstm1.save('24lstm1.keras')
save_pickle('24lstmHist1', history_lstm1)
save_pickle('24lstm1_time', lstm1_time)

lstm2 = build_lstm([64, 32], 0.0001)
start = time.time()
history_lstm2 = lstm2.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
lstm2_time = end - start
lstm2.save('24lstm2.keras')
save_pickle('24lstmHist2', history_lstm2)
save_pickle('24lstm2_time', lstm2_time)

lstm3 = build_lstm([128, 64, 32], 0.0001)
start = time.time()
history_lstm3 = lstm3.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
lstm3_time = end - start
lstm3.save('24lstm3.keras')
save_pickle('24lstmHist3', history_lstm3)
save_pickle('24lstm3_time', lstm3_time)

lstm4 = build_lstm([256, 128, 64, 32], 0.0001)
start = time.time()
history_lstm4 = lstm4.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
lstm4_time = end - start
lstm4.save('24lstm4.keras')
save_pickle('24lstmHist4', history_lstm4)
save_pickle('24lstm4_time', lstm4_time)

lstm5 = build_lstm([512, 256, 128, 64, 32], 0.0001)
start = time.time()
history_lstm5 = lstm5.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
lstm5_time = end - start
lstm5.save('24lstm5.keras')
save_pickle('24lstmHist5', history_lstm5)
save_pickle('24lstm5_time', lstm5_time)

lstm6 = build_lstm([1024, 512, 256, 128, 64, 32], 0.0001)
start = time.time()
history_lstm6 = lstm6.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
lstm6_time = end - start
lstm6.save('24lstm6.keras')
save_pickle('24lstmHist6', history_lstm6)
save_pickle('24lstm6_time', lstm6_time)
