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
from tensorflow.keras.layers import SimpleRNN, Dense, Activation, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers


# --- Load files --- #
file_path = '/home/adekkers'
df = pd.read_csv(f'{file_path}/all_data.csv')

features = ['piek','p_APX', 'kWh_real_zon', 'kWh_nom_zon', 'p_sur', 'p_shor', 'eur_tot_zon', 'Uur', 'ISP', 'maand', 'DagM', 'jaar', 'Imbalance']
target = 'Imbalance'

# input & labels aanmaken
duration = 4
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

def build_biRNN(neurons, lr):
    model = Sequential()
    for i, n in enumerate(neurons):
        # return_sequences for all layer, instead of last layer
        return_seq = i < len(neurons) - 1
        model.add(Bidirectional(SimpleRNN(n, activation='tanh', return_sequences=return_seq)))
    model.add(Dense(1))
    adam = optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def save_pickle(title, variable):
  with open(title, "wb") as file_pi:
    pickle.dump(variable, file_pi)

# test_train_split data opslaan
save_pickle("bi4_xtrain", X_train)
save_pickle("bi4_xtest", X_test)
save_pickle("bi4_ytrain", y_train)
save_pickle("bi4_ytest", y_test)
save_pickle("bi4_indtrain", ind_train)
save_pickle("bi4_indtest", ind_test)
save_pickle("bi4_dataset", dataset)

def open_pickle(file):
    with open(file, 'rb') as file_pi:
        return pickle.load(file_pi)

# X_train = open_pickle("bi4_xtrain")
# X_test = open_pickle("bi4_xtest")
# y_train = open_pickle("bi4_ytrain")
# y_test = open_pickle("bi4_ytest")
# ind_train = open_pickle("bi4_indtrain")
# ind_test = open_pickle("bi4_indtest")
# dataset = open_pickle("bi4_dataset")

epochs = 200

bi1 = build_biRNN([32], 0.0001)
start = time.time()
history_bi1 = bi1.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
bi1_time = end - start
bi1.save('4bi1.keras')
save_pickle('4biHist1', history_bi1)
save_pickle('4bi1_time', bi1_time)

bi2 = build_biRNN([64, 32], 0.0001)
start = time.time()
history_bi2 = bi2.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
bi2_time = end - start
bi2.save('4bi2.keras')
save_pickle('4biHist2', history_bi2)
save_pickle('4bi2_time', bi2_time)

bi3 = build_biRNN([128, 64, 32], 0.0001)
start = time.time()
history_bi3 = bi3.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
bi3_time = end - start
bi3.save('4bi3.keras')
save_pickle('4biHist3', history_bi3)
save_pickle('4bi3_time', bi3_time)

bi4 = build_biRNN([256, 128, 64, 32], 0.0001)
start = time.time()
history_bi4 = bi4.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
bi4_time = end - start
bi4.save('4bi4.keras')
save_pickle('4biHist4', history_bi4)
save_pickle('4bi4_time', bi4_time)


bi5 = build_biRNN([512, 256, 128, 64, 32], 0.0001)
start = time.time()
history_bi5 = bi5.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
bi5_time = end - start
bi5.save('4bi5.keras')
save_pickle('4biHist5', history_bi5)
save_pickle('4bi5_time', bi5_time)


bi6 = build_biRNN([1024, 512, 256, 128, 64, 32], 0.0001)
start = time.time()
history_bi6 = bi6.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
bi6_time = end - start
bi6.save('4bi6.keras')
save_pickle('4biHist6', history_bi6)
save_pickle('4bi6_time', bi6_time)