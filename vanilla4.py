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
from tensorflow.keras.layers import SimpleRNN, Dense, Activation
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


def build_vanilla(neurons, lr):
    model = Sequential()

    for i, n in enumerate(neurons):
        # return_sequences for all layer, instead of last layer
        return_seq = i < len(neurons) - 1
        model.add(SimpleRNN(n, activation='tanh', return_sequences=return_seq))

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr))
    return model


def save_pickle(title, variable):
  with open(title, "wb") as file_pi:
    pickle.dump(variable, file_pi)

# test_train_split data opslaan
save_pickle("vanilla4_xtrain", X_train)
save_pickle("vanilla4_xtest", X_test)
save_pickle("vanilla4_ytrain", y_train)
save_pickle("vanilla4_ytest", y_test)
save_pickle("vanilla4_indtrain", ind_train)
save_pickle("vanilla4_indtest", ind_test)
save_pickle("vanilla4_dataset", dataset)

def open_pickle(file):
    with open(file, 'rb') as file_pi:
        return pickle.load(file_pi)

# X_train = open_pickle("vanilla4_xtrain")
# X_test = open_pickle("vanilla4_xtest")
# y_train = open_pickle("vanilla4_ytrain")
# y_test = open_pickle("vanilla4_ytest")
# ind_train = open_pickle("vanilla4_indtrain")
# ind_test = open_pickle("vanilla4_indtest")
# dataset = open_pickle("vanilla4_dataset")

epochs = 200

vanilla1 = build_vanilla([32], 0.0001)
start = time.time()
history_vanilla1 = vanilla1.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
vanilla1_time = end - start
vanilla1.save('4vanilla1.keras')
save_pickle('4vanillaHist1', history_vanilla1)
save_pickle('4vanilla1_time', vanilla1_time)


vanilla2 = build_vanilla([64, 32], 0.0001)
start = time.time()
history_vanilla2 = vanilla2.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
vanilla2_time = end - start
vanilla2.save('4vanilla2.keras')
save_pickle('4vanillaHist2', history_vanilla2)
save_pickle('4vanilla2_time', vanilla2_time)

vanilla3 = build_vanilla([128, 64, 32], 0.0001)
start = time.time()
history_vanilla3 = vanilla3.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
vanilla3_time = end - start
vanilla3.save('4vanilla3.keras')
save_pickle('4vanillaHist3', history_vanilla3)
save_pickle('4vanilla3_time', vanilla3_time)

vanilla4 = build_vanilla([256, 128, 64, 32], 0.0001)
start = time.time()
history_vanilla4 = vanilla4.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
vanilla4_time = end - start
vanilla4.save('4vanilla4.keras')
save_pickle('4vanillaHist4', history_vanilla4)
save_pickle('4vanilla4_time', vanilla4_time)

vanilla5 = build_vanilla([512, 256, 128, 64, 32], 0.0001)
start = time.time()
history_vanilla5 = vanilla5.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
vanilla5_time = end - start
vanilla5.save('4vanilla5.keras')
save_pickle('4vanillaHist5', history_vanilla5)
save_pickle('4vanilla5_time', vanilla5_time)

vanilla6 = build_vanilla([1024, 512, 256, 128, 64, 32], 0.0001)
start = time.time()
history_vanilla6 = vanilla6.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
end = time.time()
vanilla6_time = end - start
vanilla6.save('4vanilla6.keras')
save_pickle('4vanillaHist6', history_vanilla6)
save_pickle('4vanilla6_time', vanilla6_time)