import math
import random

import numpy as np
from keras import models, layers
from matplotlib import pyplot as plt


def gen_sequence(seq_len = 1000):
    seq = [math.sin(i/5)/2 + math.cos(i/3)/2 + random.normalvariate(0, 0.04) for i in range(seq_len)]
    return np.array(seq)

def gen_data_from_sequence(seq_len = 1000, lookback = 10):
    seq = gen_sequence(seq_len)
    past = np.array([[[seq[j]] for j in range(i,i+lookback)] for i in range(len(seq) - lookback)])
    future = np.array([[seq[i]] for i in range(lookback,len(seq))])
    return past, future

def get_split_index(dataset):
    dataset_size = len(dataset)
    return int(dataset_size * 0.8)

data, result = gen_data_from_sequence()
split_index = get_split_index(data)

train_data, train_result = data[:split_index], result[:split_index]
test_data, test_result = data[split_index:], result[split_index:]

model = models.Sequential([
    layers.Input(shape=data.shape[1:]),
    layers.LSTM(32, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(32, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(
    train_data,
    train_result,
    epochs=50,
    batch_size=32,
    validation_split=0.125
)

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(range(len(loss)), loss, label='Training loss')
plt.plot(range(len(val_loss)), val_loss, label='Validation loss')
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

predicted_result = model.predict(test_data)
pred_length = range(len(predicted_result))
plt.plot(pred_length, predicted_result, label='Predicted_result')
plt.plot(pred_length, test_result, label='Test result')
plt.title("Predicted and test result")
plt.xlabel("Time")
plt.ylabel("Result")
plt.legend()
plt.show()
