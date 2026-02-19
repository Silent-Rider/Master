import random

from keras import models, layers

import gens
import numpy as np

from laboratory_works.lab4.callback import BestModelsCallback


def gen_data(size=500, img_size=50):
    c1 = size // 2
    c2 = size - c1

    label_c1 = np.full([c1, 1], 'Square')
    data_c1 = np.array([gens.gen_rect(img_size) for i in range(c1)])
    label_c2 = np.full([c2, 1], 'Circle')
    data_c2 = np.array([gens.gen_circle(img_size) for i in range(c2)])

    x = np.vstack((data_c1, data_c2))
    y = np.vstack((label_c1, label_c2))

    return x, y

data, labels = gen_data()
labels = np.where(labels == 'Square', 1, 0)

random.seed(42)
combined = list(zip(data, labels))
random.shuffle(combined)
data, labels = zip(*combined)
data = np.array(data)
labels = np.array(labels)

split_index = int(len(data) * 0.9)
train_data = data[:split_index]
test_data = data[split_index:]
train_labels = labels[:split_index]
test_labels = labels[split_index:]

model = models.Sequential([
    layers.Input(shape=(50,50,1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(514, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

split_index = int(len(train_data) * 0.9)

callback = BestModelsCallback(
    filepath_prefix='model',
    monitor='val_accuracy',
    mode='max',
    max_models=3
)

model.fit(train_data[:split_index],
          train_labels[:split_index],
          batch_size=32,
          epochs=15,
          callbacks=[callback],
          validation_data=(train_data[split_index:], train_labels[split_index:]))

