import random

from keras import models, layers
from matplotlib import pyplot as plt

import gens
import numpy as np

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


def draw_plots(history_dict):
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'o', color='orange', label='Training loss')
    plt.plot(epochs, val_loss_values, 'g', label='Validation loss')
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    accuracy_values = history_dict['accuracy']
    val_accuracy_values = history_dict['val_accuracy']
    plt.clf()
    plt.plot(epochs, accuracy_values, 'o', color='orange', label='Training accuracy')
    plt.plot(epochs, val_accuracy_values, 'g', label='Validation accuracy')
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


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
H = model.fit(train_data[:split_index],
          train_labels[:split_index],
          batch_size=32,
          epochs=15,
          validation_data=(train_data[split_index:], train_labels[split_index:]))

draw_plots(H.history)

size = 7
predictions = model.predict(test_data)
predictions = np.where(predictions >= 0.5, 'Square', 'Circle')[:size]
test_labels = np.where(test_labels >= 0.5, 'Square', 'Circle')[:size]

print('Expected:')
print('\t'.join(f'{i}. {v}' for i, v in enumerate(test_labels)))
print('Predictions:')
print('\t'.join(f'{i}. {v}' for i, v in enumerate(predictions)))

# --- Примеры изображений из обучающей выборки ---
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(train_data[i].squeeze(), cmap='gray')
    plt.title(f'{"Square" if train_labels[i] == 1 else "Circle"}')
    plt.axis('off')

# + 5 из тестовой (после предсказаний — с истиной и предсказанием)
plt.subplot(2, 5, 6)
plt.imshow(test_data[0].squeeze(), cmap='gray')
plt.title(f'Тест:\n {test_labels[0]}\n→ {predictions[0]}', color='green' if test_labels[0] == predictions[0] else 'red')
plt.axis('off')

for i in range(1, 5):
    plt.subplot(2, 5, 6 + i)
    plt.imshow(test_data[i].squeeze(), cmap='gray')
    correct = test_labels[i] == predictions[i]
    plt.title(f'{test_labels[i]} → {predictions[i]}',
              color='green' if correct else 'red')
    plt.axis('off')

plt.suptitle('Примеры классифицируемых объектов')
plt.tight_layout()
plt.show()

