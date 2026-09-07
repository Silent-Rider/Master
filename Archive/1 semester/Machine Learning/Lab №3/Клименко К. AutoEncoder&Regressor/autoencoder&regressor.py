import numpy as np
import pandas as pd
from keras import Input, Model
from keras.src.layers import Dense
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def generate_data(n:float = 10000) -> DataFrame:
    x = np.random.normal(3, 10, (n, 1))
    e = np.random.normal(0, 0.3, (n, 1))

    data = np.hstack([
        x**2 + e,
        np.sin(x / 2) + e,
        np.cos(2 * x) + e,
        x - 3 + e,
        -x + e,
        np.abs(x) + e,
        (x**3) / 4 + e
    ])

    # F - сокращение от feature (признак)
    data_frame = pd.DataFrame(
        data,
        columns=[f'F{(i + 1)}' for i in range(len(data[0]))]
    )
    return data_frame


def create_general_model() -> Model:
    input = Input(shape=(6,))

    #Encoder
    enc = Dense(12, activation='relu')(input)
    enc = Dense(6, activation='relu')(enc)
    enc = Dense(3, activation='relu', name='encoder')(enc)

    #Decoder
    dec = Dense(12, activation='relu', name='decoder_1')(enc)
    dec = Dense(9, activation='relu', name='decoder_2')(dec)
    dec = Dense(6, activation='linear', name='decoder')(dec)

    #Regressor
    regr = Dense(12, activation='relu')(enc)
    regr = Dense(6, activation='relu')(regr)
    regr = Dense(1, activation='linear', name='regressor')(regr)

    general_model = Model(inputs=input, outputs=[dec, regr])
    general_model.compile(optimizer='adam',
                          loss={'decoder':'mse', 'regressor':'mse'},
                          loss_weights={'decoder': 0.5, 'regressor': 1.0})
    return general_model

#Обучение общей комплексной модели с двумя output
np.random.seed(24)
FEATURE_NAMES = ['F1', 'F2', 'F3', 'F4', 'F6', 'F7']

df = generate_data()
X = df.drop(columns=['F5']).values
Y = df['F5'].values.reshape(len(X), 1)
split_index = int(len(X) * 0.8)

x_train = X[:split_index]
x_val = X[split_index:]
y_train = Y[:split_index]
y_val = Y[split_index:]

model = create_general_model()
history = model.fit(x_train,
          {'decoder':x_train, 'regressor':y_train},
          epochs=100,
          batch_size=32,
          validation_data=(x_val, {'decoder': x_val, 'regressor': y_val}))

#Разделение моделей

np.random.seed(45)

df_original = generate_data()
df_original.to_csv('original_dataset.csv', index=False)
X = df_original.drop(columns=['F5']).values

#Encoder
encoder = Model(inputs=model.input, outputs=model.get_layer('encoder').output)
encoder.save('encoder_model.h5')
encoder_pred = encoder.predict(X)
df_encoder = pd.DataFrame(encoder_pred, columns=[f'E{(i + 1)}' for i in range(len(encoder_pred[0]))])
df_encoder.to_csv('encoded_dataset.csv', index=False)

#Decoder
decoder_input = Input(shape=(3,))
dec1 = model.get_layer('decoder_1')(decoder_input)
dec2 = model.get_layer('decoder_2')(dec1)
dec_out = model.get_layer('decoder')(dec2)
decoder = Model(inputs=decoder_input, outputs=dec_out)
decoder.save('decoder_model.h5')
decoder_pred = decoder.predict(encoder_pred)
df_decoder = pd.DataFrame(decoder_pred, columns=FEATURE_NAMES)
df_decoder.to_csv('decoded_dataset.csv', index=False)

#Regressor
regressor = Model(inputs=model.input, outputs=model.get_layer('regressor').output)
regressor.save('regressor_model.h5')
regressor_pred = regressor.predict(X)
df_regressor = pd.DataFrame(regressor_pred, columns=['F5'])
df_regressor.to_csv('regressed_dataset.csv', index=False)

print("\n=== Примеры автоэнкодирования (первые 3 объекта) ===")
print("F1\tF2\tF3\tF4\tF6\tF7 | → | восст.F1\tвосст.F2\t...\n" + "-"*60)
for i in range(3):
    orig = X[i]
    rec = decoder_pred[i]
    print(f"{orig[0]:.2f}\t{orig[1]:.2f}\t{orig[2]:.2f}\t{orig[3]:.2f}\t{orig[4]:.2f}\t{orig[5]:.2f} "
          f"| → | {rec[0]:.2f}\t{rec[1]:.2f}\t{rec[2]:.2f}\t{rec[3]:.2f}\t{rec[4]:.2f}\t{rec[5]:.2f}")

# --- 2. Регрессия: "должно быть" vs "получилось" (предсказание) ---
y_true_full = df_original['F5'].values
y_pred_full = regressor_pred.flatten()

print("\n Примеры регрессии (первые 5 объектов) ")
print("Истинный F5\tПредсказанный F5\tОшибка")
print("-" * 40)
for i in range(5):
    err = abs(y_true_full[i] - y_pred_full[i])
    print(f"{y_true_full[i]:.3f}\t\t{y_pred_full[i]:.3f}\t\t{err:.3f}")

# --- Визуализация регрессии: scatter + линия идеала ---
plt.figure(figsize=(6, 5))
plt.scatter(y_true_full, y_pred_full, alpha=0.6, s=12, color='tab:blue', label='Предсказания')
plt.plot([y_true_full.min(), y_true_full.max()],
         [y_true_full.min(), y_true_full.max()],
         'r--', lw=1.5, label='y = x (идеал)')
plt.xlabel('Истинный F5')
plt.ylabel('Предсказанный F5')
plt.title('Регрессия: качество финальной модели')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Доп: метрика MAE / MSE для регрессии (на всякий) ---
mae = mean_absolute_error(y_true_full, y_pred_full)
mse = mean_squared_error(y_true_full, y_pred_full)
r2_score = r2_score(y_true_full, y_pred_full)
print(f"\nИтоговые метрики регрессии (весь датасет):\nMAE = {mae:.4f}, MSE = {mse:.4f}, R2_score = {r2_score:.4f}")