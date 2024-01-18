# Задание Вычислить значения линейной функции вида: 3x + 5
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense


def f(x): return 3 * x + 5


X = [x for x in range(-10000, 10000)]
Y = [f(x) for x in X]

net = keras.Sequential()
net.add(Dense(units=1, input_shape=(1,), activation=keras.activations.linear))
net.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.Adam(0.1))

callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=3, restore_best_weights=True)
log = net.fit(X, Y, epochs=len(X), verbose=True, callbacks=[callback])
losses = log.history['loss']

print('Обучение завершено')
print(f"Параметры обученной нейросети : {net.get_weights()}")

test_x = [-10, 3.5, 100]
predicted = [z[0] for z in net.predict(test_x)]
calculated = [f(i) for i in test_x]

print("Сравнение расчета фунукции нейросетью и аналитически:")
print(f"Нейросеть: {predicted}")
print(f"Функция:   {calculated}")

plt.plot(losses)
plt.grid(True)
plt.show()
