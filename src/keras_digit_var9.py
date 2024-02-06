import os

from keras.preprocessing import image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# вывод структуры НС в консоль
print(model.summary())

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

# обучаем
model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
# проверяем
model.evaluate(x_test, y_test_cat)

# Читаем изображение в массив
img = image.load_img("digit.png", target_size=(28, 28), color_mode="grayscale", interpolation="box")
img = image.img_to_array(img)
x = np.expand_dims(img, axis=0)

# распознаем
res = model.predict(x)
print( res )
print( np.argmax(res) )

# отображаем изображение
plt.imshow(img, cmap="grey")
plt.show()
