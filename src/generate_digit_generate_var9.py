import os.path

import numpy as np
from PIL import Image, ImageDraw
from keras import Sequential
from keras.src.layers import GRU, TimeDistributed, Activation, Dense
from keras.src.saving.saving_api import load_model
from keras.src.utils import to_categorical, array_to_img
from skimage import transform
from skimage.util import img_as_ubyte
from tensorflow import keras

PATH = 'trained'

# загружаем датасет
mnist = keras.datasets.mnist

# параметры модели
target_size = 10        # определяет ширину изображения после его уменьшения
max_intensity = 6       # ограничивает диапазон интенсивности пикселей. Например, если задано значение 6, все пиксели будут находиться в диапазоне 0-6 вместо 0-255.
units = 256             # размер пачки

num_images = 1000       # количество изображений в обучающей выборке
epochs = 10             # количество эпох

num_classes = max_intensity + 1     # устанавливает длину однократных векторных кодировок для каждого пикселя
factor = units / max_intensity      # используется для разделения интенсивности каждого пикселя таким образом, чтобы значения интенсивности попадали в диапазон от 0 до "max_intensity"


def shrink(im, size):
    return img_as_ubyte(transform.resize(im, (size, size)))


def shrink_all(x, size):
    resized = []
    for i in range(len(x)):
        im = shrink(x[i], size)
        resized.append(im)

    return np.array(resized)


def sample(model, image_size, num_classes):
    pixels = np.zeros((image_size ** 2, 1), dtype=np.uint8)
    for i in range(1, image_size ** 2):
        prefix = to_categorical(pixels[:i].reshape(1, i), num_classes=num_classes).reshape(1, i, num_classes)
        probs = model.predict(prefix)[0][-1]
        indices = list(range(num_classes))
        pixels[i] = np.random.choice(indices, p=probs)

    return pixels


# строим модель
def train(digit, num_images, epochs):
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train[(y_train == digit)]
    x_train = x_train[:num_images]

    # подготовка
    images = shrink_all(x_train, target_size)
    images = np.array(np.round(images / factor), dtype=np.uint8)
    h, w = images[0].shape
    xs = to_categorical(images, num_classes=num_classes).reshape(-1, h ** 2, num_classes)
    xs = np.hstack((np.zeros((len(images), 1, num_classes)), xs[:, 0:, :]))
    ys = np.hstack((xs[:, 1:, :], np.zeros((len(xs), 1, num_classes))))

    # создаем модель
    model = Sequential()
    model.add(GRU(units=units, input_shape=(None, num_classes), return_sequences=True))
    model.add(TimeDistributed(Dense(units=num_classes)))
    model.add(Activation(activation='softmax'))

    # тренируем модель
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(xs, ys, batch_size=num_classes, epochs=epochs)

    # сохраняем натренированную модель для числа
    model.save(PATH + '/mnist_models/model_{}.h5'.format(digit))


# генерируем изображение чистла таблица 5x5
def generate_grid(digit, output_size=64, grid_size=5):
    model = load_model('trained/mnist_models/model_{}.h5'.format(digit))

    image_array = np.zeros((output_size * grid_size, output_size * grid_size), dtype=np.uint8)

    im = Image.fromarray(image_array, mode='L')
    canvas = ImageDraw.ImageDraw(im)

    for i in range(grid_size):
        for j in range(grid_size):
            pixels = sample(model, target_size, num_classes)

            image_array = np.array(pixels * factor, dtype=np.uint8)

            image_array = image_array.reshape((target_size, target_size, 1))
            image = array_to_img(shrink(image_array, size=output_size))
            canvas.bitmap((j * output_size, i * output_size), image, fill=255)

    im.show()


# Число для генерирования картинки
digit_to_generate = 8

# тренируем модели для каждого числа
if not os.path.exists(PATH + '/mnist_models/model_{}.h5'.format(digit_to_generate)):
    # тренируем модель
    # for i in range(10):
    #    train(digit=i, num_images=num_images, epochs=epochs)
    #    print('Model for digit {} is finished'.format(i))
    train(digit=digit_to_generate, num_images=num_images, epochs=epochs)
    print('Model for digit {} is finished'.format(digit_to_generate))

# генерируем числа
generate_grid(digit_to_generate)
