# Задание Вычислить значения линейной функции вида: 3x + 5
import os

import matplotlib.pyplot as plt
from tensorflow import keras
from keras.src.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.src.optimizers import SGD
from keras.src.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.datasets import cifar10  # библиотека базы выборок cifar10

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_class = 10
epochs = 100
batch_size = 128

# данные
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# стандартизация входных данных
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train_cat = to_categorical(y_train, num_class)
y_test_cat = to_categorical(y_test, num_class)

# # строим сеть
# model = Sequential([
#     Conv2D(32, (3, 3), padding="same", activation="relu",  data_format='channels_last', input_shape=x_train[0].shape),
#     MaxPooling2D((2, 2), strides=2),
#     Conv2D(64, (3, 3), padding='same', activation='relu'),
#     MaxPooling2D((2, 2), strides=2),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dense(512, activation='relu'),
#     Dropout(0.5),
#     Dense(num_class,  activation='softmax')
# ])
# callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=3, restore_best_weights=True)
#
# print(model.summary())      # вывод структуры НС в консоль
#
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
# )
#
# log = model.fit(x_train, y_train_cat, batch_size=batch_size, epochs=epochs, verbose=True, callbacks=[callback], validation_split=0.2)
# losses = log.history['loss']
#
# print('Обучение завершено')
# print(f"Параметры обученной нейросети : {model.get_weights()}")

# model.evaluate(x_test, y_test_cat)

# plt.plot(losses)
# plt.grid(True)
# plt.show()


vgg16_model = VGG16(weights='imagenet',
                    include_top=False,
                    classes=num_class,
                    input_shape=x_train[0].shape  # input: 32x32 images with 3 channels -> (32, 32, 3) tensors.
                    )

model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)

model.add(Flatten())
model.add(Dense(512, activation='relu', name='hidden1'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu', name='hidden2'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax', name='predictions'))

model.summary()

# For a multi-class classification problem
model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.001, momentum=0.9), metrics=['accuracy'])


def lr_scheduler(epoch, learning_rate):
    return 0.001 * (0.5 ** (epoch // 20))


# construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# train the model
history = model.fit(
    # aug.flow(x_train, y_train_cat, batch_size=batch_size),
    x_train,
    y_train_cat,
    batch_size=batch_size,
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // batch_size,
    epochs=epochs,
    callbacks=[
        LearningRateScheduler(schedule=lr_scheduler, verbose=True),
        ModelCheckpoint(filepath='./weights.h5', monitor='val_accuracy', save_best_only=True, mode='max'),
        EarlyStopping(monitor='loss', min_delta=0.01, patience=3, restore_best_weights=True)
    ]
)

# We load the best weights saved by the ModelCheckpoint
model.load_weights('./weights.h5')

train_loss, train_accuracy = model.evaluate(x_train, y_train_cat, batch_size=batch_size, steps=156)
print('Training loss: {}\nTraining accuracy: {}'.format(train_loss, train_accuracy))

val_loss, val_accuracy = model.evaluate(x_test, y_test)
print('Validation loss: {}\nValidation accuracy: {}'.format(val_loss, val_accuracy))

test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, )
print('Testing loss: {}\nTesting accuracy: {}'.format(test_loss, test_accuracy))
