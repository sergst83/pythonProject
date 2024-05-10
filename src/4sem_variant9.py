import os

from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.datasets import cifar10  # библиотека базы выборок cifar10
from tensorflow.keras.layers import Dense, Flatten, Dropout

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["TF_GPU_VISIBLE_DEVICES"] = "1"

weights = 'weights.h5'
num_class = 10
epochs = 50
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


def create_model(with_dropout: bool):
    model = Sequential()
    vgg_model = VGG16(
        include_top=False,
        classes=num_class,
        input_shape=x_train[0].shape  # input: 32x32 images with 3 channels -> (32, 32, 3) tensors.
    )
    for layer in vgg_model.layers:
        model.add(layer)

    model.add(Flatten())
    model.add(Dense(512, activation='relu', name='hidden1'))
    if with_dropout:
        model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu', name='hidden2'))
    if with_dropout:
        model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax', name='predictions'))

    model.summary()

    # For a multi-class classification problem
    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.001, momentum=0.9),
                  metrics=['accuracy'])

    return model


def lr_scheduler(epoch, learning_rate):
    return 0.001 * (0.5 ** (epoch // 20))


logs = []
for with_dropuot in (True, False):
    model = create_model(with_dropuot)
    # train the model
    log = model.fit(
        x_train,
        y_train_cat,
        batch_size=batch_size,
        validation_split=0.2,
        epochs=epochs,
        callbacks=[
            LearningRateScheduler(schedule=lr_scheduler, verbose=True),
            ModelCheckpoint(filepath=weights, monitor='val_accuracy', save_best_only=True, mode='max'),
            # EarlyStopping(monitor='loss', min_delta=0.0001, patience=3, restore_best_weights=True)
        ]
    )
    logs.append(log)

    # We load the best weights saved by the ModelCheckpoint
    model.load_weights(weights)

    train_loss, train_accuracy = model.evaluate(x_train, y_train_cat, batch_size=batch_size, steps=156)
    print('Training loss: {}\nTraining accuracy: {}'.format(train_loss, train_accuracy))

    test_loss, test_accuracy = model.evaluate(x_test, y_test_cat)
    print('Testing loss: {}\nTesting accuracy: {}'.format(test_loss, test_accuracy))

plt.plot(logs[0].history['loss'], label='with dropout loss')
plt.plot(logs[0].history['accuracy'], label='with dropout accuracy')
plt.plot(logs[1].history['loss'], label='without dropout loss')
plt.plot(logs[1].history['accuracy'], label='without dropout accuracy')
plt.legend()
plt.grid(True)
plt.show()
