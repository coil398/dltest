from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.models import Sequential


def build_deep_cnn(num_classes=3):
    model = Sequential()

    model.add(Convolution2D(196, 3, 3, border_mode='same', input_shape=(1, 112, 112)))
    model.add(Activation('relu'))

    model.add(Convolution2D(256, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(512, 9, 9, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(512, 9, 9))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model
