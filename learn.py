from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
# from keras.optimizers import Adam
from keras.optimizers import SGD
from createDataset import FaceDataset
import numpy as np
import model
import pandas as pd
import matplotlib.pyplot as plt


class Schedule:

    def __init__(self, init=0.01):
        self.init = init

    def __call__(self, epoch):
        lr = self.init
        for i in range(1, epoch + 1):
            if i % 5 == 0:
                lr *= 0.5
        return lr


def get_schedule_func(init):
    return Schedule(init)


dataset = FaceDataset(['male', 'female'])
dataset.load_data_target()
x = dataset.data
y = dataset.target
n_class = len(set(y))
perm = np.random.permutation(len(y))
x = x[perm]
y = y[perm]

model = model.build_deep_cnn(n_class)
model.summary()
init_learning_rate = 1e-2
opt = SGD(lr=init_learning_rate, decay=0.0, momentum=0.9, nesterov=False)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt, metrics=["acc"])
early_stopping = EarlyStopping(
    monitor='val_loss', patience=3, verbose=0, mode='auto')
lrs = LearningRateScheduler(get_schedule_func(init_learning_rate))

hist = model.fit(x, y,
                 batch_size=128,
                 nb_epoch=50,
                 validation_split=0.1,
                 verbose=1,
                 callbacks=[early_stopping, lrs])

plt.style.use('ggplot')

df = pd.DataFrame(hist.history)
df.index += 1
df.index.name = 'epoch'
df[['acc', 'val_acc']].plot(linewidth=2)
plt.savefig('acc_history.pdf')
df[['loss', 'val_loss']].plot(linewidth=2)
plt.savefig('loss_history.pdf')
