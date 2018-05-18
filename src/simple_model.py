from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from preprocessing import *


x_train, y_train, x_test, y_test, x_valid, y_valid = preprocess2()

input_shape = x_train[0].shape
print(input_shape)

nclass = 12
#nclass = y_train något
epochs = 1
batch_size = 128
model_path = r'../model/'

inp = Input(shape=input_shape)
img_1 = Convolution2D(8, kernel_size=3, activation=activations.relu,kernel_initializer='he_normal')(inp)
img_1 = BatchNormalization()(img_1)
img_1 = Convolution2D(8, kernel_size=3, activation=activations.relu,kernel_initializer='he_normal')(img_1)
img_1 = BatchNormalization()(img_1)
img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
img_1 = Dropout(rate=0.2)(img_1)

img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu,kernel_initializer='he_normal')(img_1)
img_1 = BatchNormalization()(img_1)
img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu,kernel_initializer='he_normal')(img_1)
img_1 = BatchNormalization()(img_1)
img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
img_1 = Dropout(rate=0.2)(img_1)

img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu,kernel_initializer='he_normal')(img_1)
img_1 = BatchNormalization()(img_1)
img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
img_1 = Dropout(rate=0.2)(img_1)

img_1 = Flatten()(img_1)
dense_1 = BatchNormalization()(Dense(128, activation=activations.relu,kernel_initializer='he_normal')(img_1))
dense_1 = BatchNormalization()(Dense(128, activation=activations.relu,kernel_initializer='he_normal')(dense_1))
dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

model = models.Model(inputs=inp, outputs=dense_1)
opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer=opt, loss=losses.categorical_crossentropy,metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size, validation_data=(x_valid, y_valid),
          epochs=epochs, shuffle=True, verbose=1)


print("***Evalute model***")
score = model.evaluate(x_test, y_test, verbose=1)
print(score)


model.save(os.path.join(model_path, 'cnn.model'))