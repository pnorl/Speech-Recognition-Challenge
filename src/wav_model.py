from keras import optimizers, losses, activations, models
from keras.layers import Convolution1D, Dense, Input, Flatten, Dropout, MaxPooling1D, BatchNormalization
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential

from keras.callbacks import ModelCheckpoint, EarlyStopping,TensorBoard
from time import time
import numpy as np
import os

#Define tensorboard pathway and launch tensorboard
print("Launching tensorboard")
time_stamp=time()
tb_dir = "Graph/{}".format(time_stamp)
tensorboard=TensorBoard(log_dir=tb_dir)

#Fetch preprocessed data
savePath = r'../data/train_preprocessed/'
fileName='raw_wav'
npzfile = np.load(savePath+fileName+'.npz')
x_train, y_train = npzfile['x_train'],npzfile['y_train']
x_test, y_test = npzfile['x_test'],npzfile['y_test']
x_valid, y_valid = npzfile['x_val'],npzfile['y_val']
#x_train, y_train, x_test, y_test, x_valid, y_valid = preprocess2()

#Define params
nclass = y_train.shape[1]
epochs = 10
batch_size = 128 #Lower this if computer runs out of memory
#input_shape = x_train[0].shape
input_shape = x_train.shape[1:]
model_path = r'../model/'
model_checkpoint_path=model_path+'checkpoint/'
nameOfModel='raw_wav_unsampled'+str(time_stamp)

savePath = r'../test_results/' #Save score to


#Define computational graph
inp = Input(shape=input_shape)
img_1 = Convolution1D(8, kernel_size=5,strides=2, activation=activations.relu,kernel_initializer='he_normal')(inp)
img_1 = BatchNormalization()(img_1)
img_1 = Convolution1D(8, kernel_size=5,strides=2, activation=activations.relu,kernel_initializer='he_normal')(img_1)
img_1 = BatchNormalization()(img_1)
img_1 = MaxPooling1D(pool_size=2)(img_1)
img_1 = Dropout(rate=0.2)(img_1)

img_1 = Convolution1D(16, kernel_size=5,strides=2, activation=activations.relu,kernel_initializer='he_normal')(img_1)
img_1 = BatchNormalization()(img_1)
img_1 = Convolution1D(16, kernel_size=5,strides=2, activation=activations.relu,kernel_initializer='he_normal')(img_1)
img_1 = BatchNormalization()(img_1)
img_1 = MaxPooling1D(pool_size=2)(img_1)
img_1 = Dropout(rate=0.2)(img_1)

img_1 = Convolution1D(32, kernel_size=5,strides=1, activation=activations.relu,kernel_initializer='he_normal')(img_1)
img_1 = BatchNormalization()(img_1)
img_1 = MaxPooling1D(pool_size=2)(img_1)
img_1 = Dropout(rate=0.2)(img_1)

img_1 = Flatten()(img_1)
dense_1 = BatchNormalization()(Dense(64, activation=activations.relu,kernel_initializer='he_normal')(img_1))
dense_1 = BatchNormalization()(Dense(64, activation=activations.relu,kernel_initializer='he_normal')(dense_1))
dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

#Define model from computational graph
model = models.Model(inputs=inp, outputs=dense_1)
opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#Comipile model and print summary of graph
model.compile(optimizer=opt, loss=losses.categorical_crossentropy,metrics=['accuracy'])
model.summary()

#Define callbacks for training
callbacks = [
    tensorboard,
    EarlyStopping(
        monitor='val_loss', 
        patience=10,
        mode='max',
        verbose=1),
    ModelCheckpoint(model_checkpoint_path+str(time_stamp)+'weights.hdf5',
        monitor='val_acc', 
        save_best_only=True, 
        mode='max',
        verbose=0)
]

#Train model
print("Training model")
model.fit(x_train,y_train,batch_size=batch_size, 
    validation_data=(x_valid, y_valid),
          epochs=epochs, shuffle=True, verbose=1,callbacks=callbacks)


#Save model to 
model.save(os.path.join(model_path,nameOfModel))

#Evaluate model
print("***Evaluate model***")
score = model.evaluate(x_test, y_test, verbose=1)
print("Loss on test data:",score[0])
print("Acc on test data:",score[1])

with open(savePath+nameOfModel+'.txt','w') as file:
	file.write('Test loss, Test accuracy') 
	file.write(str(score[0])+', '+str(score[1]))

