import tensorflow as tf

import keras

from keras.layers import Input,Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

import os

import numpy as np

from keras.models import Model

from keras.datasets import cifar10

import matplotlib.pyplot as plt


batch_size = 32

num_classes = 10

epochs = 100

save_dir = os.path.join(os.getcwd(),'saved_models')

model_name = 'keras_cifar10_trained_model.h5'



(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('x_train shape:', x_train.shape)

#print(y_train)

y_train = keras.utils.to_categorical(y_train , num_classes)

y_test = keras.utils.to_categorical(y_test , num_classes)

print(y_train)



######################################################## MODEL ###################################################

X_inp = Input(shape= x_train.shape[1:])

conv2D_1 = Conv2D(32,(3,3) , padding = 'same')(X_inp)

act_1 = Activation('relu')(conv2D_1)

conv2D_2 = Conv2D(32,(3,3))(act_1)

act_2 = Activation('relu')(conv2D_2)

pool_1 = MaxPooling2D(pool_size = (2,2))(act_2)

drop_1 = Dropout(0.25)(pool_1)


conv2D_3 = Conv2D(64,(3,3) , padding = 'same')(drop_1)

act_3 = Activation('relu')(conv2D_3)

conv2D_4 = Conv2D(64,(3,3))(act_3)

act_4 = Activation('relu')(conv2D_4)

pool_2 = MaxPooling2D(pool_size = (2,2))(act_4)

drop_2 = Dropout(0.25)(pool_2)


flat_1 = Flatten()(drop_2)

dense_1 = Dense(512)(flat_1)

act_5 = Activation('relu')(dense_1)

drop_3 = Dropout(0.5)(act_5)

dense_2 = Dense(num_classes)(drop_3)

act_6 = Activation('softmax')(dense_2)


######################################################## MODEL ###################################################


model = Model(inputs=X_inp, outputs=act_6)


model.summary()


x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255


opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)


model.compile(loss='categorical_crossentropy' , optimizer = opt,metrics = ['accuracy'])


cnn = model.fit(x_train,y_train,batch_size = batch_size , epochs = epochs, validation_data = (x_test , y_test), shuffle = True)



if not os.path.isdir(save_dir):

    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)

model.save(model_path)

print('Saved trained model at %s ' % model_path)


# Score trained model.

scores = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])


################################################################# PLOT ############################################################


plt.figure(0)

plt.plot(cnn.history['acc'],'r')

plt.plot(cnn.history['val_acc'],'g')

plt.xticks(np.arange(0, 101, 2.0))

plt.rcParams['figure.figsize'] = (8, 6)

plt.xlabel("Num of Epochs")

plt.ylabel("Accuracy")

plt.title("Training Accuracy vs Validation Accuracy")

plt.legend(['train','validation'])

 

plt.savefig('accuracy.png')

 

plt.figure(1)

plt.plot(cnn.history['loss'],'r')

plt.plot(cnn.history['val_loss'],'g')

plt.xticks(np.arange(0, 101, 2.0))

plt.rcParams['figure.figsize'] = (8, 6)

plt.xlabel("Num of Epochs")

plt.ylabel("Loss")

plt.title("Training Loss vs Validation Loss")

plt.legend(['train','validation'])

plt.show()


plt.savefig('loss.png')


