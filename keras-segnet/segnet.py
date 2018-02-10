# -*- coding: utf-8 -*-
"""SegNet model for Keras.
# Reference:
- [Segnet: A deep convolutional encoder-decoder architecture for image segmentation](https://arxiv.org/pdf/1511.00561.pdf)
"""
from __future__ import absolute_import
from __future__ import print_function
import os
import numpy as np

import matplotlib 
from matplotlib import pyplot as plt

import keras
import keras.backend as K
from keras.layers import Input
from keras.layers.core import Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.utils import np_utils
from keras.applications import imagenet_utils

#####CLASS for Loss Curve
############################################################################################
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()
###########################################################################################
def preprocess_input(X):
    return imagenet_utils.preprocess_input(X)

def to_categorical(y, nb_classes):
    num_samples = len(y)
    Y = np_utils.to_categorical(y.flatten(), nb_classes)
    return Y.reshape((num_samples, int(y.size/num_samples), nb_classes))



Background = [0,0,0]
Polyp = [255,255,255]

label_colours = np.array([Background,Polyp])
def visualize(temp, plot=True):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0,1):
        r[temp==l]=label_colours[l,0]
        g[temp==l]=label_colours[l,1]
        b[temp==l]=label_colours[l,2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = (r/255.0)#[:,:,0]
    rgb[:,:,1] = (g/255.0)#[:,:,1]
    rgb[:,:,2] = (b/255.0)#[:,:,2]
    if plot:
        plt.imshow(rgb)
    else:
        return rgb

#############################################################################################
#Implementation of dice_coefficient
smooth =1.0
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

############################################################################################


#TRAINING Mode
#Input Shape of 224 x 224 like for VGG
input_shape = (224, 224, 3)
nb_classes = 2 #Binary Class Classification 0s for Background and 1s for the polyp
nb_epoch = 1#100
batch_size = 4
img_h = 224
img_w = 224
n_labels=2


#Definition of Segnet
#Define model
model = Sequential()

#Encoder Part
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

#Decoder Part
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(UpSampling2D(size=(2, 2),dim_ordering='default'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(UpSampling2D(size=(2, 2),dim_ordering='default'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(UpSampling2D(size=(2, 2),dim_ordering='default'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(nb_classes,(1,1),activation ='softmax'))


def SegNet(input_shape=(224, 224, 3), classes=2):
    # c.f. https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/bayesian_segnet_camvid.prototxt
    img_input = Input(shape=input_shape)
    x = img_input
    # Encoder
    x = Conv2D(64, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(128, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(256, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(512, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # Decoder
    x = Conv2D(512, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(classes, 1, 1, border_mode="valid")(x)
    x = Reshape((input_shape[0]*input_shape[1], classes))(x)
    x = Activation("softmax")(x)
    model = Model(img_input, x)
    return model

#LOAD DATA Here
#Quick Loading Data here:
X=np.load('training_images.npy')
print (X.shape)
y=np.load('training_annotations.npy')
print (y.shape)
#For all Ground Truth images available.
#normalise the labels
y= y/255
#y = np.reshape(y,(y.shape[0]*y.shape[1]*y.shape[2]*y.shape[3]))
print (y.shape,'size of preprocessed labels')
X_test=np.load('validation_images.npy')
print (X_test.shape,'size of testing dataset')
y_test=np.load('validation_annotations.npy')
print (y_test.shape,'size of testing labels')
y_test=y_test/255

#X, y = load_train() # need to implement, y shape is (None, 360, 480, nb_classes)
#X = preprocess_input(X)
Y = to_categorical(y, nb_classes=2)
Y_test = to_categorical(y_test,nb_classes=2)

model = SegNet(input_shape=input_shape, classes=nb_classes)

#TODO: modifications with different Optimizer Parameters. Adam Optimizers: 
#keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)


#WEIGHTED CROSS ENTROPY
#Count the Frequency of the labels for both classes
weights = np.array([0.5,10]) #Class for Background and 

#kld = KLD = kullback_leibler_divergence
model.compile(optimizer='adadelta',loss=dice_coef_loss, metrics=[dice_coef])
#model.compile(loss="kullback_leibler_divergence", optimizer='adadelta', metrics=["accuracy"])
#model.compile(loss="binary_crossentropy", optimizer='adadelta', metrics=["accuracy"])
total_score =model.fit(X, Y, batch_size=batch_size, nb_epoch=nb_epoch)#,callbacks=[plot_losses],verbose=0)

#Testing the model
score = model.evaluate(X_test, Y_test, batch_size=4, verbose=0)
print ('Test score:', score[0])
print ('Test accuracy:', score[1])
output = model.predict(X_test, batch_size=4, verbose=1)

#Get the Prediction Map
#output = model.predict_proba(X_test, verbose=2)
print(type(output))
print (output)
print(output.shape,'shape of prediction output')
pred = visualize(np.argmax(output[0],axis=1).reshape((224,224)), False)
plt.imshow(pred)


