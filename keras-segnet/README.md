# keras-segnet
SegNet model for Keras.

### The original articles
- SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation(https://arxiv.org/pdf/1511.00561v2.pdf)


## Settings Used:
Tensorflow 1.4.0 as Backend
Keras 2.1.2
python 3.5
# Use anaconda environment to avoid incompatibility issues with other python versions and Tensorflow backend versions

## Database 
https://polyp.grand-challenge.org/site/Polyp/EtisLarib/
Project Done for the Endoscopic Vision Challenge: Image to be converted to size (224,224,3). Labels also need to be converted to (224,224,1) from the GT(Ground Truth Folder) 

### Train images

```
import os
import glob
import numpy as np
import cv2
from segnet import SegNet, preprocess_input, to_categorical

input_shape = (224, 224, 3)
nb_classes = 2
nb_epoch = 100
batch_size = 4

X, y = load_train() # need to implement, y shape is (None, 224, 224, nb_classes)
X = preprocess_input(X)
Y = to_categorical(y, nb_classes)
model = SegNet(input_shape=input_shape, classes=nb_classes)
model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])
model.fit(X, Y, batch_size=batch_size, nb_epoch=nb_epoch)
```

