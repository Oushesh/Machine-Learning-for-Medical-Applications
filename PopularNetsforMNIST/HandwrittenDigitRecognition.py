import tensorflow as tf

from tensorflow.contrib.layers import flatten

import numpy as np
import random
import time
import matplotlib.pyplot as plt
#from IPython import display

#Define LeNet here
def LeNet(tensor):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    x=tensor
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Input = 14x14x6, Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 10), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits, tf.nn.softmax(logits)



#Definition of VGGNet here
def VGGNet(tensor):
     # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    #tensor = np.dstack([tensor,tensor,tensor]) #Stack the Data to produce a 3-channel Data with same values in all the 3 channels 
    print (tensor.get_shape())
    #Upsampling to 224x224
    scale=8
    new_height = int(round(28 * scale))
    new_width = int(round(28 * scale))
    x = tf.image.resize_images(tensor, [new_height,new_width])
    print (x.get_shape())
    mu = 0
    sigma = 0.1

    #224/28=8
    # Layer 1_1: Convolutional.   Input = 32x32x1. Output = 30x30x8
    conv1_1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 1, 64), mean = mu, stddev = sigma))
    conv1_1_b = tf.Variable(tf.zeros(64))
    conv1_1   = tf.nn.conv2d(x, conv1_1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_1_b

    # Layer 1_1: Activation.
    conv1_1 = tf.nn.relu(conv1_1)

    # Layer 1_2: Convolutional. Input = 30x30x8  Output= 28x28x
    conv1_2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 64), mean = mu, stddev = sigma))
    conv1_2_b = tf.Variable(tf.zeros(64))
    conv1_2   = tf.nn.conv2d(conv1_1, conv1_2_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_2_b

    # Layer 1_1: Activation.
    conv1_2 = tf.nn.relu(conv1_2)

    # Layer 1: Pooling. 
    conv1_2 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2_1: Convolutional. 
    conv2_1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 128), mean = mu, stddev = sigma))
    conv2_1_b = tf.Variable(tf.zeros(128))
    conv2_1   = tf.nn.conv2d(conv1_2, conv2_1_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_1_b
    
    # Activation.
    conv2_1 = tf.nn.relu(conv2_1)

    # Layer 2_2: Convolutional. 
    conv2_2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 128, 128), mean = mu, stddev = sigma))
    conv2_2_b = tf.Variable(tf.zeros(128))
    conv2_2   = tf.nn.conv2d(conv2_1, conv2_2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_2_b
    
    # Activation.
    conv2_2 = tf.nn.relu(conv2_2)

    # Layer 2: Pooling. Input = 10x10x16. 
    conv2_2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 3_1: Convolutional. 
    #conv3_1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 128, 256), mean = mu, stddev = sigma))
    #conv3_1_b = tf.Variable(tf.zeros(256))
    #conv3_1   = tf.nn.conv2d(conv2_2, conv3_1_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_1_b
    
    # Activation.
    #conv3_1 = tf.nn.relu(conv3_1)


    # Layer 3_2: Convolutional. 
    #conv3_2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 256), mean = mu, stddev = sigma))
    #conv3_2_b = tf.Variable(tf.zeros(256))
    #conv3_2   = tf.nn.conv2d(conv3_1, conv3_2_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_2_b
    
     # Activation.
    #conv3_2 = tf.nn.relu(conv3_2)

    # Layer 3_3: Convolutional. 
    #conv3_3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 256), mean = mu, stddev = sigma))
    #conv3_3_b = tf.Variable(tf.zeros(256))
    #conv3_3   = tf.nn.conv2d(conv3_2, conv3_3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_3_b
    
    # Activation.
    #conv3_3 = tf.nn.relu(conv3_3)

    # Layer 3: Pooling.  
    #conv3_3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 4_1: Convolutional. 
    #conv4_1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 512), mean = mu, stddev = sigma))
    #conv4_1_b = tf.Variable(tf.zeros(512))
    #conv4_1   = tf.nn.conv2d(conv3_3, conv4_1_W, strides=[1, 1, 1, 1], padding='VALID') + conv4_1_b
    
    # Activation.
    #conv4_1 = tf.nn.relu(conv4_1)

    #Layer 4_2: Convolutional. 
    #conv4_2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), mean = mu, stddev = sigma))
    #conv4_2_b = tf.Variable(tf.zeros(512))
    #conv4_2   = tf.nn.conv2d(conv4_1, conv4_2_W, strides=[1, 1, 1, 1], padding='VALID') + conv4_2_b
    
    # Activation.
    #conv4_2 = tf.nn.relu(conv4_2)

    #Layer 4_3: Convolutional. 
    #conv4_3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), mean = mu, stddev = sigma))
    #conv4_3_b = tf.Variable(tf.zeros(512))
    #conv4_3   = tf.nn.conv2d(conv4_2, conv4_3_W, strides=[1, 1, 1, 1], padding='VALID') + conv4_3_b
    
    # Activation.
    #conv4_3 = tf.nn.relu(conv4_3)

    # Layer 3: Pooling.  
    #conv4_3 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #Layer 5_1: Convolutional. 
    #conv5_1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), mean = mu, stddev = sigma))
    #conv5_1_b = tf.Variable(tf.zeros(512))
    #conv5_1   = tf.nn.conv2d(conv4_3, conv5_1_W, strides=[1, 1, 1, 1], padding='VALID') + conv4_3_b
    
    # Activation.
    #conv5_1 = tf.nn.relu(conv5_1)

    #Layer 5_2: Convolutional. 
    #conv5_2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), mean = mu, stddev = sigma))
    #conv5_2_b = tf.Variable(tf.zeros(512))
    #conv5_2   = tf.nn.conv2d(conv5_1, conv5_2_W, strides=[1, 1, 1, 1], padding='VALID') + conv5_2_b
    
    # Activation.
    #conv5_2 = tf.nn.relu(conv5_2)

    #Layer 5_3: Convolutional. 
    #conv5_3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), mean = mu, stddev = sigma))
    #conv5_3_b = tf.Variable(tf.zeros(512))
    #conv5_3   = tf.nn.conv2d(conv5_2, conv5_3_W, strides=[1, 1, 1, 1], padding='VALID') + conv5_3_b
    
    # Activation.
    #conv5_3 = tf.nn.relu(conv5_3)

    # Layer 3: Pooling.  
    #conv5_3 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Fully Connected Layer 1
    fc0   = flatten(conv2_2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120. Use 120 instead of 4096
    shape = int(np.prod(conv2_2.get_shape()[1:]))
    fc1_W = tf.Variable(tf.truncated_normal(shape=(shape, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # Fully Connected Layer 1: Activation.
    fc1    = tf.nn.relu(fc1)

    # Fully Connected Layer 2  #Use 120 instead of 4096
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 120), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(120))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Fully Connected Layer 2: Activation.
    fc2    = tf.nn.relu(fc2)

    # Fully Connected Layer 3 Use 84 instead of 1000
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(84))
    #fc3    =  tf.matmul(fc2, fc3_W) + fc3_b 
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits, tf.nn.softmax(logits)

#RESNET
#Helper Functions
def weights_init(shape):
    '''
    Weights initialization helper function.
    
    Input(s): shape - Type: int list, Example: [5, 5, 32, 32], This parameter is used to define dimensions of weights tensor
    
    Output: tensor of weights in shape defined with the input to this function
    '''
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def bias_init(shape, bias_value=0.01):
    '''
    Bias initialization helper function.
    
    Input(s): shape - Type: int list, Example: [32], This parameter is used to define dimensions of bias tensor.
              bias_value - Type: float number, Example: 0.01, This parameter is set to be value of bias tensor.
    
    Output: tensor of biases in shape defined with the input to this function
    '''
    return tf.Variable(tf.constant(bias_value, shape=shape))
def conv2d_custom(input, filter_size, num_of_channels, num_of_filters, activation=tf.nn.relu, dropout=None,
                  padding='SAME', max_pool=True, strides=(1, 1)):  
    '''
    This function is used to define a convolutional layer for a network,
    
    Input(s): input - this is input into convolutional layer (Previous layer or an image)
              filter_size - also called kernel size, kernel is moved (convolved) across an image. Example: 3
              number_of_channels - how many channels the input tensor has
              number_of_filters - this is hyperparameter, and this will set one of dimensions of the output tensor from 
                                  this layer. Note: this number will be number_of_channels for the layer after this one
              max_pool - if this is True, output tensor will be 2x smaller in size. Max pool is there to decrease spartial 
                        dimensions of our output tensor, so computation is less expensive.
              padding - the way that we pad input tensor with zeros ("SAME" or "VALID")
              activation - the non-linear function used at this layer.
              
              
    Output: Convolutional layer with input parameters.
    '''
    weights = weights_init([filter_size, filter_size, num_of_channels, num_of_filters])
    bias = bias_init([num_of_filters])
    
    layer = tf.nn.conv2d(input, filter=weights, strides=[1, strides[0], strides[1], 1], padding=padding) + bias
    
    if activation != None:
        layer = activation(layer)
    
    if max_pool:
        layer = tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    if dropout != None:
        layer = tf.nn.dropout(layer, dropout)
        
    return layer
def flatten(layer):
    '''
    This method is used to convert convolutional output (4 dimensional tensor) into 2 dimensional tensor.
    
    Input(s): layer - the output from last conv layer in your network (4d tensor)
    
    Output(s): reshaped - reshaped layer, 2 dimensional matrix
               elements_num - number of features for this layer
    '''
    shape = layer.get_shape()
    
    num_elements_ = shape[1:4].num_elements()
    
    flattened_layer = tf.reshape(layer, [-1, num_elements_])
    return flattened_layer, num_elements_
def dense_custom(input, input_size, output_size, activation=tf.nn.relu, dropout=None):
    '''
    This function is used to define a fully connected layer for a network,
    
    Input(s): input - this is input into fully connected (Dense) layer (Previous layer or an image)
              input_size - how many neurons/features the input tensor has. Example: input.shape[1]
              output_shape - how many neurons this layer will have
              activation - the non-linear function used at this layer.    
              dropout - the regularization method used to prevent overfitting. The way it works, we randomly turn off
                        some neurons in this layer
              
    Output: fully connected layer with input parameters.
    '''
    weights = weights_init([input_size, output_size])
    bias = bias_init([output_size])
    
    layer = tf.matmul(input, weights) + bias
    
    if activation != None:
        layer = activation(layer)
    
    if dropout != None:
        layer = tf.nn.dropout(layer, dropout)
        
    return layer
def residual_unit(layer):
    '''
    Input(s): layer - conv layer before this res unit
    
    Output(s): ResUnit layer - implemented as described in the paper
    '''
    step1 = tf.layers.batch_normalization(layer)
    step2 = tf.nn.relu(step1)
    step3 = conv2d_custom(step2, 3, 32, 32, activation=None, max_pool=False) #32 number of feautres is hyperparam
    step4 = tf.layers.batch_normalization(step3)
    step5 = tf.nn.relu(step4)
    step6 = conv2d_custom(step5, 3, 32, 32, activation=None, max_pool=False)
    return layer + step6
num_of_layers = 20
between_strides = num_of_layers/5
def ResNet(tensor):
    inputs=tensor 
    prev1 = conv2d_custom(inputs, 3, 1, 32, activation=None, max_pool=False)
    prev1 = tf.layers.batch_normalization(prev1)
    for i in range(3): # this number * between_strides = number_of_layers
        for j in range(int(between_strides)):
            prev1 = residual_unit(prev1)
    #After 2 res units we perform strides 2x2, which will reduce data
    prev1 = conv2d_custom(inputs, 3, 1, 32, activation=None, max_pool=False, strides=[2, 2])
    prev1 = tf.layers.batch_normalization(prev1)
    #after all resunits we have last conv layer, than flattening and output layer
    last_conv = conv2d_custom(prev1, 3, 32, 10, activation=None, max_pool=False)
    flat, features = flatten(last_conv)
    logits = dense_custom(flat, features, 10, activation=None)
    return logits, tf.nn.softmax(logits)


#input size = 28x28, output size =10 class labels
def old_ResNet(tensor):
    #TODO: 2 residual blocks then connect to FCN
    #Definition of Residual Block
    x=tensor
    mu = 0
    sigma = 0.1

    #RESIDUAL LAYER
    #Residual Block of Depth 1 with 32 Filters, 3x3 Kernels and ReLu activations
    # Layer 1: Convolutional.   
    conv1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 1, 32), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(32))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    # Layer 1: Activation.
    conv1 = tf.nn.relu(conv1)
    #Max Pooling
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #Residual Block of Depth 2 with 64 Filters, 3x3 Kernels and ReLu activations
    conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 64), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(64))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    # Layer 1: Activation.
    conv2 = tf.nn.relu(conv2)
    #Max Pooling
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #Add the residual Layer to the original Input: Reference Deep Residual Learning for Image Recognition Paper
    #x_padded= tf.Variable(tf.zeros())
    #conv2 +=x


    # Fully Connected Layer 1
    fc0   = flatten(conv2)
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120. Use 120 instead of 4096
    shape = int(np.prod(conv2.get_shape()[1:]))
    fc1_W = tf.Variable(tf.truncated_normal(shape=(shape, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # Fully Connected Layer 1: Activation.
    fc1    = tf.nn.relu(fc1)

    # Fully Connected Layer 2  #Use 120 instead of 4096
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Fully Connected Layer 2: Activation.
    fc2    = tf.nn.relu(fc2)

    # Fully Connected Layer 3 Use 84 instead of 1000
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 10), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(10))
    #fc3    =  tf.matmul(fc2, fc3_W) + fc3_b 
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits, tf.nn.softmax(logits)


#CNN Training starts here
def visualizeCurves(curves, handle=None):
    if not handle:
        handle = plt.figure()

    fig = plt.figure(handle.number)
    fig.clear()
    ax = plt.axes()
    plt.cla()

    counter = len(curves[list(curves.keys())[0]])
    x = np.linspace(0, counter, num=counter)
    for key, value in curves.items():
        value_ = np.array(value).astype(np.double)
        mask = np.isfinite(value_)
        ax.plot(x[mask], value_[mask], label=key)
    plt.legend(loc='upper right')
    plt.title("Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    #display.clear_output(wait=True)
    plt.show()
    
def printNumberOfTrainableParams():
    total_parameters = 0
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for variable in variables:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parametes = 1
        for dim in shape:
            #    print(dim)
            variable_parametes *= dim.value
        # print(variable_parametes)
        total_parameters += variable_parametes
    print(total_parameters)




# Config
config = {}
config['batchsize'] = 128
config['learningrate'] = 0.01
config['numEpochs'] = 10



# Download and read in MNIST automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Image Shape: {}".format(mnist.train.images[0].shape))
print()
print("Training Set:   {} samples".format(len(mnist.train.images)))
print("Validation Set: {} samples".format(len(mnist.validation.images)))
print("Test Set:       {} samples".format(len(mnist.test.images)))



# Visualize a sample from MNIST
index = random.randint(0, len(mnist.train.images))
image = mnist.train.images[index].reshape((28, 28))

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")



# Clear Graph
tf.reset_default_graph()

# Define placeholders
inputs = {}
inputs['data'] = tf.placeholder(tf.float32, [None, 32, 32, 1])
inputs['labels'] = tf.placeholder(tf.float32, [None, 10])
inputs['phase'] = tf.placeholder(tf.bool)

# Define a dictionary for storing curves
curves = {}
curves['training'] = []
curves['validation'] = []

# Instantiate the model operations 
#TODO: instantiate the code for ResNet 
#logits, probabilities = LeNet(inputs['data']) # Or VGGNet or ResNet or AlexNet
#logits, probabilities = VGGNet(inputs['data'])
logits, probabilities = ResNet(inputs['data'])
printNumberOfTrainableParams()

# Define loss function in a numerically stable way
# DONT: cross_entropy = tf.reduce_mean(-tf.reduce_sum( * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = inputs['labels']))

# Operations for assessing the accuracy of the classifier
correct_prediction = tf.equal(tf.argmax(probabilities,1), tf.argmax(inputs['labels'],1))
accuracy_operation = tf.cast(correct_prediction, tf.float32)

# Idea: Use different optimizers?
# SGD vs ADAM
#train_step = tf.train.AdamOptimizer(config['learningrate']).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(config['learningrate']).minimize(cross_entropy)



# Create TensorFlow Session and initialize all weights
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# Run!
numTrainSamples = len(mnist.train.images)
numValSamples = len(mnist.validation.images)
numTestSamples = len(mnist.test.images)
for e in range(config['numEpochs']):
    avg_loss_in_current_epoch = 0
    for i in range(0, numTrainSamples, config['batchsize']):
        batch_data, batch_labels = mnist.train.next_batch(config['batchsize'])
        batch_data = batch_data.reshape((batch_data.shape[0], 28, 28, 1))
        
        batch_data = np.pad(batch_data, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
        fetches = {
            'optimizer': train_step,
            'loss': cross_entropy
        }
        results = sess.run(fetches, feed_dict={inputs['data']: batch_data, inputs['labels']: batch_labels})
        avg_loss_in_current_epoch += results['loss']
    avg_loss_in_current_epoch = avg_loss_in_current_epoch / i
    curves['training'] += [avg_loss_in_current_epoch]
        
    for i in range(0, numValSamples, config['batchsize']):
        # Use Matplotlib to visualize the loss on the training and validation set
        batch_data, batch_labels = mnist.validation.next_batch(config['batchsize'])
        batch_data = batch_data.reshape((batch_data.shape[0], 28, 28, 1))
        
        # TODO: Preprocess the images in the batch
        batch_data = np.pad(batch_data, ((0,0),(2,2),(2,2),(0,0)), 'constant')
        
        fetches = {
            'loss': cross_entropy
        }
        results = sess.run(fetches, feed_dict={inputs['data']: batch_data, inputs['labels']: batch_labels})
        avg_loss_in_current_epoch += results['loss']
    avg_loss_in_current_epoch = avg_loss_in_current_epoch / i
    curves['validation'] += [avg_loss_in_current_epoch]
    
    print('Done with epoch %d' % (e))
    visualizeCurves(curves)


# Test
accumulated_predictions = np.array([])
for i in range(0, numValSamples, config['batchsize']):
    # Use Matplotlib to visualize the loss on the training and validation set
    batch_data, batch_labels = mnist.test.next_batch(config['batchsize'])
    batch_data = batch_data.reshape((batch_data.shape[0], 28, 28, 1))
        
    # TODO: Preprocess the images in the batch
    batch_data = np.pad(batch_data, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    fetches = {
        'accuracy': accuracy_operation
    }
    results = sess.run(fetches, feed_dict={inputs['data']: batch_data, inputs['labels']: batch_labels})
    
    if i==0:
        accumulated_predictions = results['accuracy']
    else:
        accumulated_predictions = np.append(accumulated_predictions, results['accuracy'])
accuracy = np.mean(accumulated_predictions)
print(accuracy)
