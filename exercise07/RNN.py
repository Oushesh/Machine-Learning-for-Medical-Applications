#Partly written by Oushesh Haradhun
#Simple LSTM RNN Network 
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib import rnn

# Training Parameters
learning_rate = 0.001 #0.0001 #Play with learning rate and repor results
training_steps = 10000#2000
batch_size = 128#32
display_step=200

#Load mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Network Parameters
num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
w1 = tf.Variable(tf.random_normal([num_hidden, num_classes])) #Change init function
b1=tf.Variable(tf.random_normal([num_classes]))

### Define RNN cell:

#RNN Cell LSTM Definition
def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], w1) + b1

#outputs=[]
#cell=...
#state = ... #Define state cell via rnn

for time_step in range(timesteps):
    #call cell for each time step
#user either last predicion or avarage all predictions and compare results.
	lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)



### Compute logits with w1 and b1
logits= RNN(X,w1,b1)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")
    acc_total=0.0
    for step in range(0, int(mnist.test.images.shape[0]/batch_size)):
        #Get batches from the mnist.test.images and mnist.test.labels
        acc=sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
        acc_total+=acc
    print("Testing Accuracy:",acc_total/(mnist.test.images.shape[0]/batch_size))
