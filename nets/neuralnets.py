"""

Configuration for different nets.

"""

import h5py
import numpy as np
import tensorflow as tf

from dnninputs import *
# -----------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Helper methods.
# -----------------------------------------------------------------------------
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def max_pool_2x2x2(x):
  return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                           strides=[1, 2, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def max_pool_nrs(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 1, 1, 1], padding='SAME')


def avg_pool_2x2(x):
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# -----------------------------------------------------------------------------
# Set up the neural network.
# -----------------------------------------------------------------------------
def getNet(name,x_input):

    if(name == "MNISTbasic"):
        return MNISTbasic_net(x_input)
    if(name == "MNISTadv"):
        return MNISTadv_net(x_input)
    if(name == "MNISTadv3d"):
        return MNISTadv3d_net(x_input)
    if(name== "MNISTnew"):
        return MNISTnewLS_net(x_input)
# -----------------------------------------------------------------------------
# DNN definitions
# -----------------------------------------------------------------------------
def MNISTbasic_net(x_input):

    # Softmax readout: 2 classifications
    W = weight_variable([3*npix, 2])
    b = bias_variable([2])
    y = tf.nn.softmax(tf.matmul(x_input, W) + b)

    return y


# Function to write for part 2.
def MNISTadv_net(x_input):

    #print "To be written!"
    #return 0 # Delete this line once the correct method has been written

    # Resize the array to pdim x pdim x 3
    x_image = tf.reshape(x_input, [-1,pdim,pdim,nchannels])

    # First convolutional layer
    W_conv1 = weight_variable([5, 5, nchannels, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Densely connected layer
    new_dim = int(pdim / 4)
    W_fc1 = weight_variable([new_dim * new_dim * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, new_dim*new_dim*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = 0.4 #tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Softmax readout
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y

# Function to write for part 2.
def MNISTadv3d_net(x_input):

    print "Creating 3D MNIST net..."

    # Resize the array to pdim x pdim x 3
    x_image = tf.reshape(x_input, [-1,pdim,pdim,pdim,nchannels])

    # First convolutional layer
    W_conv1 = weight_variable([5, 5, 5, nchannels, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2x2(h_conv1)
    print "Created first convolutional layer..."

    # Second convolutional layer
    W_conv2 = weight_variable([5, 5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2x2(h_conv2)
    print "Created second convolutional layer..."

    # Densely connected layer
    new_dim = int(pdim / 4)
    W_fc1 = weight_variable([new_dim * new_dim * new_dim * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, new_dim*new_dim*new_dim*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    print "Created densely connected layer..."

    # Dropout
    keep_prob = 0.4 #tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Softmax readout
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y

def MNISTnewLS_net(x_input):

    #defining aux variables
    #defining number of Max_pooling layers
    MPL = 2
    #define the reduction coefficient
    red_coef= (2**MPL)
    # ------------
    #Parameters for the first convolutional layer
    #-----
    #define number of maps in the first layer
    # possibly a power of 2
    Nmaps_c1 = 32
    #patch size for the first layer
    pat_size1 = 6
    #-----
    #Parameters for the second convolutional layer
    #-----
    #define number of maps in the second convolutional layer
    # possibly a power of 2
    Nmaps_c2 = 64
    #patch size for the second layer
    pat_size2 = 3
    #-----
    #Parameters for the third convolutional layer
    #-----
    #define number of maps in the third convolutional layer
    # possibly a power of 2
    Nmaps_c3 = 64
    #patch size for the third layer
    pat_size3 = 5
    #---------
    #Parameters for the fourth convolutional layer
    #-----
    #define number of maps in the third convolutional layer
    # possibly a power of 2
    Nmaps_c4 = 128
    #patch size for the third layer
    pat_size4 = 3
    #--------
    #Parameters for a block of two parallell convolutional layers
    #--------
    Nmaps_parall1 = 128
    pat_size_p1 = 3
    #--------
    #Parameters for another block of two parallell convolutional layers
    #--------
    Nmaps_parall2 = 128
    pat_size_p2 = 5
    #----------
    #----------
    #if more convolutional layers, define more...
    #--------------
    #define number of maps for last layer (non-conv)
    Nmaps_fc1 = 1024
    Nmaps_fc2 = 2048
    #--------------
    #---------------
    #define keep (non-dropout) probability
    keep_prob = 0.5
    #----------------
    #-------------
    #defining the new image dimension after max_pool
    new_dim = int(pdim/red_coef)
    #define the new dimension
    #resizing input
    x_image = tf.reshape(x_input, [-1,pdim,pdim,3])
    #defining parameters first layer
    W_conv1 = weight_variable([pat_size1,pat_size1,3,Nmaps_c1])
    b_conv1 = bias_variable([Nmaps_c1])
    #makes convolution
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    #-------
    #second layer
    W_conv2 = weight_variable([pat_size2,pat_size2,Nmaps_c1, Nmaps_c2])
    b_conv2 = bias_variable([Nmaps_c2])
    #makes convolution
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    #-------
    #third layer
    W_conv3 = weight_variable([pat_size3,pat_size3,Nmaps_c2, Nmaps_c3])
    b_conv3 = bias_variable([Nmaps_c3])
    #makes convolution
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
    h_pool3 = max_pool_nrs(h_conv3)
    #--------------
    #--------------
    #-----------------
    #intermediate layer: two parallell convolutional layers
    W_conv_p11 = weight_variable([pat_size_p1,pat_size_p1,Nmaps_c3,Nmaps_parall1])
    b_conv_p11 = bias_variable([Nmaps_parall1])
    h_conv_p11 = tf.nn.relu(conv2d(h_conv3, W_conv_p11) + b_conv_p11)
    W_conv_p12 = weight_variable([pat_size_p1,pat_size_p1,Nmaps_c3,Nmaps_parall1])
    b_conv_p12 = bias_variable([Nmaps_parall1])
    h_conv_p12 = tf.nn.relu(conv2d(h_conv3, W_conv_p12) + b_conv_p12)
    #--------
    #----------
    # two blocks of two parallell layers
    #--------
    #first two: get as input the output of previous two
    W_conv_p21 = weight_variable([pat_size_p2,pat_size_p2,Nmaps_parall1,Nmaps_parall2])
    b_conv_p21 = bias_variable([Nmaps_parall2])
    h_conv_p21 = tf.nn.relu(conv2d(h_conv_p11, W_conv_p21) + b_conv_p21)
    W_conv_p22 = weight_variable([pat_size_p2 + 1 ,pat_size_p2 + 1 ,Nmaps_parall1,Nmaps_parall2])
    b_conv_p22 = bias_variable([Nmaps_parall2])
    h_conv_p22 = tf.nn.relu(conv2d(h_conv_p12, W_conv_p22) + b_conv_p22)
    #-------
    # second two: get input of previous single convolutional layer
    W_conv_p23 = weight_variable([pat_size_p1,pat_size_p1,Nmaps_c3,Nmaps_parall2])
    b_conv_p23 = bias_variable([Nmaps_parall2])
    h_conv_p23 = tf.nn.relu(conv2d(h_conv3, W_conv_p23) + b_conv_p23)
    W_conv_p24 = weight_variable([pat_size_p1,pat_size_p1,Nmaps_c3,Nmaps_parall2])
    b_conv_p24 = bias_variable([Nmaps_parall2])
    h_conv_p24 = tf.nn.relu(conv2d(h_pool3, W_conv_p24) + b_conv_p24)
    print h_conv_p21
    print h_conv_p22
    print h_conv_p23
    print h_conv_p24
    #------------------------------
    #--------
    # concat results in one single array
    h_conv_concat = tf.concat(3,[h_conv_p21,h_conv_p22, h_conv_p23, h_conv_p24])
    print h_conv_concat
    #----------------
    #max_pooling before assing to the last layer
    h_pool_concat = avg_pool_2x2(h_conv_concat)
    #----------
    #-------------
    #last convolutional layer
    W_conv4 = weight_variable([pat_size4,pat_size4,4*Nmaps_parall2, Nmaps_c4])
    b_conv4 = bias_variable([Nmaps_c4])
    #makes convolution
    h_conv4 = tf.nn.relu(conv2d(h_pool_concat, W_conv4) + b_conv4)
    #----------------------
    #---------------------
    #fully conncected layers
    #parameters FC1
    W_fc1 = weight_variable([new_dim * new_dim * Nmaps_c4, Nmaps_fc1])
    b_fc1 = bias_variable([Nmaps_fc1])
    #calculates
    h_conv4_flat = tf.reshape(h_conv4, [-1, new_dim * new_dim * Nmaps_c4])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)
    #parameters FC2
    W_fc2 = weight_variable([Nmaps_fc1, Nmaps_fc2])
    b_fc2 = bias_variable([Nmaps_fc2])
    #calculation
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    #preparation for readout layer
    #dropout
#    keep_prob = tf.placeholder(tf.float32)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
    #softmax layer --- tells the output
    W_final = weight_variable([Nmaps_fc2, 2])
    b_final = bias_variable([2])

    y_conv=tf.nn.softmax(tf.matmul(h_fc2_drop, W_final) + b_final)

    return y_conv




