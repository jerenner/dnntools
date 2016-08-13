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

def max_pool_2x2s1(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 1, 1, 1], padding='SAME')

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
    if(name == "MNISTnew"):
        return MNISTnewLS_net(x_input)
    if(name == "NEXTGoogLe"):
        return NEXTGoogLe_net(x_input)

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

    # Resize the array to pdim x pdim x 10
    x_image = tf.reshape(x_input, [-1,pdim,pdim,nchannels])

    # First convolutional layer
    W_conv1 = weight_variable([3, 3, nchannels, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_norm1 = tf.nn.l2_normalize(h_pool1, 3)

    # Second convolutional layer
    W_conv2 = weight_variable([2, 2, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_norm1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_norm2 = tf.nn.l2_normalize(h_pool2, 3)

    # Densely connected layer
    new_dim = int(pdim / 4)
    W_fc1 = weight_variable([new_dim * new_dim * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_norm2_flat = tf.reshape(h_norm2, [-1, new_dim*new_dim*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_norm2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = 0.7 #tf.placeholder("float")
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
    W_conv1 = weight_variable([3, 3, 3, nchannels, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2x2(h_conv1)
    print "Created first convolutional layer..."

    # Second convolutional layer
    W_conv2 = weight_variable([2, 2, 2, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2x2(h_conv2)
    print "Created second convolutional layer..."

    # Densely connected layer
    new_dim = int(pdim / 4)
    W_fc1 = weight_variable([new_dim * new_dim * new_dim * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, new_dim*new_dim*new_dim*64])
    #h_pool1_flat = tf.reshape(h_pool1, [-1, new_dim*new_dim*new_dim*32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    print "Created densely connected layer..."

    # Dropout
    keep_prob = 0.7 #tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Softmax readout
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y

def NEXTGoogLe_net(x_input):

    # Resize the array to pdim x pdim x nchannels
    x_image = tf.reshape(x_input, [-1,pdim,pdim,nchannels])

    # 5x5 convolution: pdim x pdim x 64
    W_conv1 = weight_variable([5, 5, nchannels, 64])
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # 2x2 max pool: (pdim/2) x (pdim/2) x 64
    h_pool1 = max_pool_2x2(h_conv1)

    # local response normalization
    h_norm1 = tf.nn.l2_normalize(h_pool1, 3)

    # 1x1 convolution: (pdim/2) x (pdim/2) x 64
    W_conv2 = weight_variable([1, 1, 64, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_norm1, W_conv2) + b_conv2)

    # 3x3 convolution: (pdim/2) x (pdim/2) x 192
    W_conv3 = weight_variable([3, 3, 64, 192])
    b_conv3 = bias_variable([192])
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
    
    # local response normalization
    h_norm2 = tf.nn.l2_normalize(h_conv3, 3)
   
    # 2x2 max pool: (pdim/4) x (pdim/4) x 192
    h_pool2 = max_pool_2x2(h_norm2)

    ###
    # --- Inception a
    # --- Branch1: 1x1 conv (64 filters)
    W_conv4a = weight_variable([1, 1, 192, 64])
    b_conv4a = bias_variable([64])
    h_conv4a = tf.nn.relu(conv2d(h_pool2, W_conv4a) + b_conv4a)

    # --- Branch2: 1x1 conv (96 filters) + 3x3 conv (128 filters)
    W_conv4b = weight_variable([1, 1, 192, 96])
    b_conv4b = bias_variable([96])
    h_conv4b = tf.nn.relu(conv2d(h_pool2, W_conv4b) + b_conv4b)
    W_conv4c = weight_variable([3, 3, 96, 128])
    b_conv4c = bias_variable([128])
    h_conv4c = tf.nn.relu(conv2d(h_conv4b, W_conv4c) + b_conv4c)

    # --- Branch3: 1x1 conv (16 filters) + 5x5 conv (32 filters)
    W_conv4d = weight_variable([1, 1, 192, 16])
    b_conv4d = bias_variable([16])
    h_conv4d = tf.nn.relu(conv2d(h_pool2, W_conv4d) + b_conv4d)
    W_conv4e = weight_variable([5, 5, 16, 32])
    b_conv4e = bias_variable([32])
    h_conv4e = tf.nn.relu(conv2d(h_conv4d, W_conv4e) + b_conv4e)

    # --- Branch4: 2x2 max pool, 1x1 conv (32 filters)
    h_pool3 = max_pool_2x2s1(h_pool2)
    W_conv4f = weight_variable([1, 1, 192, 32])
    b_conv4f = bias_variable([32])
    h_conv4f = tf.nn.relu(conv2d(h_pool3, W_conv4f) + b_conv4f)

    # --- Concatenation: output is (pdim/4) x (pdim/4) x 256
    h_iout1 = tf.concat(3,[h_conv4a,h_conv4c,h_conv4e,h_conv4f])
    ###

    # 5x5 max pool: output is (pdim/20) x (pdim/20) x 256
    h_pool4 = tf.nn.max_pool(h_iout1, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding='SAME')
   
    ###
    # --- Inception b
    # --- Branch1: 1x1 conv (128 filters)
    W_conv5a = weight_variable([1, 1, 256, 128])
    b_conv5a = bias_variable([128])
    h_conv5a = tf.nn.relu(conv2d(h_pool4, W_conv5a) + b_conv5a)

    # --- Branch2: 1x1 conv (128 filters) + 3x3 conv (192 filters)
    W_conv5b = weight_variable([1, 1, 256, 128])
    b_conv5b = bias_variable([128])
    h_conv5b = tf.nn.relu(conv2d(h_pool4, W_conv5b) + b_conv5b)
    W_conv5c = weight_variable([3, 3, 128, 192])
    b_conv5c = bias_variable([192])
    h_conv5c = tf.nn.relu(conv2d(h_conv5b, W_conv5c) + b_conv5c)

    # --- Branch3: 1x1 conv (32 filters) + 5x5 conv (96 filters)
    W_conv5d = weight_variable([1, 1, 256, 32])
    b_conv5d = bias_variable([32])
    h_conv5d = tf.nn.relu(conv2d(h_pool4, W_conv5d) + b_conv5d)
    W_conv5e = weight_variable([5, 5, 32, 96])
    b_conv5e = bias_variable([96])
    h_conv5e = tf.nn.relu(conv2d(h_conv5d, W_conv5e) + b_conv5e)

    # --- Branch4: 2x2 max pool, 1x1 conv (64 filters)
    h_pool5 = max_pool_2x2s1(h_pool4)
    W_conv5f = weight_variable([1, 1, 256, 64])
    b_conv5f = bias_variable([64])
    h_conv5f = tf.nn.relu(conv2d(h_pool5, W_conv5f) + b_conv5f)

    # --- Concatenation: output is (pdim/20) x (pdim/20) x 480
    h_iout2 = tf.concat(3,[h_conv5a,h_conv5c,h_conv5e,h_conv5f])
    ###

    # 5x5 average pool
    h_avg1 = tf.nn.avg_pool(h_iout2, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding='SAME')

    # Fully-connected layer
    new_dim = int(pdim / 100)
    W_fc1 = weight_variable([new_dim * new_dim * 480, 1024])
    b_fc1 = bias_variable([1024])
    h_avg1_flat = tf.reshape(h_avg1, [-1, new_dim*new_dim*480])
    h_fc1 = tf.nn.relu(tf.matmul(h_avg1_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = 0.4
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




