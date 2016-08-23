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
  initial = tf.truncated_normal(shape, stddev=0.5)
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

def max_pool_3x3s2(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
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

    # 7x7 convolution: pdim x pdim x 64
    W_conv1 = weight_variable([7, 7, nchannels, 64])
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # 3x3 max pool (stride 2): (pdim/2) x (pdim/2) x 64
    h_pool1 = max_pool_3x3s2(h_conv1)

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
   
    # 3x3 max pool (stride 2): (pdim/4) x (pdim/4) x 192
    h_pool2 = max_pool_3x3s2(h_norm2)

    ###
    # --- Inception 3a
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
    h_iout3a = tf.concat(3,[h_conv4a,h_conv4c,h_conv4e,h_conv4f])
    ###

    ###
    # --- Inception 3b
    # --- Branch1: 1x1 conv (128 filters)
    W_conv5a = weight_variable([1, 1, 256, 128])
    b_conv5a = bias_variable([128])
    h_conv5a = tf.nn.relu(conv2d(h_iout3a, W_conv5a) + b_conv5a)

    # --- Branch2: 1x1 conv (128 filters) + 3x3 conv (192 filters)
    W_conv5b = weight_variable([1, 1, 256, 128])
    b_conv5b = bias_variable([128])
    h_conv5b = tf.nn.relu(conv2d(h_iout3a, W_conv5b) + b_conv5b)
    W_conv5c = weight_variable([3, 3, 128, 192])
    b_conv5c = bias_variable([192])
    h_conv5c = tf.nn.relu(conv2d(h_conv5b, W_conv5c) + b_conv5c)

    # --- Branch3: 1x1 conv (32 filters) + 5x5 conv (96 filters)
    W_conv5d = weight_variable([1, 1, 256, 32])
    b_conv5d = bias_variable([32])
    h_conv5d = tf.nn.relu(conv2d(h_iout3a, W_conv5d) + b_conv5d)
    W_conv5e = weight_variable([5, 5, 32, 96])
    b_conv5e = bias_variable([96])
    h_conv5e = tf.nn.relu(conv2d(h_conv5d, W_conv5e) + b_conv5e)

    # --- Branch4: 2x2 max pool, 1x1 conv (64 filters)
    h_pool4 = max_pool_2x2s1(h_iout3a)
    W_conv5f = weight_variable([1, 1, 256, 64])
    b_conv5f = bias_variable([64])
    h_conv5f = tf.nn.relu(conv2d(h_pool4, W_conv5f) + b_conv5f)

    # --- Concatenation: output is (pdim/4) x (pdim/4) x 480
    h_iout3b = tf.concat(3,[h_conv5a,h_conv5c,h_conv5e,h_conv5f])
    ###

    # 3x3 max pool (stride 2): output is (pdim/8) x (pdim/8) x 480
    h_pool5 = max_pool_3x3s2(h_iout3b)

    ###
    # --- Inception 4a
    # --- Branch1: 1x1 conv (192 filters)
    W_conv6a = weight_variable([1, 1, 480, 192])
    b_conv6a = bias_variable([192])
    h_conv6a = tf.nn.relu(conv2d(h_pool5, W_conv6a) + b_conv6a)

    # --- Branch2: 1x1 conv (96 filters) + 3x3 conv (208 filters)
    W_conv6b = weight_variable([1, 1, 480, 96])
    b_conv6b = bias_variable([96])
    h_conv6b = tf.nn.relu(conv2d(h_pool5, W_conv6b) + b_conv6b)
    W_conv6c = weight_variable([3, 3, 96, 208])
    b_conv6c = bias_variable([208])
    h_conv6c = tf.nn.relu(conv2d(h_conv6b, W_conv6c) + b_conv6c)

    # --- Branch3: 1x1 conv (16 filters) + 5x5 conv (48 filters)
    W_conv6d = weight_variable([1, 1, 480, 16])
    b_conv6d = bias_variable([16])
    h_conv6d = tf.nn.relu(conv2d(h_pool5, W_conv6d) + b_conv6d)
    W_conv6e = weight_variable([5, 5, 16, 48])
    b_conv6e = bias_variable([48])
    h_conv6e = tf.nn.relu(conv2d(h_conv6d, W_conv6e) + b_conv6e)

    # --- Branch4: 2x2 max pool, 1x1 conv (64 filters)
    h_pool6 = max_pool_2x2s1(h_pool5)
    W_conv6f = weight_variable([1, 1, 480, 64])
    b_conv6f = bias_variable([64])
    h_conv6f = tf.nn.relu(conv2d(h_pool6, W_conv6f) + b_conv6f)

    # --- Concatenation: output is (pdim/8) x (pdim/8) x 512
    h_iout4a = tf.concat(3,[h_conv6a,h_conv6c,h_conv6e,h_conv6f])
    ###

    ###
    # --- Inception 4b
    # --- Branch1: 1x1 conv (160 filters)
    W_conv7a = weight_variable([1, 1, 512, 160])
    b_conv7a = bias_variable([160])
    h_conv7a = tf.nn.relu(conv2d(h_iout4a, W_conv7a) + b_conv7a)

    # --- Branch2: 1x1 conv (112 filters) + 3x3 conv (224 filters)
    W_conv7b = weight_variable([1, 1, 512, 112])
    b_conv7b = bias_variable([112])
    h_conv7b = tf.nn.relu(conv2d(h_iout4a, W_conv7b) + b_conv7b)
    W_conv7c = weight_variable([3, 3, 112, 224])
    b_conv7c = bias_variable([224])
    h_conv7c = tf.nn.relu(conv2d(h_conv7b, W_conv7c) + b_conv7c)

    # --- Branch3: 1x1 conv (24 filters) + 5x5 conv (64 filters)
    W_conv7d = weight_variable([1, 1, 512, 24])
    b_conv7d = bias_variable([24])
    h_conv7d = tf.nn.relu(conv2d(h_iout4a, W_conv7d) + b_conv7d)
    W_conv7e = weight_variable([5, 5, 24, 64])
    b_conv7e = bias_variable([64])
    h_conv7e = tf.nn.relu(conv2d(h_conv7d, W_conv7e) + b_conv7e)

    # --- Branch4: 2x2 max pool, 1x1 conv (64 filters)
    h_pool7 = max_pool_2x2s1(h_iout4a)
    W_conv7f = weight_variable([1, 1, 512, 64])
    b_conv7f = bias_variable([64])
    h_conv7f = tf.nn.relu(conv2d(h_pool7, W_conv7f) + b_conv7f)

    # --- Concatenation: output is (pdim/8) x (pdim/8) x 512
    h_iout4b = tf.concat(3,[h_conv7a,h_conv7c,h_conv7e,h_conv7f])
    ###

    ###
    # --- Inception 4c
    # --- Branch1: 1x1 conv (128 filters)
    W_conv8a = weight_variable([1, 1, 512, 128])
    b_conv8a = bias_variable([128])
    h_conv8a = tf.nn.relu(conv2d(h_iout4b, W_conv8a) + b_conv8a)

    # --- Branch2: 1x1 conv (128 filters) + 3x3 conv (256 filters)
    W_conv8b = weight_variable([1, 1, 512, 128])
    b_conv8b = bias_variable([128])
    h_conv8b = tf.nn.relu(conv2d(h_iout4b, W_conv8b) + b_conv8b)
    W_conv8c = weight_variable([3, 3, 128, 256])
    b_conv8c = bias_variable([256])
    h_conv8c = tf.nn.relu(conv2d(h_conv8b, W_conv8c) + b_conv8c)

    # --- Branch3: 1x1 conv (24 filters) + 5x5 conv (64 filters)
    W_conv8d = weight_variable([1, 1, 512, 24])
    b_conv8d = bias_variable([24])
    h_conv8d = tf.nn.relu(conv2d(h_iout4b, W_conv8d) + b_conv8d)
    W_conv8e = weight_variable([5, 5, 24, 64])
    b_conv8e = bias_variable([64])
    h_conv8e = tf.nn.relu(conv2d(h_conv8d, W_conv8e) + b_conv8e)

    # --- Branch4: 2x2 max pool, 1x1 conv (64 filters)
    h_pool8 = max_pool_2x2s1(h_iout4b)
    W_conv8f = weight_variable([1, 1, 512, 64])
    b_conv8f = bias_variable([64])
    h_conv8f = tf.nn.relu(conv2d(h_pool8, W_conv8f) + b_conv8f)

    # --- Concatenation: output is (pdim/8) x (pdim/8) x 512
    h_iout4c = tf.concat(3,[h_conv8a,h_conv8c,h_conv8e,h_conv8f])
    ###

    ###
    # --- Inception 4d
    # --- Branch1: 1x1 conv (112 filters)
    W_conv9a = weight_variable([1, 1, 512, 112])
    b_conv9a = bias_variable([112])
    h_conv9a = tf.nn.relu(conv2d(h_iout4c, W_conv9a) + b_conv9a)

    # --- Branch2: 1x1 conv (144 filters) + 3x3 conv (288 filters)
    W_conv9b = weight_variable([1, 1, 512, 144])
    b_conv9b = bias_variable([144])
    h_conv9b = tf.nn.relu(conv2d(h_iout4c, W_conv9b) + b_conv9b)
    W_conv9c = weight_variable([3, 3, 144, 288])
    b_conv9c = bias_variable([288])
    h_conv9c = tf.nn.relu(conv2d(h_conv9b, W_conv9c) + b_conv9c)

    # --- Branch3: 1x1 conv (32 filters) + 5x5 conv (64 filters)
    W_conv9d = weight_variable([1, 1, 512, 32])
    b_conv9d = bias_variable([32])
    h_conv9d = tf.nn.relu(conv2d(h_iout4c, W_conv9d) + b_conv9d)
    W_conv9e = weight_variable([5, 5, 32, 64])
    b_conv9e = bias_variable([64])
    h_conv9e = tf.nn.relu(conv2d(h_conv9d, W_conv9e) + b_conv9e)

    # --- Branch4: 2x2 max pool, 1x1 conv (64 filters)
    h_pool9 = max_pool_2x2s1(h_iout4c)
    W_conv9f = weight_variable([1, 1, 512, 64])
    b_conv9f = bias_variable([64])
    h_conv9f = tf.nn.relu(conv2d(h_pool9, W_conv9f) + b_conv9f)

    # --- Concatenation: output is (pdim/8) x (pdim/8) x 528
    h_iout4d = tf.concat(3,[h_conv9a,h_conv9c,h_conv9e,h_conv9f])
    ###

    ###
    # --- Inception 4e
    # --- Branch1: 1x1 conv (256 filters)
    W_conv10a = weight_variable([1, 1, 528, 256])
    b_conv10a = bias_variable([256])
    h_conv10a = tf.nn.relu(conv2d(h_iout4d, W_conv10a) + b_conv10a)

    # --- Branch2: 1x1 conv (160 filters) + 3x3 conv (320 filters)
    W_conv10b = weight_variable([1, 1, 528, 160])
    b_conv10b = bias_variable([160])
    h_conv10b = tf.nn.relu(conv2d(h_iout4d, W_conv10b) + b_conv10b)
    W_conv10c = weight_variable([3, 3, 160, 320])
    b_conv10c = bias_variable([320])
    h_conv10c = tf.nn.relu(conv2d(h_conv10b, W_conv10c) + b_conv10c)

    # --- Branch3: 1x1 conv (32 filters) + 5x5 conv (128 filters)
    W_conv10d = weight_variable([1, 1, 528, 32])
    b_conv10d = bias_variable([32])
    h_conv10d = tf.nn.relu(conv2d(h_iout4d, W_conv10d) + b_conv10d)
    W_conv10e = weight_variable([5, 5, 32, 128])
    b_conv10e = bias_variable([128])
    h_conv10e = tf.nn.relu(conv2d(h_conv10d, W_conv10e) + b_conv10e)

    # --- Branch4: 2x2 max pool, 1x1 conv (128 filters)
    h_pool10 = max_pool_2x2s1(h_iout4d)
    W_conv10f = weight_variable([1, 1, 528, 128])
    b_conv10f = bias_variable([128])
    h_conv10f = tf.nn.relu(conv2d(h_pool10, W_conv10f) + b_conv10f)

    # --- Concatenation: output is (pdim/8) x (pdim/8) x 832
    h_iout4e = tf.concat(3,[h_conv10a,h_conv10c,h_conv10e,h_conv10f])
    ###

    # 3x3 max pool (stride 2): output is (pdim/16) x (pdim/16) x 832
    h_pool11 = max_pool_3x3s2(h_iout4e)

    ###
    # --- Inception 5a
    # --- Branch1: 1x1 conv (256 filters)
    W_conv11a = weight_variable([1, 1, 832, 256])
    b_conv11a = bias_variable([256])
    h_conv11a = tf.nn.relu(conv2d(h_pool11, W_conv11a) + b_conv11a)

    # --- Branch2: 1x1 conv (160 filters) + 3x3 conv (320 filters)
    W_conv11b = weight_variable([1, 1, 832, 160])
    b_conv11b = bias_variable([160])
    h_conv11b = tf.nn.relu(conv2d(h_pool11, W_conv11b) + b_conv11b)
    W_conv11c = weight_variable([3, 3, 160, 320])
    b_conv11c = bias_variable([320])
    h_conv11c = tf.nn.relu(conv2d(h_conv11b, W_conv11c) + b_conv11c)

    # --- Branch3: 1x1 conv (32 filters) + 5x5 conv (128 filters)
    W_conv11d = weight_variable([1, 1, 832, 32])
    b_conv11d = bias_variable([32])
    h_conv11d = tf.nn.relu(conv2d(h_pool11, W_conv11d) + b_conv11d)
    W_conv11e = weight_variable([5, 5, 32, 128])
    b_conv11e = bias_variable([128])
    h_conv11e = tf.nn.relu(conv2d(h_conv11d, W_conv11e) + b_conv11e)

    # --- Branch4: 2x2 max pool, 1x1 conv (128 filters)
    h_pool12 = max_pool_2x2s1(h_pool11)
    W_conv11f = weight_variable([1, 1, 832, 128])
    b_conv11f = bias_variable([128])
    h_conv11f = tf.nn.relu(conv2d(h_pool12, W_conv11f) + b_conv11f)

    # --- Concatenation: output is (pdim/16) x (pdim/16) x 832
    h_iout5a = tf.concat(3,[h_conv11a,h_conv11c,h_conv11e,h_conv11f])
    ###

    ###
    # --- Inception 5b
    # --- Branch1: 1x1 conv (384 filters)
    W_conv12a = weight_variable([1, 1, 832, 384])
    b_conv12a = bias_variable([384])
    h_conv12a = tf.nn.relu(conv2d(h_iout5a, W_conv12a) + b_conv12a)

    # --- Branch2: 1x1 conv (192 filters) + 3x3 conv (384 filters)
    W_conv12b = weight_variable([1, 1, 832, 192])
    b_conv12b = bias_variable([192])
    h_conv12b = tf.nn.relu(conv2d(h_iout5a, W_conv12b) + b_conv12b)
    W_conv12c = weight_variable([3, 3, 192, 384])
    b_conv12c = bias_variable([384])
    h_conv12c = tf.nn.relu(conv2d(h_conv12b, W_conv12c) + b_conv12c)

    # --- Branch3: 1x1 conv (48 filters) + 5x5 conv (128 filters)
    W_conv12d = weight_variable([1, 1, 832, 48])
    b_conv12d = bias_variable([48])
    h_conv12d = tf.nn.relu(conv2d(h_iout5a, W_conv12d) + b_conv12d)
    W_conv12e = weight_variable([5, 5, 48, 128])
    b_conv12e = bias_variable([128])
    h_conv12e = tf.nn.relu(conv2d(h_conv12d, W_conv12e) + b_conv12e)

    # --- Branch4: 2x2 max pool, 1x1 conv (128 filters)
    h_pool12 = max_pool_2x2s1(h_iout5a)
    W_conv12f = weight_variable([1, 1, 832, 128])
    b_conv12f = bias_variable([128])
    h_conv12f = tf.nn.relu(conv2d(h_iout5a, W_conv12f) + b_conv12f)

    # --- Concatenation: output is (pdim/16) x (pdim/16) x 1024
    h_iout5b = tf.concat(3,[h_conv12a,h_conv12c,h_conv12e,h_conv12f])
    ###

    ###
    # -- Mid-net readout layer 1 (input is the concatenated output of inception layer 4a: (pdim/8) x (pdim/8) x 512)

    # 5x5 avg. pool (stride 3); output is int(pdim/24) x int(pdim/24) x 512
    h_avg1_m1 = tf.nn.avg_pool(h_iout4a, ksize=[1, 5, 5, 1], strides=[1, 3, 3, 1], padding='VALID')
    
    # --- 1x1 conv (128 filters): output is int(pdim/24) x int(pdim/24) x 128
    W_conv1_m1 = weight_variable([1, 1, 512, 128])
    b_conv1_m1 = bias_variable([128])
    h_conv1_m1 = tf.nn.relu(conv2d(h_avg1_m1, W_conv1_m1) + b_conv1_m1)

    # Densely connected layer (1024 neurons)
    new_dim = int(pdim / 24)
    W_fc1_m1 = weight_variable([new_dim * new_dim * 128, 1024])
    b_fc1_m1 = bias_variable([1024])
    h_convm1_flat = tf.reshape(h_conv1_m1, [-1, new_dim*new_dim*128])
    h_fc1_m1 = tf.nn.relu(tf.matmul(h_convm1_flat, W_fc1_m1) + b_fc1_m1)

    # Dropout
    keep_prob = 0.7 #tf.placeholder("float")
    h_fcm1_drop = tf.nn.dropout(h_fc1_m1, keep_prob)

    # Softmax readout
    W_fc2_m1 = weight_variable([1024, 2])
    b_fc2_m1 = bias_variable([2])
    y_m1 = tf.nn.softmax(tf.matmul(h_fcm1_drop, W_fc2_m1) + b_fc2_m1)
    ###

    ###
    # -- Mid-net readout layer 2 (input is the concatenated output of inception layer 4d: (pdim/8) x (pdim/8) x 528)

    # 5x5 avg. pool (stride 3); output is (pdim/24) x (pdim/24) x 528
    h_avg1_m2 = tf.nn.avg_pool(h_iout4d, ksize=[1, 5, 5, 1], strides=[1, 3, 3, 1], padding='VALID')

    # --- 1x1 conv (128 filters): output is (pdim/124 x (pdim/24) x 128
    W_conv1_m2 = weight_variable([1, 1, 528, 128])
    b_conv1_m2 = bias_variable([128])
    h_conv1_m2 = tf.nn.relu(conv2d(h_avg1_m2, W_conv1_m2) + b_conv1_m2)

    # Densely connected layer (1024 neurons)
    new_dim = int(pdim / 24)
    W_fc1_m2 = weight_variable([new_dim * new_dim * 128, 1024])
    b_fc1_m2 = bias_variable([1024])
    h_convm2_flat = tf.reshape(h_conv1_m2, [-1, new_dim*new_dim*128])
    h_fc1_m2 = tf.nn.relu(tf.matmul(h_convm2_flat, W_fc1_m2) + b_fc1_m2)

    # Dropout
    keep_prob = 0.7 #tf.placeholder("float")
    h_fcm2_drop = tf.nn.dropout(h_fc1_m2, keep_prob)

    # Softmax readout
    W_fc2_m2 = weight_variable([1024, 2])
    b_fc2_m2 = bias_variable([2])
    y_m2 = tf.nn.softmax(tf.matmul(h_fcm2_drop, W_fc2_m2) + b_fc2_m2)

    ###
    
    ###
    # -- Final readout layer

    # 7x7 average pool
    h_avg3 = tf.nn.avg_pool(h_iout5b, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='SAME')

    # Fully-connected layer
    new_dim = int(pdim / 16)
    W_fc1 = weight_variable([new_dim * new_dim * 1024, 1024])
    b_fc1 = bias_variable([1024])
    h_avg1_flat = tf.reshape(h_avg3, [-1, new_dim*new_dim*1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_avg1_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = 0.4
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Softmax readout
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    ###

    # Create the final output layer by concatenating the 3 readout layers.
    yout = tf.concat(1,[y,y_m1,y_m2])

    return yout

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




