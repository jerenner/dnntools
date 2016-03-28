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

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# -----------------------------------------------------------------------------
# Set up the neural network.
# -----------------------------------------------------------------------------
def getNet(name,x_input):

    if(name == "MNISTbasic"):
        return MNISTbasic_net(x_input)
    if(name == "MNISTadv"):
        return MNISTadv_net(x_input)

# -----------------------------------------------------------------------------
# DNN definitions
# -----------------------------------------------------------------------------
def MNISTbasic_net(x_input):

    # Softmax readout: 2 classifications
    W = weight_variable([3*npix, 2])
    b = bias_variable([2])
    y = tf.nn.softmax(tf.matmul(x_input, W) + b)

    return y

