from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def conv2d(inputs, num_outputs, kernel_shape, strides=[1, 1], mask_type=None, scope="conv2d"):
  """
  Args:
    inputs: nhwc
    kernel_shape: [height, width]
    mask_type: None or 'A' or 'B' or 'C'
  Returns:
    outputs: nhwc
  """
  with tf.variable_scope(scope) as scope:
    kernel_h, kernel_w = kernel_shape
    stride_h, stride_w = strides
    batch_size, height, width, in_channel = inputs.get_shape().as_list()

    center_h = kernel_h // 2
    center_w = kernel_w // 2

    assert kernel_h % 2 == 1 and kernel_w % 2 == 1, "kernel height and width must be odd number"
    mask = np.zeros((kernel_h, kernel_w, in_channel, num_outputs), dtype=np.float32)
    if mask_type is not None:
      #C
      mask[:center_h, :, :, :] = 1
      if mask_type == 'A':
        mask[center_h, :center_w, :, :] = 1
        """
        mask[center_h, :center_w, :, :] = 1
        #G channel
        mask[center_h, center_w, 0:in_channel:3, 1:num_outputs:3] = 1
        #B Channel
        mask[center_h, center_w, 0:in_channel:3, 2:num_outputs:3] = 1
        mask[center_h, center_w, 1:in_channel:3, 2:num_outputs:3] = 1
        """
      if mask_type == 'B':
        mask[center_h, :center_w+1, :, :] = 1
        """
        mask[center_h, :center_w, :, :] = 1
        #R Channel
        mask[center_h, center_w, 0:in_channel:3, 0:num_outputs:3] = 1
        #G channel
        mask[center_h, center_w, 0:in_channel:3, 1:num_outputs:3] = 1
        mask[center_h, center_w, 1:in_channel:3, 1:num_outputs:3] = 1
        #B Channel
        mask[center_h, center_w, :, 2:num_outputs:3] = 1
        """
    else:
      mask[:, :, :, :] = 1

    weights_shape = [kernel_h, kernel_w, in_channel, num_outputs]
    weights = tf.get_variable("weights", weights_shape,
      tf.float32, tf.truncated_normal_initializer(stddev=0.1))
    weights = weights * mask
    biases = tf.get_variable("biases", [num_outputs],
          tf.float32, tf.constant_initializer(0.0))

    outputs = tf.nn.conv2d(inputs, weights, [1, stride_h, stride_w, 1], padding="SAME")
    outputs = tf.nn.bias_add(outputs, biases)

    return outputs

def gated_conv2d(inputs, state, kernel_shape, scope):
  """
  Args:
    inputs: nhwc
    state:  nhwc
    kernel_shape: [height, width]
  Returns:
    outputs: nhwc
    new_state: nhwc
  """
  with tf.variable_scope(scope) as scope:
    batch_size, height, width, in_channel = inputs.get_shape().as_list()
    kernel_h, kernel_w = kernel_shape
    #state route
    left = conv2d(state, 2 * in_channel, kernel_shape, strides=[1, 1], mask_type='C', scope="conv_s1")
    left1 = left[:, :, :, 0:in_channel]
    left2 = left[:, :, :, in_channel:]
    left1 = tf.nn.tanh(left1)
    left2 = tf.nn.sigmoid(left2)
    new_state = left1 * left2
    left2right = conv2d(left, 2 * in_channel, [1, 1], strides=[1, 1], scope="conv_s2")
    #input route
    right = conv2d(inputs, 2 * in_channel, [1, kernel_w], strides=[1, 1], mask_type='B', scope="conv_r1")
    right = right + left2right
    right1 = right[:, :, :, 0:in_channel]
    right2 = right[:, :, :, in_channel:]
    right1 = tf.nn.tanh(right1)
    right2 = tf.nn.sigmoid(right2)
    up_right = right1 * right2
    up_right = conv2d(up_right, in_channel, [1, 1], strides=[1, 1], mask_type='B', scope="conv_r2")
    outputs = inputs + up_right

    return outputs, new_state

def batch_norm(x, train=True, scope=None):
  return tf.contrib.layers.batch_norm(x, center=True, scale=True, updates_collections=None, is_training=train, trainable=True, scope=scope)

def resnet_block(inputs, num_outputs, kernel_shape, strides=[1, 1], scope=None, train=True):
  """
  Args:
    inputs: nhwc
    num_outputs: int
    kernel_shape: [kernel_h, kernel_w]
  Returns:
    outputs: nhw(num_outputs)
  """
  with tf.variable_scope(scope) as scope:
    conv1 = conv2d(inputs, num_outputs, kernel_shape, strides=[1, 1], mask_type=None, scope="conv1")
    bn1 = batch_norm(conv1, train=train, scope='bn1')
    relu1 = tf.nn.relu(bn1)
    conv2 = conv2d(relu1, num_outputs, kernel_shape, strides=[1, 1], mask_type=None, scope="conv2")
    bn2 = batch_norm(conv2, train=train, scope='bn2')
    output = inputs + bn2

    return output 

def deconv2d(inputs, num_outputs, kernel_shape, strides=[1, 1], scope="deconv2d"):
  """
  Args:
    inputs: nhwc
    num_outputs: int
    kernel_shape: [kernel_h, kernel_w]
    strides: [stride_h, stride_w]
  Returns:
    outputs: nhwc
  """
  with tf.variable_scope(scope) as scope:
    return tf.contrib.layers.convolution2d_transpose(inputs, num_outputs, kernel_shape, strides, \
          padding='SAME', weights_initializer=tf.truncated_normal_initializer(stddev=0.1), \
          biases_initializer=tf.constant_initializer(0.0))