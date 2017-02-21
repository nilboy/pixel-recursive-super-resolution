from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from ops import *

class Net(object):
  def __init__(self, hr_images, lr_images, scope):
    """
    Args:[0, 255]
      hr_images: [batch_size, hr_height, hr_width, in_channels] float32
      lr_images: [batch_size, lr_height, lr_width, in_channels] float32
    """
    with tf.variable_scope(scope) as scope:
      self.train = tf.placeholder(tf.bool)
      self.construct_net(hr_images, lr_images)
  def prior_network(self, hr_images):
    """
    Args:[-0.5, 0.5]
      hr_images: [batch_size, hr_height, hr_width, in_channels]
    Returns:
      prior_logits: [batch_size, hr_height, hr_width, 3*256]
    """
    with tf.variable_scope('prior') as scope:
      conv1 = conv2d(hr_images, 64, [7, 7], strides=[1, 1], mask_type='A', scope="conv1")
      inputs = conv1
      state = conv1
      for i in range(20):
        inputs, state = gated_conv2d(inputs, state, [5, 5], scope='gated' + str(i))
      conv2 = conv2d(inputs, 1024, [1, 1], strides=[1, 1], mask_type='B', scope="conv2")
      conv2 = tf.nn.relu(conv2)
      prior_logits = conv2d(conv2, 3 * 256, [1, 1], strides=[1, 1], mask_type='B', scope="conv3")

      prior_logits = tf.concat([prior_logits[:, :, :, 0::3], prior_logits[:, :, :, 1::3], prior_logits[:, :, :, 2::3]], 3)

      return prior_logits


  def conditioning_network(self, lr_images):
    """
    Args:[-0.5, 0.5]
      lr_images: [batch_size, lr_height, lr_width, in_channels]
    Returns:
      conditioning_logits: [batch_size, hr_height, hr_width, 3*256]
    """
    res_num = 6
    with tf.variable_scope('conditioning') as scope:
      inputs = lr_images
      inputs = conv2d(inputs, 32, [1, 1], strides=[1, 1], mask_type=None, scope="conv_init")
      for i in range(2):
        for j in range(res_num):
          inputs = resnet_block(inputs, 32, [3, 3], strides=[1, 1], scope='res' + str(i) + str(j), train=self.train)
        inputs = deconv2d(inputs, 32, [3, 3], strides=[2, 2], scope="deconv" + str(i))
        inputs = tf.nn.relu(inputs)
      for i in range(res_num):
        inputs = resnet_block(inputs, 32, [3, 3], strides=[1, 1], scope='res3' + str(i), train=self.train)
      conditioning_logits = conv2d(inputs, 3*256, [1, 1], strides=[1, 1], mask_type=None, scope="conv")

      return conditioning_logits

  def softmax_loss(self, logits, labels):
    logits = tf.reshape(logits, [-1, 256])
    labels = tf.cast(labels, tf.int32)
    labels = tf.reshape(labels, [-1])
    return tf.losses.sparse_softmax_cross_entropy(
           labels, logits)
  def construct_net(self, hr_images, lr_images):
    """
    Args: [0, 255]
    """
    #labels
    labels = hr_images
    #normalization images [-0.5, 0.5]
    hr_images = hr_images / 255.0 - 0.5
    lr_images = lr_images / 255.0 - 0.5
    self.prior_logits = self.prior_network(hr_images)
    self.conditioning_logits = self.conditioning_network(lr_images)

    loss1 = self.softmax_loss(self.prior_logits + self.conditioning_logits, labels)
    loss2 = self.softmax_loss(self.conditioning_logits, labels)
    loss3 = self.softmax_loss(self.prior_logits, labels)

    self.loss = loss1 + loss2
    tf.summary.scalar('loss', self.loss)
    tf.summary.scalar('loss_prior', loss3)