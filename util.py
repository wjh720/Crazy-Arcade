import numpy as np
import tensorflow as tf  # pylint: ignore-module
import functools
import copy
import os
import collections

# ================================================================
# Make consistent with numpy
# ================================================================

def intprod(x):
    return int(np.prod(x))

def flattenallbut0(x):
    return tf.reshape(x, [-1, intprod(x.get_shape().as_list()[1:])])

def ch(x):
    if(x=='l'):return 0
    if(x=='r'):return 1
    if(x=='u'):return 2
    if(x=='d'):return 3
    if(x=='b'):return 4
    if(x=='s'):return 5

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None,
           summary_tag=None, trainable = True):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = intprod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = intprod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections, trainable=trainable)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.zeros_initializer(),
                            collections=collections, trainable=trainable)

        if summary_tag is not None:
            tf.summary.image(summary_tag,
                             tf.transpose(tf.reshape(w, [filter_size[0], filter_size[1], -1, 1]),
                                          [2, 0, 1, 3]),
                             max_images=10)

        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def dense(x, size, name, weight_init=None, bias=True, trainable = True):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init, trainable=trainable)
    ret = tf.matmul(x, w)
    if bias:
        b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer(), trainable=trainable)
        return ret + b
    else:
        return ret