from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools

import tensorflow as tf
from tensorflow.contrib.model_pruning.python import pruning

"""
implement mobilenet v1 with weights pruning. 
MobileNet v1 reference:
weights pruning reference:https://arxiv.org/pdf/1710.01878.pdf
"""

"""
MOBILENETV1_STRUCTURE = [
    Conv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=2, depth=1024),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024),
    AvgPool(pool=[7,7], stride=1),
    FC(1024*3755),
    Softmax
]
"""

def _variable_with_weight_decay(shape, initializier, wd=None, name="weights"):

    """
    Variable create helper, create a varable of shape with initializer and l2 loss weight decay
    :param shape: shape
    :param initilizier: normal or Xavier normal 
    :param wd: l2 weight decay
    :return: var
    """
    var = tf.get_variable(name=name, shape=shape, initializer=initializier)
    collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd is None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="weight_l2_loss")
        tf.add_to_collection(collection_name, weight_decay)
    return var


def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def max_pool_2d(x, size=(2, 2), stride=(2, 2), name='pooling'):
    """
    Max pooling 2D Wrapper
    :param x: (tf.tensor) The input to the layer (N,H,W,C).
    :param size: (tuple) This specifies the size of the filter as well as the stride.
    :param stride: (tuple) specifies the stride of pooling.
    :param name: (string) Scope name.
    :return: The output is the same input but halfed in both width and height (N,H/2,W/2,C).
    """
    size_x, size_y = size
    stride_x, stride_y = stride
    return tf.nn.max_pool(x, ksize=[1, size_x, size_y, 1], strides=[1, stride_x, stride_y, 1], padding='VALID',
                          name=name)


def avg_pool_2d(x, size=(7, 7), stride=(1, 1), name='avg_pooling'):
    """
        Average pooling 2D Wrapper
        :param x: (tf.tensor) The input to the layer (N,H,W,C).
        :param size: (tuple) This specifies the size of the filter as well as the stride.
        :param name: (string) Scope name.
        :return: The output is the same input but halfed in both width and height (N,H/2,W/2,C).
    """
    size_x, size_y = size
    stride_x, stride_y = stride
    return tf.nn.avg_pool(x, ksize=[1, size_x, size_y, 1], strides=[1, stride_x, stride_y, 1], padding='VALID',
                          name=name)

def dropout(inputs, dropout_keep_prob, is_training, name="dropout"):
    """Dropout special layer"""

    def drop_out(conv_out):
        if is_training:
            return tf.nn.dropout(conv_out, dropout_keep_prob)
        else:
            return tf.nn.dropout(conv_out, 1.0)

    if dropout_keep_prob != -1:
        output = drop_out(inputs)
    else:
        output = inputs
    return output

def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.
  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out


def _conv2d(inputs, weights=None, num_filters=16,
            kernel_size=(3,3), strides=(1, 1), padding='SAME',
            initializer=tf.contrib.layers.xavier_initializer(),
            l2_strenth=0.0, bias=0.0, name="conv2d", is_pruning=False):

    with tf.variable_scope(name):
        stride = [1, strides[0], strides[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], inputs.shape[-1], num_filters]

        #with tf.name_scope(name):
        if weights==None:
            weights = _variable_with_weight_decay(kernel_shape, initializier=initializer, wd=l2_strenth, name="weights")
        if isinstance(bias, float):
            bias = tf.get_variable("bias",[num_filters], initializer=tf.constant_initializer(bias))

        _variable_summaries(weights)
        _variable_summaries(bias)
        if is_pruning == True:
            weights = pruning.apply_mask(weights)
        conv2d_out = tf.nn.conv2d(inputs, weights, strides=stride, padding=padding)
        conv2d_out = tf.nn.bias_add(conv2d_out, bias)

        _variable_summaries(conv2d_out)
    return conv2d_out


def conv2d(inputs, weights=None, num_filters=16,
            kernel_size=(3,3), strides=(1, 1), padding='SAME',
            initializer=tf.contrib.layers.xavier_initializer(),
            l2_strenth=0.0, bias=0.0, name="conv2d", is_pruning=False,
           activation=None, batch_normlization=False, max_pool=False,
           keep_prob=-1, is_training=True):

    out_conv = _conv2d(inputs, weights=weights, num_filters=num_filters,
                       kernel_size=kernel_size, strides=strides,
                       padding=padding, initializer=initializer,
                       l2_strenth=l2_strenth, bias=bias, name=name, is_pruning=is_pruning)
    if batch_normlization:
        #out_conv = tf.layers.batch_normalization(out_conv, training=is_training)
        out_conv = tf.contrib.layers.batch_norm(out_conv, decay=0.9997, epsilon=0.001, center=True, scale=True, is_training=is_training)

    if activation:
        out_conv = activation(out_conv)

    def drop_out(conv_out):
        if is_training:
            return tf.nn.dropout(conv_out, keep_prob)
        else:
            return tf.nn.dropout(conv_out, 1.0)

    if keep_prob != -1:
        out_conv = drop_out(out_conv)

    if max_pool:
        out_conv = max_pool_2d(out_conv)

    return out_conv


def _depthwise_conv2d(inputs, weights=None,
            kernel_size=(3,3), strides=(1, 1),
            padding='SAME',
            initializer=tf.contrib.layers.xavier_initializer(),
            l2_strenth=0.0, bias=0.0, name="depthwise_conv2d", is_pruning=False):

    with tf.variable_scope(name):
        stride = [1, strides[0], strides[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], inputs.shape[-1], 1]

        with tf.name_scope(name):
            if weights==None:
                weights = _variable_with_weight_decay(kernel_shape, initializier=initializer, wd=l2_strenth, name="weights")
            if isinstance(bias, float):
                bias = tf.get_variable("bias",[inputs.shape[-1]], initializer=tf.constant_initializer(bias))

            _variable_summaries(weights)
            _variable_summaries(bias)
            if is_pruning == True:
                weights = pruning.apply_mask(weights)
            conv2d_out = tf.nn.depthwise_conv2d(inputs, weights, strides=stride, padding=padding)
            conv2d_out = tf.nn.bias_add(conv2d_out, bias)

            _variable_summaries(conv2d_out)
    return conv2d_out


def depthwise_conv2d(inputs, weights=None,
            kernel_size=(3,3), strides=(1, 1), padding='SAME',
            initializer=tf.contrib.layers.xavier_initializer(),
            l2_strenth=0.0, bias=0.0, name="depthwise_conv2d",
            is_pruning=False,
            activation=None,
            batch_normlization=False,
            is_training=True):

    out_conv = _depthwise_conv2d(inputs, weights=weights,
                       kernel_size=kernel_size, strides=strides,
                       padding=padding, initializer=initializer,
                       l2_strenth=l2_strenth, bias=bias, name=name, is_pruning=is_pruning)
    if batch_normlization:
        #out_conv = tf.layers.batch_normalization(out_conv, training=is_training)
        out_conv = tf.contrib.layers.batch_norm(out_conv, decay=0.9997, epsilon=0.001, center=True, scale=True, is_training=is_training)


    if activation:
        out_conv = activation(out_conv)
    return out_conv

def depthwise_separable_conv2d(inputs, weight_depthwise=None, weight_pointwise=None,
                               depth_multiplier=1.0,
                               num_filters=16,
                               kernel_size=(3,3),
                               strides=(1,1),
                               padding='SAME',
                               initializer=tf.contrib.layers.xavier_initializer(),
                               l2_strenth=0.0, bias=(0.0,0.0), name="depthwise_separable_conv2d",
                               depthwise_pruning=False,
                               pointwise_pruning=False,
                               activation=None,
                               batch_normlization=False,
                               is_training=True
                               ):

    total_num = int(num_filters*depth_multiplier)
    with tf.variable_scope(name):
        depthwise_out = depthwise_conv2d(inputs=inputs,weights=weight_depthwise,
                                         kernel_size=kernel_size,
                                         strides=strides,
                                         padding=padding,
                                         initializer=initializer,
                                         l2_strenth=l2_strenth,
                                         bias=bias[0],
                                         is_pruning=depthwise_pruning,
                                         activation=activation,
                                         batch_normlization=batch_normlization,
                                         is_training=is_training)

        pointwise_out = conv2d(depthwise_out, weight_pointwise,
                               num_filters=total_num,
                               kernel_size=(1,1),
                               bias=bias[1],
                               initializer=initializer,
                               l2_strenth=l2_strenth,
                               is_pruning=pointwise_pruning,
                               is_training=is_training,
                               batch_normlization=batch_normlization,
                               activation=activation,
                               name="pointwise_conv2d")

        return depthwise_out, pointwise_out


def _dense(inputs, weights=None, num_classes=1000, bias=0.0,
           l2_strength=0.0,
           initializer=tf.contrib.layers.xavier_initializer(),
           is_pruning=False,
           name="fully_connect"):

    last_shape = inputs.get_shape()[-1].value
    with tf.variable_scope(name):
        if weights == None:
            weights = _variable_with_weight_decay([last_shape, num_classes], initializer, wd=l2_strength)
        if isinstance(bias, float):
            bias = tf.get_variable("bias", [num_classes], dtype=tf.float32, initializer=tf.constant_initializer(bias))
        if is_pruning:
            weights = pruning.apply_mask(weights)
        _variable_summaries(weights)
        _variable_summaries(bias)

        out = tf.nn.bias_add(tf.matmul(inputs, weights), bias)

    return out

def dense(inputs, weights=None, num_classes=1000, bias=0.0,
          l2_strength=0.0,
          initializer=tf.contrib.layers.xavier_initializer(),
          is_pruning=False,
          is_training=True,
          batch_normalizition=False,
          actviation=None,
          keep_prob=-1,
          name="fully_connect"):

        out = _dense(inputs, weights=weights,num_classes=num_classes,bias=bias,
                     l2_strength=l2_strength,
                     initializer=initializer,
                     is_pruning=is_pruning,
                     name=name)
        if batch_normalizition:
            out = tf.layers.batch_normalization(out, training=is_training)
        if actviation:
            out = actviation(out)

        def drop_out(conv_out):
            if is_training:
                return tf.nn.dropout(conv_out, keep_prob)
            else:
                return tf.nn.dropout(conv_out, 1.0)

        if keep_prob != -1:
            out = drop_out(out)
        return out

def mobilenet_v1(inputs,
                 num_classes=3755,
                 dropout_keep_prob=0.999,
                 is_training=True,
                 min_depth=8,
                 depth_multiplier=1.0,
                 prediction_fn=tf.contrib.layers.softmax,
                 spatial_squeeze=True,
                 scope='MobilenetV1',
                 is_pruning=True):

    endpoints = {}
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                         len(input_shape))
    with tf.variable_scope(name_or_scope=scope):
        conv2d_0 = conv2d(inputs, num_filters=depth(32), kernel_size=(3,3), strides=(2,2), padding='SAME',
                          activation=tf.nn.relu6, batch_normlization=True)
        conv2d_1_depthwise, conv2d_1_pointwise = depthwise_separable_conv2d(conv2d_0, None, None,
                                                                            depth_multiplier=depth_multiplier,
                                                                            num_filters=64,
                                                                            kernel_size=(3,3),
                                                                            strides=(1,1),
                                                                            padding='SAME',
                                                                            batch_normlization=True,
                                                                            activation=tf.nn.relu6,
                                                                            pointwise_pruning=is_pruning,
                                                                            is_training=is_training,
                                                                            name="depthwise_separable_conv2d_1"
                                                                            )
        conv2d_2_depthwise, conv2d_2_pointwise = depthwise_separable_conv2d(conv2d_1_pointwise, None, None,
                                                                            depth_multiplier=depth_multiplier,
                                                                            num_filters=128,
                                                                            kernel_size=(3,3),
                                                                            strides=(2,2),
                                                                            padding='SAME',
                                                                            batch_normlization=True,
                                                                            activation=tf.nn.relu6,
                                                                            pointwise_pruning=is_pruning,
                                                                            is_training=is_training,
                                                                            name="depthwise_separable_conv2d_2"
                                                                            )
        conv2d_3_depthwise, conv2d_3_pointwise = depthwise_separable_conv2d(conv2d_2_pointwise, None, None,
                                                                            depth_multiplier=depth_multiplier,
                                                                            num_filters=128,
                                                                            kernel_size=(3,3),
                                                                            strides=(1,1),
                                                                            padding='SAME',
                                                                            batch_normlization=True,
                                                                            activation=tf.nn.relu6,
                                                                            pointwise_pruning=is_pruning,
                                                                            is_training=is_training,
                                                                            name="depthwise_separable_conv2d_3"
                                                                            )
        conv2d_4_depthwise, conv2d_4_pointwise = depthwise_separable_conv2d(conv2d_3_pointwise, None, None,
                                                                            depth_multiplier=depth_multiplier,
                                                                            num_filters=256,
                                                                            kernel_size=(3,3),
                                                                            strides=(2,2),
                                                                            padding='SAME',
                                                                            batch_normlization=True,
                                                                            activation=tf.nn.relu6,
                                                                            pointwise_pruning=is_pruning,
                                                                            is_training=is_training,
                                                                            name="depthwise_separable_conv2d_4"
                                                                            )

        conv2d_5_depthwise, conv2d_5_pointwise = depthwise_separable_conv2d(conv2d_4_pointwise, None, None,
                                                                            depth_multiplier=depth_multiplier,
                                                                            num_filters=256,
                                                                            kernel_size=(3,3),
                                                                            strides=(1,1),
                                                                            padding='SAME',
                                                                            batch_normlization=True,
                                                                            activation=tf.nn.relu6,
                                                                            pointwise_pruning=is_pruning,
                                                                            is_training=is_training,
                                                                            name="depthwise_separable_conv2d_5"
                                                                            )
        conv2d_6_depthwise, conv2d_6_pointwise = depthwise_separable_conv2d(conv2d_5_pointwise, None, None,
                                                                            depth_multiplier=depth_multiplier,
                                                                            num_filters=512,
                                                                            kernel_size=(3, 3),
                                                                            strides=(2, 2),
                                                                            padding='SAME',
                                                                            batch_normlization=True,
                                                                            activation=tf.nn.relu6,
                                                                            pointwise_pruning=is_pruning,
                                                                            is_training=is_training,
                                                                            name="depthwise_separable_conv2d_6"
                                                                            )


        conv2d_7_depthwise, conv2d_7_pointwise = depthwise_separable_conv2d(conv2d_6_pointwise, None, None,
                                                                            depth_multiplier=depth_multiplier,
                                                                            num_filters=512,
                                                                            kernel_size=(3, 3),
                                                                            strides=(1, 1),
                                                                            padding='SAME',
                                                                            batch_normlization=True,
                                                                            activation=tf.nn.relu6,
                                                                            pointwise_pruning=is_pruning,
                                                                            is_training=is_training,
                                                                            name="depthwise_separable_conv2d_7"
                                                                            )
        conv2d_8_depthwise, conv2d_8_pointwise = depthwise_separable_conv2d(conv2d_7_pointwise, None, None,
                                                                            depth_multiplier=depth_multiplier,
                                                                            num_filters=512,
                                                                            kernel_size=(3, 3),
                                                                            strides=(1, 1),
                                                                            padding='SAME',
                                                                            batch_normlization=True,
                                                                            activation=tf.nn.relu6,
                                                                            pointwise_pruning=is_pruning,
                                                                            is_training=is_training,
                                                                            name="depthwise_separable_conv2d_8"
                                                                            )
        conv2d_9_depthwise, conv2d_9_pointwise = depthwise_separable_conv2d(conv2d_8_pointwise, None, None,
                                                                            depth_multiplier=depth_multiplier,
                                                                            num_filters=512,
                                                                            kernel_size=(3, 3),
                                                                            strides=(1, 1),
                                                                            padding='SAME',
                                                                            batch_normlization=True,
                                                                            activation=tf.nn.relu6,
                                                                            pointwise_pruning=is_pruning,
                                                                            is_training=is_training,
                                                                            name="depthwise_separable_conv2d_9"
                                                                            )
        conv2d_10_depthwise, conv2d_10_pointwise = depthwise_separable_conv2d(conv2d_9_pointwise, None, None,
                                                                            depth_multiplier=depth_multiplier,
                                                                            num_filters=512,
                                                                            kernel_size=(3, 3),
                                                                            strides=(1, 1),
                                                                            padding='SAME',
                                                                            batch_normlization=True,
                                                                            activation=tf.nn.relu6,
                                                                            pointwise_pruning=is_pruning,
                                                                            is_training=is_training,
                                                                            name="depthwise_separable_conv2d_10"
                                                                            )
        conv2d_11_depthwise, conv2d_11_pointwise = depthwise_separable_conv2d(conv2d_10_pointwise, None, None,
                                                                              depth_multiplier=depth_multiplier,
                                                                              num_filters=512,
                                                                              kernel_size=(3, 3),
                                                                              strides=(1, 1),
                                                                              padding='SAME',
                                                                              batch_normlization=True,
                                                                              activation=tf.nn.relu6,
                                                                              pointwise_pruning=is_pruning,
                                                                              is_training=is_training,
                                                                              name="depthwise_separable_conv2d_11"
                                                                              )
        conv2d_12_depthwise, conv2d_12_pointwise = depthwise_separable_conv2d(conv2d_11_pointwise, None, None,
                                                                              depth_multiplier=depth_multiplier,
                                                                              num_filters=1024,
                                                                              kernel_size=(3, 3),
                                                                              strides=(2, 2),
                                                                              padding='SAME',
                                                                              batch_normlization=True,
                                                                              activation=tf.nn.relu6,
                                                                              pointwise_pruning=is_pruning,
                                                                              is_training=is_training,
                                                                              name="depthwise_separable_conv2d_12"
                                                                              )
        conv2d_13_depthwise, conv2d_13_pointwise = depthwise_separable_conv2d(conv2d_12_pointwise, None, None,
                                                                              depth_multiplier=depth_multiplier,
                                                                              num_filters=1024,
                                                                              kernel_size=(3, 3),
                                                                              strides=(1, 1),
                                                                              padding='SAME',
                                                                              batch_normlization=True,
                                                                              activation=tf.nn.relu6,
                                                                              pointwise_pruning=is_pruning,
                                                                              is_training=is_training,
                                                                              name="depthwise_separable_conv2d_13"
                                                                              )

        kernel_size = _reduced_kernel_size_for_small_input(conv2d_13_pointwise, [7, 7])
        net = avg_pool_2d(conv2d_13_pointwise, kernel_size, stride=(1,1))
        net = dropout(net, dropout_keep_prob=dropout_keep_prob, is_training=is_training, name='Dropout_1b')
        logits = conv2d(net, num_filters=num_classes, kernel_size=(1,1), strides=(1,1), is_pruning=is_pruning, name="fc")
        if spatial_squeeze:
            logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

        if prediction_fn:
            predict = prediction_fn(logits, scope='Predictions')
            endpoints['prediction'] = predict

    return logits, endpoints