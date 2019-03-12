'''
covlution layer，pool layer，initialization。。。。
'''
from __future__ import division
import tensorflow as tf
import numpy as np
import cv2


# Weight initialization (Xavier's init)
def weight_xavier_init(shape, n_inputs, n_outputs, activefunction='sigomd', uniform=True, variable_name=None):
    with tf.device('/cpu:0'):
        if activefunction == 'sigomd':
            if uniform:
                init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
                initial = tf.random_uniform(shape, -init_range, init_range)
                return tf.get_variable(name=variable_name, initializer=initial, trainable=True)
            else:
                stddev = tf.sqrt(2.0 / (n_inputs + n_outputs))
                initial = tf.truncated_normal(shape, mean=0.0, stddev=stddev)
                return tf.get_variable(name=variable_name, initializer=initial, trainable=True)
        elif activefunction == 'relu':
            if uniform:
                init_range = tf.sqrt(6.0 / (n_inputs + n_outputs)) * np.sqrt(2)
                initial = tf.random_uniform(shape, -init_range, init_range)
                return tf.get_variable(name=variable_name, initializer=initial, trainable=True)
            else:
                stddev = tf.sqrt(2.0 / (n_inputs + n_outputs)) * np.sqrt(2)
                initial = tf.truncated_normal(shape, mean=0.0, stddev=stddev)
                return tf.get_variable(name=variable_name, initializer=initial, trainable=True)
        elif activefunction == 'tan':
            if uniform:
                init_range = tf.sqrt(6.0 / (n_inputs + n_outputs)) * 4
                initial = tf.random_uniform(shape, -init_range, init_range)
                return tf.get_variable(name=variable_name, initializer=initial, trainable=True)
            else:
                stddev = tf.sqrt(2.0 / (n_inputs + n_outputs)) * 4
                initial = tf.truncated_normal(shape, mean=0.0, stddev=stddev)
                return tf.get_variable(name=variable_name, initializer=initial, trainable=True)


# Bias initialization
def bias_variable(shape, variable_name=None):
    with tf.device('/cpu:0'):
        initial = tf.constant(0.1, shape=shape)
        return tf.get_variable(name=variable_name, initializer=initial, trainable=True)


# 3D convolution
def conv3d(x, W, stride=1):
    conv_3d = tf.nn.conv3d(x, W, strides=[1, stride, stride, stride, 1], padding='SAME')
    return conv_3d


# Max Pooling
def max_pool3d(x, depth=False):
    """
        depth flag:False is z axis is same between input and output,true is z axis is input is twice than output
        """
    if depth:
        pool3d = tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
    else:
        pool3d = tf.nn.max_pool3d(x, ksize=[1, 2, 2, 1, 1], strides=[1, 2, 2, 1, 1], padding='SAME')
    return pool3d


# Batch Normalization
def normalizationlayer(x, is_train, height=None, width=None, image_z=None, norm_type=None, G=16, esp=1e-5, scope=None):
    """
    :param x:input data with shap of[batch,height,width,channel]
    :param is_train:flag of normalizationlayer,True is training,False is Testing
    :param height:in some condition,the data height is in Runtime determined,such as through deconv layer and conv2d
    :param width:in some condition,the data width is in Runtime determined
    :param image_z:
    :param norm_type:normalization type:support"batch","group","None"
    :param G:in group normalization,channel is seperated with group number(G)
    :param esp:Prevent divisor from being zero
    :param scope:normalizationlayer scope
    :return:
    """
    with tf.name_scope(scope + norm_type):
        if norm_type == None:
            output = x
        elif norm_type == 'batch':
            # is_train is True when Training,is False when Testing
            output = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=is_train)
        elif norm_type == "group":
            # tranpose:[bs,z,h,w,c]to[bs,c,z,h,w]following the paper
            x = tf.transpose(x, [0, 4, 1, 2, 3])
            N, C, Z, H, W = x.get_shape().as_list()
            G = min(G, C)
            if H == None and W == None and Z == None:
                Z, H, W = image_z, height, width
            x = tf.reshape(x, [-1, G, C // G, Z, H, W])
            mean, var = tf.nn.moments(x, [2, 3, 4, 5], keep_dims=True)
            x = (x - mean) / tf.sqrt(var + esp)
            gama = tf.get_variable(scope + norm_type + 'group_gama', [C], initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable(scope + norm_type + 'group_beta', [C], initializer=tf.constant_initializer(0.0))
            gama = tf.reshape(gama, [1, C, 1, 1, 1])
            beta = tf.reshape(beta, [1, C, 1, 1, 1])
            output = tf.reshape(x, [-1, C, Z, H, W]) * gama + beta
            # tranpose:[bs,c,z,h,w]to[bs,z,h,w,c]following the paper
            output = tf.transpose(output, [0, 2, 3, 4, 1])
        return output


# resnet add_connect
def resnet_Add(x1, x2):
    if x1.get_shape().as_list()[4] != x2.get_shape().as_list()[4]:
        # Option A: Zero-padding
        residual_connection = x2 + tf.pad(x1, [[0, 0], [0, 0], [0, 0], [0, 0],
                                               [0, x2.get_shape().as_list()[4] -
                                                x1.get_shape().as_list()[4]]])
    else:
        residual_connection = x2 + x1
    return residual_connection


# Convert class labels from scalars to one-hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# 2 => [0 0 1 0 0 0 0 0 0 0]
def dense_to_one_hot(labels_dense, num_classes):
    """
    :param labels_dense:
    :param num_classes:
    label number must start from zero,such as:0,1,2,3,4,...
    :return:
    """
    num_labels = labels_dense.shape[0]
    labels_one_hot = np.zeros((num_labels, num_classes))
    # method one
    for index in range(num_labels):
        for classindex in range(num_classes):
            if labels_dense[index] == classindex:
                labels_one_hot[index, classindex] = 1
    # # method two have bug
    # index_offset = np.arange(num_labels) * num_classes
    # labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
