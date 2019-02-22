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


# 3D upsampling
def upsample3d(x, scale_factor, scope=None):
    ''''
    X shape is [nsample,dim,rows, cols, channel]
    out shape is[nsample,dim*scale_factor,rows*scale_factor, cols*scale_factor, channel]
    '''
    x_shape = tf.shape(x)
    k = tf.ones([scale_factor, scale_factor, scale_factor, x_shape[-1], x_shape[-1]])
    # note k.shape = [dim,rows, cols, depth_in, depth_output]
    output_shape = tf.stack(
        [x_shape[0], x_shape[1] * scale_factor, x_shape[2] * scale_factor, x_shape[3] * scale_factor, x_shape[4]])
    upsample = tf.nn.conv3d_transpose(value=x, filter=k, output_shape=output_shape,
                                      strides=[1, scale_factor, scale_factor, scale_factor, 1],
                                      padding='SAME', name=scope)
    return upsample


# 3D deconvolution
def deconv3d(x, W, samefeature=False, depth=False):
    """
    depth flag:False is z axis is same between input and output,true is z axis is input is twice than output
    """
    x_shape = tf.shape(x)
    if depth:
        if samefeature:
            output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] * 2, x_shape[4]])
        else:
            output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] * 2, x_shape[4] // 2])
        deconv = tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, 2, 2, 2, 1], padding='SAME')
    else:
        if samefeature:
            output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3], x_shape[4]])
        else:
            output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3], x_shape[4] // 2])
        deconv = tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, 2, 2, 1, 1], padding='SAME')
    return deconv


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


# Unet crop and concat
def crop_and_concat(x1, x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2,
               (x1_shape[2] - x2_shape[2]) // 2, (x1_shape[3] - x2_shape[3]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 4)


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
            output = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_train=is_train)
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


def save_images(images, size, path):
    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j * h:j * h + h, i * w:i * w + w] = image
    result = merge_img * 255.
    result = np.clip(result, 0, 255).astype('uint8')
    return cv2.imwrite(path, result)
