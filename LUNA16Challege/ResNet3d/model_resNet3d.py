'''

'''
from ResNet3d.layer import (conv3d, normalizationlayer, max_pool3d, resnet_Add, weight_xavier_init, bias_variable,
                            dense_to_one_hot)
import tensorflow as tf
import numpy as np
import os


def conv_relu_drop(x, kernal, drop, phase, image_z=None, height=None, width=None, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
        conv = conv3d(x, W) + B
        conv = normalizationlayer(conv, is_train=phase, height=height, width=width, image_z=image_z, norm_type='group',
                                  scope=scope)
        conv = tf.nn.dropout(tf.nn.relu(conv), drop)
        return conv


def full_connected_relu_drop(x, kernal, drop, activefunction='relu', scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        FC = tf.matmul(x, W) + B
        if activefunction == 'relu':
            FC = tf.nn.relu(FC)
            FC = tf.nn.dropout(FC, drop)
        elif activefunction == 'softmax':
            FC = tf.nn.softmax(FC)
        return FC


def _create_conv_net(X, image_z, image_width, image_height, image_channel, drop, phase, n_class=1):
    inputX = tf.reshape(X, [-1, image_z, image_width, image_height, image_channel])  # shape=(?, 32, 32, 1)
    # Vnet model
    # layer1->convolution
    layer0 = conv_relu_drop(x=inputX, kernal=(3, 3, 3, image_channel, 16), drop=drop, phase=phase, scope='layer0')
    layer1 = conv_relu_drop(x=layer0, kernal=(3, 3, 3, 16, 16), drop=drop, phase=phase, scope='layer1')
    layer1 = resnet_Add(x1=layer0, x2=layer1)
    # down sampling1
    down1 = max_pool3d(x=layer1, depth=True)
    # down1 = down_sampling(x=layer1, kernal=(3, 3, 3, 32, 64), drop=drop, scope='down1')
    # layer2->convolution
    layer2 = conv_relu_drop(x=down1, kernal=(3, 3, 3, 16, 32), drop=drop, phase=phase, scope='layer2_1')
    layer2 = conv_relu_drop(x=layer2, kernal=(3, 3, 3, 32, 32), drop=drop, phase=phase, scope='layer2_2')
    layer2 = resnet_Add(x1=down1, x2=layer2)
    # down sampling2
    down2 = max_pool3d(x=layer2, depth=True)
    # down2 = down_sampling(x=layer2, kernal=(3, 3, 3, 64, 128), drop=drop, scope='down2')
    # layer3->convolution
    layer3 = conv_relu_drop(x=down2, kernal=(3, 3, 3, 32, 64), drop=drop, phase=phase, scope='layer3_1')
    layer3 = conv_relu_drop(x=layer3, kernal=(3, 3, 3, 64, 64), drop=drop, phase=phase, scope='layer3_2')
    layer3 = resnet_Add(x1=down2, x2=layer3)
    # down sampling3
    down3 = max_pool3d(x=layer3, depth=True)
    # down3 = down_sampling(x=layer3, kernal=(3, 3, 3, 128, 256), drop=drop, scope='down3')
    # layer4->convolution
    layer4 = conv_relu_drop(x=down3, kernal=(3, 3, 3, 64, 128), drop=drop, phase=phase, scope='layer4_1')
    layer4 = conv_relu_drop(x=layer4, kernal=(3, 3, 3, 128, 128), drop=drop, phase=phase, scope='layer4_2')
    layer4 = resnet_Add(x1=down3, x2=layer4)
    # down sampling4
    down4 = max_pool3d(x=layer4, depth=True)
    # down4 = down_sampling(x=layer4, kernal=(3, 3, 3, 256, 512), drop=drop, scope='down4')
    # layer5->convolution
    layer5 = conv_relu_drop(x=down4, kernal=(3, 3, 3, 128, 256), drop=drop, phase=phase, scope='layer5_1')
    layer5 = conv_relu_drop(x=layer5, kernal=(3, 3, 3, 256, 256), drop=drop, phase=phase, scope='layer5_2')
    layer5 = resnet_Add(x1=down4, x2=layer5)
    # layer6->FC1
    layer6 = tf.reshape(layer5, [-1, 3 * 3 * 3 * 256])  # shape=(?, 512)
    layer6 = full_connected_relu_drop(x=layer6, kernal=(3 * 3 * 3 * 256, 512), drop=drop, activefunction='relu',
                                      scope='fc1')
    # layer7->output
    output = full_connected_relu_drop(x=layer6, kernal=(512, n_class), drop=drop, activefunction='regression',
                                      scope='output')
    return output


# Serve data by batches
def _next_batch(train_images, train_labels, batch_size, index_in_epoch):
    start = index_in_epoch
    index_in_epoch += batch_size

    num_examples = train_images.shape[0]
    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end], index_in_epoch


class ResNet3dModule(object):
    """
        A ResNet3d implementation
        :param image_height: number of height in the input image
        :param image_width: number of width in the input image
        :param image_depth: number of depth in the input image
        :param channels: number of channels in the input image
        :param costname: name of the cost function.Default is "dice coefficient"
    """

    def __init__(self, image_height, image_width, image_depth, channels=1, n_class=2, costname="cross_entropy",
                 inference=False, model_path=None):
        self.image_width = image_width
        self.image_height = image_height
        self.image_depth = image_depth
        self.channels = channels
        self.n_class = n_class

        self.X = tf.placeholder("float", shape=[None, self.image_depth, self.image_height, self.image_width,
                                                self.channels])
        self.Y_gt = tf.placeholder("float", shape=[None, self.n_class])
        self.lr = tf.placeholder('float')
        self.phase = tf.placeholder(tf.bool)
        self.drop = tf.placeholder('float')

        self.Y_pred_logits = _create_conv_net(self.X, self.image_depth, self.image_width, self.image_height,
                                              self.channels, self.drop, self.phase, n_class=n_class)
        self.cost = self.__get_cost(costname, self.Y_pred_logits)

        self.Y_pred = tf.nn.softmax(self.Y_pred_logits)
        self.predict = tf.argmax(self.Y_pred, 1)
        self.accuracy = self.__get_accuracy(self.Y_pred)
        if inference:
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            self.sess = tf.InteractiveSession()
            self.sess.run(init)
            saver.restore(self.sess, model_path)

    def __get_cost(self, cost_name, Y_pred):
        if cost_name == "cross_entropy":
            # the function first calculate softmax then calculate the
            # cross_entropy(-tf.reduce_sum(self.Y_gt*tf.log(Y_pred))),
            # logits don't through the tf.nn,softmax function
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_gt, logits=Y_pred))
            return cost

    def __get_accuracy(self, Y_pred):
        correct_predict = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(self.Y_gt, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, 'float'))
        return accuracy

    def train(self, train_images, train_lanbels, model_path, logs_path, learning_rate,
              dropout_conv=0.8, train_epochs=5, batch_size=1):
        # label one_hot coding
        label_counts = np.unique(train_lanbels).shape[0]
        train_labels_onehot = dense_to_one_hot(train_lanbels, label_counts)
        train_labels_onehot = train_labels_onehot.astype(np.float)
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if not os.path.exists(logs_path + "model\\"):
            os.makedirs(logs_path + "model\\")
        model_path = logs_path + "model\\" + model_path

        # update the moving average of batch norm before finish the training step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # ensures that we execute the update_ops before performing the train_step
            train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)

        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        merged_summary_op = tf.summary.merge_all()
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        sess.run(init)

        DISPLAY_STEP = 1
        index_in_epoch = 0

        train_epochs = train_images.shape[0] * train_epochs
        for i in range(train_epochs):
            # get new batch
            batch_xs_path, batch_ys, index_in_epoch = _next_batch(train_images, train_labels_onehot, batch_size,
                                                                  index_in_epoch)
            batch_xs = np.empty((len(batch_xs_path), self.image_depth, self.image_height, self.image_width,
                                 self.channels))
            for num in range(len(batch_xs_path)):
                batchimage = np.reshape(np.load(batch_xs_path[num][0]),
                                        (self.image_depth, self.image_height, self.image_width, self.channels))
                batch_xs[num, :, :, :] = batchimage
            # Extracting images and labels from given data
            batch_xs = batch_xs.astype(np.float)
            batch_ys = batch_ys.astype(np.float)
            # Normalize from [0:255] => [0.0:1.0]
            batch_xs = np.multiply(batch_xs, 1.0 / 255.0)
            # check progress on every 1st,2nd,...,10th,20th,...,100th... step
            if i % DISPLAY_STEP == 0 or (i + 1) == train_epochs:
                train_loss, train_accuracy = sess.run([self.cost, self.accuracy],
                                                      feed_dict={self.X: batch_xs[batch_size // 5:],
                                                                 self.Y_gt: batch_ys[batch_size // 5:],
                                                                 self.lr: learning_rate,
                                                                 self.drop: dropout_conv,
                                                                 self.phase: 1})
                validataion_accuracy = self.accuracy.eval(feed_dict={self.X: batch_xs[0:batch_size // 5],
                                                                     self.Y_gt: batch_ys[0:batch_size // 5],
                                                                     self.lr: learning_rate,
                                                                     self.drop: dropout_conv,
                                                                     self.phase: 1})
                print('epochs %d training_loss ,training_accuracy,validation_accuracy => %.5f,%.5f,%5f ' % (
                    i, train_loss, train_accuracy, validataion_accuracy))
                save_path = saver.save(sess, model_path, global_step=i)
                print("Model saved in file:", save_path)
                if i % (DISPLAY_STEP * 10) == 0 and i:
                    DISPLAY_STEP *= 10

                    # train on batch
            _, summary = sess.run([train_op, merged_summary_op], feed_dict={self.X: batch_xs,
                                                                            self.Y_gt: batch_ys,
                                                                            self.lr: learning_rate,
                                                                            self.drop: dropout_conv,
                                                                            self.phase: 1})
            summary_writer.add_summary(summary, i)
        summary_writer.close()

        save_path = saver.save(sess, model_path)
        print("Model saved in file:", save_path)

    def prediction(self, test_images):
        test_images = np.reshape(test_images, (
            test_images.shape[0], test_images.shape[1], test_images.shape[2], test_images.shape[3], 1))
        test_images = test_images.astype(np.float)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        predictvalue = np.zeros(test_images.shape[0])
        predict_probvalue = np.zeros(test_images.shape[0], np.float32)
        y_dummy = np.empty((test_images.shape[0], self.n_class))
        for i in range(test_images.shape[0]):
            predictvaluetmp, predict_probvaluetmp = self.sess.run([self.predict, self.Y_pred],
                                                                  feed_dict={self.X: [test_images[i]],
                                                                             self.Y_gt: y_dummy,
                                                                             self.drop: 1,
                                                                             self.phase: 1})
            predictvalue[i], predict_probvalue[i] = predictvaluetmp, predict_probvaluetmp[0][1]
        return predictvalue, predict_probvalue
