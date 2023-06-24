import os
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from abc import ABC, abstractmethod


class Network(ABC):
    def __init__(self):
        self.inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
        self.outputs = self.inputs
        self.normalize = False
        self.variables = {}
        self.setup()

    def predict(self, session, images):
        predictions = []

        for image in images:
            if self.normalize:
                batch = np.array([image / 255.0])
            else:
                batch = np.array([image])

            predictions.append(self.outputs.eval(feed_dict={self.inputs: batch}, session=session)[0])

        return predictions

    def score(self, session, images, labels, top_k):
        if type(top_k) is int:
            top_k = [top_k]

        predictions = self.predict(session, images)
        correct_predictions = [0 for _ in range(len(top_k))]

        for i in range(len(top_k)):
            for prediction, label in zip(predictions, labels):
                predicted_labels = np.argsort(prediction)[::-1][:top_k[i]]

                if label in predicted_labels:
                    correct_predictions[i] += 1

        scores = [n / len(predictions) for n in correct_predictions]

        if len(scores) == 1:
            return scores[0]
        else:
            return scores

    def convolution(self, name, shape, stride=1, padding='SAME', groups=1):
        weights = tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01))
        biases = tf.Variable(tf.constant(0.0, shape=[shape[3]], dtype=tf.float32))

        if groups == 1:
            self.outputs = tf.nn.conv2d(self.outputs, weights, [1, stride, stride, 1], padding=padding)
        else:
            grouped_inputs = tf.split(self.outputs, groups, 3)
            grouped_weights = tf.split(weights, groups, 3)
            grouped_outputs = [tf.nn.conv2d(i, w, [1, stride, stride, 1], padding=padding)
                               for i, w in zip(grouped_inputs, grouped_weights)]

            self.outputs = tf.concat(grouped_outputs, 3)

        self.outputs = tf.nn.relu(tf.nn.bias_add(self.outputs, biases))
        self.variables['%s_W' % name] = weights
        self.variables['%s_b' % name] = biases

        return self

    def pooling(self, k=2, stride=2, padding='SAME'):
        self.outputs = tf.nn.max_pool(self.outputs, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding=padding)

        return self

    def fully_connected(self, name, shape, activation):
        if shape[0] == -1:
            shape[0] = int(np.prod(self.outputs.get_shape()[1:]))

        weights = tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01))
        biases = tf.Variable(tf.constant(1.0, shape=[shape[1]], dtype=tf.float32))

        self.outputs = tf.nn.bias_add(tf.matmul(self.outputs, weights), biases)

        if activation is not None:
            self.outputs = activation(self.outputs)

        self.variables['%s_W' % name] = weights
        self.variables['%s_b' % name] = biases

        return self

    def subtract_mean(self):
        self.outputs = self.outputs - tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3])

        return self

    def flatten(self):
        flattened_shape = int(np.prod(self.outputs.get_shape()[1:]))

        self.outputs = tf.reshape(self.outputs, [-1, flattened_shape])

        return self

    def reverse_channels(self):
        channels = tf.unstack(self.outputs, axis=-1)
        self.outputs = tf.stack([channels[2], channels[1], channels[0]], axis=-1)

        return self

    def pad(self, n):
        self.outputs = tf.pad(self.inputs, [[0, 0], [0, n], [0, n], [0, 0]])

        return self

    def local_response_normalization(self, depth_radius, alpha, beta, bias):
        self.outputs = tf.nn.local_response_normalization(self.outputs, depth_radius=depth_radius,
                                                          alpha=alpha, beta=beta, bias=bias)

        return self

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def load(self, session):
        pass


class AlexNet(Network):
    # based on http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

    def setup(self):
        self.pad(n=3). \
            subtract_mean(). \
            reverse_channels(). \
            convolution('conv1', [11, 11, 3, 96], stride=4, groups=1). \
            local_response_normalization(depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0). \
            pooling(k=3, stride=2, padding='VALID'). \
            convolution('conv2', [5, 5, 48, 256], stride=1, groups=2). \
            local_response_normalization(depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0). \
            pooling(k=3, stride=2, padding='VALID'). \
            convolution('conv3', [3, 3, 256, 384], stride=1, groups=1). \
            convolution('conv4', [3, 3, 192, 384], stride=1, groups=2). \
            convolution('conv5', [3, 3, 192, 256], stride=1, groups=2). \
            pooling(k=3, stride=2, padding='VALID'). \
            flatten(). \
            fully_connected('fc6', [-1, 4096], activation=tf.nn.relu). \
            fully_connected('fc7', [4096, 4096], activation=tf.nn.relu). \
            fully_connected('fc8', [4096, 1000], activation=tf.nn.softmax)

    def load(self, session):
        weights = np.load(os.path.join(os.path.dirname(__file__), 'models', 'AlexNet.npy'), allow_pickle=True, encoding='latin1').item()

        for key in weights.keys():
            session.run(self.variables['%s_W' % key].assign(weights[key][0]))
            session.run(self.variables['%s_b' % key].assign(weights[key][1]))


class VGGNet(Network):
    # based on http://www.cs.toronto.edu/~frossard/post/vgg16/

    def setup(self):
        self.subtract_mean(). \
            convolution('conv1_1', [3, 3, 3, 64]). \
            convolution('conv1_2', [3, 3, 64, 64]). \
            pooling(). \
            convolution('conv2_1', [3, 3, 64, 128]). \
            convolution('conv2_2', [3, 3, 128, 128]). \
            pooling(). \
            convolution('conv3_1', [3, 3, 128, 256]). \
            convolution('conv3_2', [3, 3, 256, 256]). \
            convolution('conv3_3', [3, 3, 256, 256]). \
            pooling(). \
            convolution('conv4_1', [3, 3, 256, 512]). \
            convolution('conv4_2', [3, 3, 512, 512]). \
            convolution('conv4_3', [3, 3, 512, 512]). \
            pooling(). \
            convolution('conv5_1', [3, 3, 512, 512]). \
            convolution('conv5_2', [3, 3, 512, 512]). \
            convolution('conv5_3', [3, 3, 512, 512]). \
            pooling(). \
            flatten(). \
            fully_connected('fc6', [-1, 4096], activation=tf.nn.relu). \
            fully_connected('fc7', [4096, 4096], activation=tf.nn.relu). \
            fully_connected('fc8', [4096, 1000], activation=tf.nn.softmax)

    def load(self, session):
        weights = np.load(os.path.join(os.path.dirname(__file__), 'models', 'VGGNet16.npz'), allow_pickle=True)

        for key in weights.keys():
            session.run(self.variables[key].assign(weights[key]))


class ResNet(Network):
    # based on https://github.com/ry/tensorflow-resnet

    def setup(self):
        self.normalize = True

    def load(self, session):
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'ResNet')
        saver = tf.train.import_meta_graph(os.path.join(model_path, 'ResNet-L50.meta'))
        saver.restore(session, os.path.join(model_path, 'ResNet-L50.ckpt'))

        graph = tf.get_default_graph()

        self.outputs = graph.get_tensor_by_name('prob:0')
        self.inputs = graph.get_tensor_by_name('images:0')
