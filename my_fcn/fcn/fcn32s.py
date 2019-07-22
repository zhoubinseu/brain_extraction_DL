from layers.layer_random_init import _conv_layer
from layers.layer_random_init import _conv_layer_without_relu
from layers.layer_random_init import _pooling_layer
from layers.layer_random_init import _upscore_layer
import tensorflow as tf
from tensorflow.contrib.keras.api.keras.layers import ConvLSTM2D
import numpy as np

#fcn32s网络结构
class fcn32s():
    def build(self, x, class_num, channel, wd, train=True):
        # x: [batch_size,rows,cols,channel] batch_size = 256

        self.conv1_1 = _conv_layer(x, [3, 3, channel, 64], 'conv1_1')
        self.conv1_2 = _conv_layer(self.conv1_1, [3, 3, 64, 64], 'conv1_2')
        self.pooling1 = _pooling_layer(self.conv1_2, 'pooling1')

        self.conv2_1 = _conv_layer(self.pooling1, [3, 3, 64, 128], 'conv2_1')
        self.conv2_2 = _conv_layer(self.conv2_1, [3, 3, 128, 128], 'conv2_2')
        self.pooling2 = _pooling_layer(self.conv2_2, 'pooling2')

        self.conv3_1 = _conv_layer(self.pooling2, [3, 3, 128, 256], 'conv3_1')
        self.conv3_2 = _conv_layer(self.conv3_1, [3, 3, 256, 256], 'conv3_2')
        self.conv3_3 = _conv_layer(self.conv3_2, [3, 3, 256, 256], 'conv3_3')
        self.pooling3 = _pooling_layer(self.conv3_3, 'pooling3')

        self.conv4_1 = _conv_layer(self.pooling3, [3, 3, 256, 512], 'conv4_1')
        self.conv4_2 = _conv_layer(self.conv4_1, [3, 3, 512, 512], 'conv4_2')
        self.conv4_3 = _conv_layer(self.conv4_2, [3, 3, 512, 512], 'conv4_3')
        self.pooling4 = _pooling_layer(self.conv4_3, 'pooling4')

        self.conv5_1 = _conv_layer(self.pooling4, [3, 3, 512, 512], 'conv5_1')
        self.conv5_2 = _conv_layer(self.conv5_1, [3, 3, 512, 512], 'conv5_2')
        self.conv5_3 = _conv_layer(self.conv5_2, [3, 3, 512, 512], 'conv5_3')
        self.pooling5 = _pooling_layer(self.conv5_3, 'pooling5')

        self.conv6 = _conv_layer(self.pooling5, [7, 7, 512, 4096], 'conv6')
        if train:
            self.conv6 = tf.nn.dropout(self.conv6, 0.5)
        self.conv7 = _conv_layer(self.conv6, [1, 1, 4096, 4096], 'conv7')
        if train:
            self.conv7 = tf.nn.dropout(self.conv7, 0.5)
        self.score_fr = _conv_layer_without_relu(self.conv7, [1, 1, 4096, class_num], 'conv7_1x1conv', wd=wd)  #[256,4,8,2]

        self.upscore = _upscore_layer(self.score_fr, shape=tf.shape(x), ksize=64, stride=32,
                                      num_class=class_num, name='final_upscore')

        return self.upscore
