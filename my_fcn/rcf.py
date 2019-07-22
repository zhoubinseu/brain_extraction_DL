# edge detection cnn

import tensorflow as tf
import numpy as np
from rcf_utils import _conv_layers
from rcf_utils import _max_pooling
from rcf_utils import _up_sample



# a crf network based on vgg16
class crf_vgg16():
    def build(self, x, channel, train=True):
        #stage1
        self.conv1_1 = _conv_layers("conv1_1", x, [3, 3, channel, 64])
        self.conv1_2 = _conv_layers("conv1_2", self.conv1_1, [3, 3, 64, 64])
        self.pooling1 = _max_pooling("pooling1", self.conv1_2)

        self.conv1_1_down = _conv_layers("conv1_1_down", self.conv1_1, [1, 1, 64, 21])
        self.conv1_2_down = _conv_layers("conv1_2_down", self.conv1_2, [1, 1, 64, 21])
        self.sum1 = tf.add(self.conv1_1_down, self.conv1_2_down)
        self.score1 = _conv_layers("score1", self.sum1, [1, 1, 21, 1])

        #stage2
        self.conv2_1 = _conv_layers("conv2_1", self.pooling1, [3, 3, 64, 128])
        self.conv2_2 = _conv_layers("conv2_2", self.conv2_1, [3, 3, 128, 128])
        self.pooling2 = _max_pooling("pooling2", self.conv2_2)

        self.conv2_1_down = _conv_layers("conv2_1_down", self.conv2_1, [1, 1, 128, 21])
        self.conv2_2_down = _conv_layers("conv2_2_down", self.conv2_2, [1, 1, 128, 21])
        self.sum2 = tf.add(self.conv2_1_down, self.conv2_2_down)
        self.score2 = _conv_layers("score2", self.sum2, [1, 1, 21, 1])
        self.deconv2 = _up_sample("deconv2", self.score2, shape=tf.shape(x), ksize=4, stride=2, output_channel=1)

        #stage3
        self.conv3_1 = _conv_layers("conv3_1", self.pooling2, [3, 3, 128, 256])
        self.conv3_2 = _conv_layers("conv3_2", self.conv3_1, [3, 3, 256, 256])
        self.conv3_3 = _conv_layers("conv3_3", self.conv3_2, [3, 3, 256, 256])
        self.pooling3 = _max_pooling("pooling3", self.conv3_3)

        self.conv3_1_down = _conv_layers("conv3_1_down", self.conv3_1, [1, 1, 256, 21])
        self.conv3_2_down = _conv_layers("conv3_2_down", self.conv3_2, [1, 1, 256, 21])
        self.conv3_3_down = _conv_layers("conv3_3_down", self.conv3_3, [1, 1, 256, 21])
        self.tmp3 = tf.add(self.conv3_1_down, self.conv3_2_down)
        self.sum3 = tf.add(self.tmp3, self.conv3_3_down)
        self.score3 = _conv_layers("score3", self.sum3, [1, 1, 21, 1])
        self.deconv3 = _up_sample("deconv3", self.score3, shape=tf.shape(x), ksize=8, stride=4, output_channel=1)

        #stage4
        self.conv4_1 = _conv_layers("conv4_1", self.pooling3, [3, 3, 256, 512])
        self.conv4_2 = _conv_layers("conv4_2", self.conv4_1, [3, 3, 512, 512])
        self.conv4_3 = _conv_layers("conv4_3", self.conv4_2, [3, 3, 512, 512])
        self.pooling4 = _max_pooling("pooling4", self.conv4_3)

        self.conv4_1_down = _conv_layers("conv4_1_down", self.conv4_1, [1, 1, 512, 21])
        self.conv4_2_down = _conv_layers("conv4_2_down", self.conv4_2, [1, 1, 512, 21])
        self.conv4_3_down = _conv_layers("conv4_3_down", self.conv4_3, [1, 1, 512, 21])
        self.tmp4 = tf.add(self.conv4_1_down, self.conv4_2_down)
        self.sum4 = tf.add(self.tmp4, self.conv4_3_down)
        self.score4 = _conv_layers("score4", self.sum4, [1, 1, 21, 1])
        self.deconv4 = _up_sample("deconv4", self.score4, shape=tf.shape(x), ksize=16, stride=8, output_channel=1)

        #stage5
        self.conv5_1 = _conv_layers("conv5_1", self.pooling4, [3, 3, 512, 512])
        self.conv5_2 = _conv_layers("conv5_2", self.conv5_1, [3, 3, 512, 512])
        self.conv5_3 = _conv_layers("conv5_3", self.conv5_2, [3, 3, 512, 512])

        self.conv5_1_down = _conv_layers("conv5_1_down", self.conv5_1, [1, 1, 512, 21])
        self.conv5_2_down = _conv_layers("conv5_2_down", self.conv5_2, [1, 1, 512, 21])
        self.conv5_3_down = _conv_layers("conv5_3_down", self.conv5_3, [1, 1, 512, 21])
        self.tmp5 = tf.add(self.conv5_1_down, self.conv5_2_down)
        self.sum5 = tf.add(self.tmp5, self.conv5_3_down)
        self.score5 = _conv_layers("score5", self.sum5, [1, 1, 21, 1])
        self.deconv5 = _up_sample("deconv5", self.score5, shape=tf.shape(x), ksize=32, stride=16, output_channel=1)

        #fusion
        self.fuse = tf.concat([self.score1, self.deconv2, self.deconv3, self.deconv4, self.deconv5], axis=3)
        self.pred = _conv_layers("pred", self.fuse, [1, 1, 5, 1])

        return self.score1, self.deconv2, self.deconv3, self.deconv4, self.deconv5, self.pred