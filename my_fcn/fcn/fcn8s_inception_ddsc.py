from layers.layer_random_init import _conv_layer
from layers.layer_random_init import _conv_layer_without_relu
from layers.layer_random_init import _pooling_layer
from layers.layer_random_init import _upscore_layer
from layers.layer_random_init import _inception_layer
from layers.layers import conv_bn_relu
import tensorflow as tf
from tensorflow.contrib.keras.api.keras.layers import UpSampling2D
import numpy as np

#fcn8s的结构中引入inception
class fcn8s_inception_ddsc():
    def build(self, x, class_num, channel, wd, train=True):

        self.conv1_1 = _conv_layer(x, [3, 3, channel, 64], 'conv1_1')
        self.conv1_2 = _conv_layer(self.conv1_1, [3, 3, 64, 64], 'conv1_2')
        self.pooling1 = _pooling_layer(self.conv1_2, 'pooling1')

        self.conv2_1 = _conv_layer(self.pooling1, [3, 3, 64, 128], 'conv2_1')
        self.conv2_2 = _conv_layer(self.conv2_1, [3, 3, 128, 128], 'conv2_2')
        self.pooling2 = _pooling_layer(self.conv2_2, 'pooling2')

        # self.inception_block2_1 = _inception_layer(self.pooling1, in_c=64, out_1=32, out_21=32, out_22=32, out_31=32, out_32=64, out_33=64)
        # self.inception_block2_2 = _inception_layer(self.inception_block2_1, in_c=128, out_1=32, out_21=32, out_22=32, out_31=32,
        #                                            out_32=64, out_33=64)
        # self.pooling2 = _pooling_layer(self.inception_block2_2, 'pooling2')

        # self.conv3_1 = _conv_layer(self.pooling2, [3, 3, 128, 256], 'conv3_1')
        # self.conv3_2 = _conv_layer(self.conv3_1, [3, 3, 256, 256], 'conv3_2')
        # self.conv3_3 = _conv_layer(self.conv3_2, [3, 3, 256, 256], 'conv3_3')
        # self.pooling3 = _pooling_layer(self.conv3_3, 'pooling3')

        self.inception_block3_1 = _inception_layer('inception3_1', self.pooling2, in_c=128, out_1=64, out_21=64, out_22=64, out_31=64,
                                                 out_32=128, out_33=64, out_41=64, out_42=128, out_43=64)  # output channel 256
        self.inception_block3_2 = _inception_layer('inception3_2', self.inception_block3_1, in_c=256, out_1=64, out_21=64, out_22=64, out_31=64,
                                                 out_32=128, out_33=64, out_41=64, out_42=128, out_43=64)  # output channel 256
        self.inception_block3_3 = _inception_layer('inception3_3', self.inception_block3_2, in_c=256, out_1=64, out_21=64, out_22=64, out_31=64,
                                                 out_32=128, out_33=64, out_41=64, out_42=128, out_43=64)  # output channel 256
        self.pooling3 = _pooling_layer(self.inception_block3_3, 'pooling3')

        self.inception_block4_1 = _inception_layer('inception4_1', self.pooling3, in_c=256, out_1=128, out_21=128, out_22=128, out_31=128,
                                                 out_32=256, out_33=128, out_41=128, out_42=256, out_43=128)  # output 512
        self.inception_block4_2 = _inception_layer('inception4_2', self.inception_block4_1, in_c=512, out_1=128, out_21=128, out_22=128, out_31=128,
                                                 out_32=256, out_33=128, out_41=128, out_42=256, out_43=128)  # output 512
        self.inception_block4_3 = _inception_layer('inception4_3', self.inception_block4_2, in_c=512, out_1=128, out_21=128, out_22=128, out_31=128,
                                                 out_32=256, out_33=128, out_41=128, out_42=256, out_43=128)  # output 512
        self.pooling4 = _pooling_layer(self.inception_block4_3, 'pooling4')

        self.inception_block5_1 = _inception_layer('inception5_1', self.pooling4, in_c=512, out_1=128, out_21=128, out_22=128, out_31=128,
                                                 out_32=256, out_33=128, out_41=128, out_42=256, out_43=128)  # output 512
        self.inception_block5_2 = _inception_layer('inception5_2', self.inception_block5_1, in_c=512, out_1=128, out_21=128, out_22=128, out_31=128,
                                                 out_32=256, out_33=128, out_41=128, out_42=256, out_43=128)  # output 512
        self.inception_block5_3 = _inception_layer('inception5_3', self.inception_block5_2, in_c=512, out_1=128, out_21=128, out_22=128, out_31=128,
                                                 out_32=256, out_33=128, out_41=128, out_42=256, out_43=128)  # output 512
        self.pooling5 = _pooling_layer(self.inception_block5_3, 'pooling5')

        self.conv6 = _conv_layer(self.pooling5, [7, 7, 512, 4096], 'conv6')
        if train:
            self.conv6 = tf.nn.dropout(self.conv6, 0.5)
        self.conv7 = _conv_layer(self.conv6, [1, 1, 4096, 4096], 'conv7')
        if train:
            self.conv7 = tf.nn.dropout(self.conv7, 0.5)
        self.score_fr = _conv_layer_without_relu(self.conv7, [1, 1, 4096, 64], 'conv7_1x1conv', wd=wd)

        self.pooling4_conv = _conv_layer_without_relu(self.pooling4, [1, 1, 512, 64], 'pool4_1x1conv', wd=wd)

        self.pooling3_conv = _conv_layer_without_relu(self.pooling3, [1, 1, 256, 64], 'pool3_1x1conv', wd=wd)

        self.upconv5 = UpSampling2D((2, 2))(self.score_fr)#1/16
        self.upconv4 = UpSampling2D((2, 2))(self.pooling4_conv)#1/8
        self.upconv3 = UpSampling2D((8, 8))(self.pooling3_conv)#1
        self.conv5_feature = self.Dense_decoder_feature_generation(self.upconv5, name="gen1")

        self.upconv4 = self.fusion1(self.conv5_feature, self.upconv4)#1/8
        self.conv4_feature = self.Dense_decoder_feature_generation(self.upconv4, name="gen2")

        self.upconv3 = self.fusion2(self.conv5_feature, self.conv4_feature, self.upconv3)
        self.conv3_feature = self.Dense_decoder_feature_generation(self.upconv3, name="gen3")

        self.upscore = conv_bn_relu(self.conv3_feature, [3, 3, np.shape(self.conv3_feature)[3], 64], 1, name="final1")
        self.upscore = conv_bn_relu(self.upscore, [3, 3, 64, class_num], 1, name="final2")

        return self.upscore

    def Conv_Pool_Conv(self, inputs, name):
        with tf.variable_scope(name) as scope:
            D = np.shape(inputs)[3]
            outputs = conv_bn_relu(inputs, [3, 3, D, D], stride=1)
            outputs = tf.nn.max_pool(outputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling')
            outputs = conv_bn_relu(outputs, [3, 3, D, D//4], stride=1)
            return outputs

    def generation_stage(self, inputs, name):
        with tf.variable_scope(name) as scope:
            D = np.shape(inputs)[3]
            x = conv_bn_relu(inputs, [1, 1, D, D//2], 1, "c1")
            x = conv_bn_relu(x, [3, 3, D//2, D//2], 1, "c2")
            x = conv_bn_relu(x, [1, 1, D//2, D], 1, "c3")
            outputs = tf.add(inputs, x)
            return outputs

    def Dense_decoder_feature_generation(self, inputs, name):
        with tf.variable_scope(name) as scope:
            branch1 = self.Conv_Pool_Conv(inputs, "b1")
            branch2 = self.Conv_Pool_Conv(inputs, "b2")
            branch3 = self.Conv_Pool_Conv(inputs, "b3")
            branch4 = self.Conv_Pool_Conv(inputs, "b4")

            features = tf.concat([branch1, branch2, branch3, branch4], axis=3)
            D = np.shape(inputs)[3]
            features = conv_bn_relu(features, [3, 3, D, D], stride=1)

            for i in range(4):
                features = self.generation_stage(features, "stage"+str(i))
            features = UpSampling2D((2, 2))(features)
            features = conv_bn_relu(features, [3, 3, D, D], 1)
            return features

    def fusion1(self, conv5_feature, upconv4, name="fusion1"): #(1/16, 1/8)
        with tf.variable_scope(name) as scope:
            output_channel = np.shape(upconv4)[3]

            conv5_feature = conv_bn_relu(conv5_feature, [3,3, np.shape(conv5_feature)[3], output_channel], 1)
            conv5_feature = UpSampling2D((2,2))(conv5_feature)

            upconv4 = conv_bn_relu(upconv4, [3, 3, np.shape(upconv4)[3], output_channel], 1)

            output = tf.add(conv5_feature, upconv4)
            output = conv_bn_relu(output, [3, 3, output_channel, output_channel], 1)
            return output

    def fusion2(self, conv5_feature, conv4_feature, upconv3, name="fusion2"):#(1/16, 1/8, 1)
        with tf.variable_scope(name) as scope:
            output_channel = np.shape(upconv3)[3]

            conv5_feature = conv_bn_relu(conv5_feature, [3, 3, np.shape(conv5_feature)[3], output_channel], 1)
            conv5_feature = UpSampling2D((16, 16))(conv5_feature)

            conv4_feature = conv_bn_relu(conv4_feature, [3, 3, np.shape(conv4_feature)[3], output_channel], 1)
            conv4_feature = UpSampling2D((8, 8))(conv4_feature)

            upconv3 = conv_bn_relu(upconv3, [3, 3, np.shape(upconv3)[3], output_channel], 1)

            output = tf.add(conv5_feature, conv4_feature)
            output = tf.add(output, upconv3)
            output = conv_bn_relu(output, [3, 3, output_channel, output_channel], 1)
            return output