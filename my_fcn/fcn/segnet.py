import tensorflow as tf
import numpy as np
from layers.layers import Conv_BN_Relu, Upsampling

class segnet():
    def build(self, input, input_channel, class_num=2):
        self.pool_size = [1,2,2,1]
        self.pool_stride = [1,2,2,1]
        
        # encoder

        # encoder 1
        self.conv1_1 = Conv_BN_Relu(input, [3, 3, input_channel, 64], "conv1_1")
        self.conv1_2 = Conv_BN_Relu(self.conv1_1, [3, 3, 64, 64], "conv1_2")
        self.pooling1 = tf.nn.max_pool(self.conv1_2, self.pool_size, self.pool_stride, padding="SAME", name="pooling1")
        # encoder 2
        self.conv2_1 = Conv_BN_Relu(self.pooling1, [3, 3, 64, 128], "conv2_1")
        self.conv2_2 = Conv_BN_Relu(self.conv2_1, [3, 3, 128, 128], "conv2_2")
        self.pooling2 = tf.nn.max_pool(self.conv2_2, self.pool_size, self.pool_stride, padding="SAME", name="pooling2")
        # encoder 3
        self.conv3_1 = Conv_BN_Relu(self.pooling2, [3, 3, 128, 256], "conv3_1")
        self.conv3_2 = Conv_BN_Relu(self.conv3_1, [3, 3, 256, 256], "conv3_2")
        self.conv3_3 = Conv_BN_Relu(self.conv3_2, [3, 3, 256, 256], "conv3_3")
        self.pooling3 = tf.nn.max_pool(self.conv3_3, self.pool_size, self.pool_stride, padding="SAME", name="pooling3")
        # encoder 4
        self.conv4_1 = Conv_BN_Relu(self.pooling3, [3, 3, 256, 512], "conv4_1")
        self.conv4_2 = Conv_BN_Relu(self.conv4_1, [3, 3, 512, 512], "conv4_2")
        self.conv4_3 = Conv_BN_Relu(self.conv4_2, [3, 3, 512, 512], "conv4_3")
        self.pooling4 = tf.nn.max_pool(self.conv4_3, self.pool_size, self.pool_stride, padding="SAME", name="pooling4")
        # encoder 5
        self.conv5_1 = Conv_BN_Relu(self.pooling4, [3, 3, 512, 512], "conv5_1")
        self.conv5_2 = Conv_BN_Relu(self.conv5_1, [3, 3, 512, 512], "conv5_2")
        self.conv5_3 = Conv_BN_Relu(self.conv5_2, [3, 3, 512, 512], "conv5_3")
        self.pooling5 = tf.nn.max_pool(self.conv5_3, self.pool_size, self.pool_stride, padding="SAME", name="pooling5")

        # decoder

        # decoder 5
        self.upsample5 = Upsampling(self.pooling5, tf.shape(self.pooling4), ksize=4, stride=2, output_channel=512, name="upsample5")
        self.deconv5_3 = Conv_BN_Relu(self.upsample5, [3, 3, 512, 512], "deconv5_3")
        self.deconv5_2 = Conv_BN_Relu(self.deconv5_3, [3, 3, 512, 512], "deconv5_2")
        self.deconv5_1 = Conv_BN_Relu(self.deconv5_2, [3, 3, 512, 512], "deconv5_1")
        # decoder 4
        self.upsample4 = Upsampling(self.deconv5_1, tf.shape(self.pooling3), ksize=4, stride=2, output_channel=512, name="upsample4")
        self.deconv4_3 = Conv_BN_Relu(self.upsample4, [3, 3, 512, 512], "deconv4_3")
        self.deconv4_2 = Conv_BN_Relu(self.deconv4_3, [3, 3, 512, 512], "deconv4_2")
        self.deconv4_1 = Conv_BN_Relu(self.deconv4_2, [3, 3, 512, 256], "deconv4_1")
        # decoder 3
        self.upsample3 = Upsampling(self.deconv4_1, tf.shape(self.pooling2), ksize=4, stride=2, output_channel=256, name="upsample3")
        self.deconv3_3 = Conv_BN_Relu(self.upsample3, [3, 3, 256, 256], "deconv3_3")
        self.deconv3_2 = Conv_BN_Relu(self.deconv3_3, [3, 3, 256, 256], "deconv3_2")
        self.deconv3_1 = Conv_BN_Relu(self.deconv3_2, [3, 3, 256, 128], "deconv3_1")
        # decoder 2
        self.upsample2 = Upsampling(self.deconv3_1, tf.shape(self.pooling1), ksize=4, stride=2, output_channel=128, name="upsample2")
        self.deconv2_2 = Conv_BN_Relu(self.upsample2, [3, 3, 128, 128], "deconv2_2")
        self.deconv2_1 = Conv_BN_Relu(self.deconv2_2, [3, 3, 128, 64], "deconv2_1")
        # decoder 1
        self.upsample1 = Upsampling(self.deconv2_1, tf.shape(input), ksize=4, stride=2, output_channel=64, name="upsample1")
        self.deconv1_2 = Conv_BN_Relu(self.upsample1, [3, 3, 64, 64], "deconv1_2")
        self.deconv1_1 = Conv_BN_Relu(self.deconv1_2, [3, 3, 64, class_num], "deconv1_1")
        # deconv1_1 will be processed by softmax and compute loss
        return self.deconv1_1