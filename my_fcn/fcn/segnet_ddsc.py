import tensorflow as tf
import numpy as np
from layers.layers import Conv_BN_Relu, Upsampling, Dense_decoder_feature_generation

class segnet_ddsc():
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
        self.decoder5_feature = Dense_decoder_feature_generation(self.deconv5_1, name="decoder5_feature")
        # decoder 4
        self.fusion4 = self.dense_decoder_fusion4(self.decoder5_feature, self.pooling4, name="fusion4")
        self.upsample4 = Upsampling(self.fusion4, tf.shape(self.pooling3), ksize=4, stride=2, output_channel=512, name="upsample4")
        self.deconv4_3 = Conv_BN_Relu(self.upsample4, [3, 3, 512, 512], "deconv4_3")
        self.deconv4_2 = Conv_BN_Relu(self.deconv4_3, [3, 3, 512, 512], "deconv4_2")
        self.deconv4_1 = Conv_BN_Relu(self.deconv4_2, [3, 3, 512, 256], "deconv4_1")
        self.decoder4_feature = Dense_decoder_feature_generation(self.deconv4_1, name="decoder4_feature")
        # decoder 3
        self.fusion3 = self.dense_decoder_fusion3(self.decoder5_feature, self.decoder4_feature, self.pooling3, name="fusion3")
        self.upsample3 = Upsampling(self.fusion3, tf.shape(self.pooling2), ksize=4, stride=2, output_channel=256, name="upsample3")
        self.deconv3_3 = Conv_BN_Relu(self.upsample3, [3, 3, 256, 256], "deconv3_3")
        self.deconv3_2 = Conv_BN_Relu(self.deconv3_3, [3, 3, 256, 256], "deconv3_2")
        self.deconv3_1 = Conv_BN_Relu(self.deconv3_2, [3, 3, 256, 128], "deconv3_1")
        self.decoder3_feature = Dense_decoder_feature_generation(self.deconv3_1, name="decoder3_feature")
        # decoder 2
        self.fusion2 = self.dense_decoder_fusion2(self.decoder5_feature, self.decoder4_feature, self.decoder3_feature, self.pooling2, name="fusion2")
        self.upsample2 = Upsampling(self.fusion2, tf.shape(self.pooling1), ksize=4, stride=2, output_channel=128, name="upsample2")
        self.deconv2_2 = Conv_BN_Relu(self.upsample2, [3, 3, 128, 128], "deconv2_2")
        self.deconv2_1 = Conv_BN_Relu(self.deconv2_2, [3, 3, 128, 64], "deconv2_1")
        self.decoder2_feature = Dense_decoder_feature_generation(self.deconv2_1, name="decoder2_feature")
        # decoder 1
        self.fusion1 = self.dense_decoder_fusion1(self.decoder5_feature, self.decoder4_feature, self.decoder3_feature, self.decoder2_feature, self.pooling1, name="fusion1")
        self.upsample1 = Upsampling(self.fusion1, tf.shape(input), ksize=4, stride=2, output_channel=64, name="upsample1")
        self.deconv1_2 = Conv_BN_Relu(self.upsample1, [3, 3, 64, 64], "deconv1_2")
        self.deconv1_1 = Conv_BN_Relu(self.deconv1_2, [3, 3, 64, 64], "deconv1_1")
        self.decoder1_feature = Dense_decoder_feature_generation(self.deconv1_1, name="decoder1_feature")
        self.pred = Conv_BN_Relu(self.decoder1_feature, [3, 3, 64, class_num], "pred")
        # pred will be processed by softmax and compute loss
        return self.pred


    def dense_decoder_fusion4(self, decoder5_feature, encoder4, name=None):#1/16, 1/16
        with tf.variable_scope(name):
            output_channel = 512
            decoder5_feature = Conv_BN_Relu(decoder5_feature, [3, 3, 512, output_channel], "dec5")
            encoder4 = Conv_BN_Relu(encoder4, [3, 3, 512, output_channel], "enc4")
            output = tf.add(decoder5_feature, encoder4)
            return output


    
    def dense_decoder_fusion3(self, decoder5_feature, decoder4_feature, encoder3, name=None):#1/16, 1/8, 1/8
        with tf.variable_scope(name):
            output_channel = 256
            decoder5_feature = Conv_BN_Relu(decoder5_feature, [3, 3, 512, output_channel], "dec5")
            decoder5_feature = Upsampling(decoder5_feature, tf.shape(encoder3), ksize=4, stride=2, output_channel=output_channel, name="up_dec5")
            decoder4_feature = Conv_BN_Relu(decoder4_feature, [3, 3, 256, output_channel], "dec4")
            encoder3 = Conv_BN_Relu(encoder3, [3, 3, 256, output_channel], "enc3")
            output = tf.add(decoder5_feature, decoder4_feature)
            output = tf.add(output, encoder3)
            return output
            

    def dense_decoder_fusion2(self, decoder5_feature, decoder4_feature, decoder3_feature, encoder2, name=None):#1/16, 1/8, 1/4, 1/4
        with tf.variable_scope(name):
            output_channel = 128
            decoder5_feature = Conv_BN_Relu(decoder5_feature, [3, 3, 512, output_channel], "dec5")
            decoder5_feature = Upsampling(decoder5_feature, tf.shape(encoder2), ksize=8, stride=4, output_channel=output_channel, name="up_dec5")
            decoder4_feature = Conv_BN_Relu(decoder4_feature, [3, 3, 256, output_channel], "dec4")
            decoder4_feature = Upsampling(decoder4_feature, tf.shape(encoder2), ksize=4, stride=2, output_channel=output_channel, name="up_dec4")
            decoder3_feature = Conv_BN_Relu(decoder3_feature, [3, 3, 128, output_channel], "dec3")
            encoder2 = Conv_BN_Relu(encoder2, [3, 3, 128, output_channel], "enc2")
            output = tf.add(decoder5_feature, decoder4_feature)
            output = tf.add(output, decoder3_feature)
            output = tf.add(output, encoder2)
            return output

    def dense_decoder_fusion1(self, decoder5_feature, decoder4_feature, decoder3_feature, decoder2_feature, encoder1, name=None):#1/16, 1/8, 1/4, 1/2, 1/2
        with tf.variable_scope(name):
            output_channel = 64
            decoder5_feature = Conv_BN_Relu(decoder5_feature, [3, 3, 512, output_channel], "dec5")
            decoder5_feature = Upsampling(decoder5_feature, tf.shape(encoder1), ksize=16, stride=8, output_channel=output_channel, name="up_dec5")
            decoder4_feature = Conv_BN_Relu(decoder4_feature, [3, 3, 256, output_channel], "dec4")
            decoder4_feature = Upsampling(decoder4_feature, tf.shape(encoder1), ksize=8, stride=4, output_channel=output_channel, name="up_dec4")
            decoder3_feature = Conv_BN_Relu(decoder3_feature, [3, 3, 128, output_channel], "dec3")
            decoder3_feature = Upsampling(decoder3_feature, tf.shape(encoder1), ksize=4, stride=2, output_channel=output_channel, name="up_dec3")
            decoder2_feature = Conv_BN_Relu(decoder2_feature, [3, 3, 64, output_channel], "dec2")
            encoder1 = Conv_BN_Relu(encoder1, [3, 3, 64, output_channel], "enc1")
            output = tf.add(decoder5_feature, decoder4_feature)
            output = tf.add(output, decoder3_feature)
            output = tf.add(output, decoder2_feature)
            output = tf.add(output, encoder1)
            return output
