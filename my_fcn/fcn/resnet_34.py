import tensorflow as tf
import numpy as np
from layers.layers import residual_block
from layers.layer_random_init import get_bn_result
from layers.layer_random_init import _upscore_layer
from layers.layer_random_init import _conv_layer_without_relu
from layers.layer_random_init import _conv_layer
from layers.layer_random_init import _pooling_layer


#使用Resnet-34(conv1部分进行了修改)进行语义分割
class resnet_34():
    def build(self, x, class_num, channel, wd, train=True):
        #conv1
        self.conv1_1 = _conv_layer(x, [3,3,channel, 64], 'conv1_1')
        self.conv1_2 = _conv_layer(self.conv1_1, [3,3,64,64], 'conv1_2')
        self.pooling1 = _pooling_layer(self.conv1_2, 'pooling1')
        print("pooling1：", np.shape(self.pooling1))
        self.conv2_1 = _conv_layer(self.pooling1, [3, 3, 64, 64], 'conv2_1')
        self.conv2_2 = _conv_layer(self.conv2_1, [3, 3, 64, 64], 'conv2_2')
        self.pooling2 = _pooling_layer(self.conv2_2, 'pooling2')
        print("pooling2: ", np.shape(self.pooling2))
        #conv2_x   3 blocks
        self.conv2_block1 = residual_block("conv2_block1", self.pooling2, 64, True)
        self.conv2_block2 = residual_block("conv2_block2", self.conv2_block1, 64)
        self.conv2_block3 = residual_block("conv2_block3", self.conv2_block2, 64)
        #conv3_x   4 blocks
        self.conv3_block1 = residual_block("conv3_block1", self.conv2_block3, 128)
        self.conv3_block2 = residual_block("conv3_block2", self.conv3_block1, 128)
        self.conv3_block3 = residual_block("conv3_block3", self.conv3_block2, 128)
        self.conv3_block4 = residual_block("conv3_block4", self.conv3_block3, 128)#1/8
        print("conv3_block4: ", np.shape(self.conv3_block4))
        #conv4_x   6 blocks
        self.conv4_block1 = residual_block("conv4_block1", self.conv3_block4, 256)
        self.conv4_block2 = residual_block("conv4_block2", self.conv4_block1, 256)
        self.conv4_block3 = residual_block("conv4_block3", self.conv4_block2, 256)
        self.conv4_block4 = residual_block("conv4_block4", self.conv4_block3, 256)
        self.conv4_block5 = residual_block("conv4_block5", self.conv4_block4, 256)
        self.conv4_block6 = residual_block("conv4_block6", self.conv4_block5, 256)#1/16
        print(np.shape(self.conv4_block6))
        #conv5_x   3 blocks
        self.conv5_block1 = residual_block("conv5_block1", self.conv4_block6, 512)
        self.conv5_block2 = residual_block("conv5_block2", self.conv5_block1, 512)
        self.conv5_block3 = residual_block("conv5_block3", self.conv5_block2, 512)#1/32
        print(np.shape(self.conv5_block3))
        #1/32
        self.score_32 = _conv_layer_without_relu(self.conv5_block3, [1,1,512,class_num], "1x1conv_1", wd=wd)
        #1/16
        self.score_16 = _conv_layer_without_relu(self.conv4_block6, [1,1,256,class_num], "1x1conv_2", wd=wd)
        #1/8
        self.score_8 = _conv_layer_without_relu(self.conv3_block4, [1,1,128,class_num], "1x1conv_3", wd=wd)

        self.upscore1 = _upscore_layer(self.score_32, shape=tf.shape(self.score_16), ksize=4, stride=2, num_class=class_num, name="upscore1")
        self.fuse1 = tf.add(self.score_16, self.upscore1, name="fuse1")
        self.upscore2 = _upscore_layer(self.fuse1, shape=tf.shape(self.score_8), ksize=4, stride=2, num_class=class_num, name="upscore2")
        self.fuse2 = tf.add(self.score_8, self.upscore2, name="fuse2")
        self.upscore = _upscore_layer(self.fuse2, shape=tf.shape(x), ksize=16, stride=8, num_class=class_num, name="final_upscore")

        return self.upscore


