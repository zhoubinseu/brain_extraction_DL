import tensorflow as tf
import numpy as np
from layers.layers import bottleneck_block
from layers.layer_random_init import get_bn_result
from layers.layer_random_init import _upscore_layer
from layers.layer_random_init import _conv_layer_without_relu
from layers.layer_random_init import _conv_layer
from layers.layer_random_init import _pooling_layer

class res101():
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
        #conv2_x  3 blocks
        self.conv2_block1 = bottleneck_block("conv2_block1", self.pooling2, 64, first_block=True, change_stride=False)
        self.conv2_block2 = bottleneck_block("conv2_block2", self.conv2_block1, 64)
        self.conv2_block3 = bottleneck_block("conv2_block3", self.conv2_block2, 64)
        #conv3_x  4 blocks
        self.conv3_block1 = bottleneck_block("conv3_block1", self.conv2_block3, 128, change_stride=True)
        self.conv3_block2 = bottleneck_block("conv3_block2", self.conv3_block1, 128)
        self.conv3_block3 = bottleneck_block("conv3_block3", self.conv3_block2, 128)
        self.conv3_block4 = bottleneck_block("conv3_block4", self.conv3_block3, 128)
        print("conv3_block4: ", np.shape(self.conv3_block4))
        #conv4_x  23 blocks
        self.conv4_block1 = bottleneck_block("conv4_block1", self.conv3_block4, 256, change_stride=True)
        self.conv4_block2 = bottleneck_block("conv4_block2", self.conv4_block1, 256)
        self.conv4_block3 = bottleneck_block("conv4_block3", self.conv4_block2, 256)
        self.conv4_block4 = bottleneck_block("conv4_block4", self.conv4_block3, 256)
        self.conv4_block5 = bottleneck_block("conv4_block5", self.conv4_block4, 256)
        self.conv4_block6 = bottleneck_block("conv4_block6", self.conv4_block5, 256)
        self.conv4_block7 = bottleneck_block("conv4_block7", self.conv4_block6, 256)
        self.conv4_block8 = bottleneck_block("conv4_block8", self.conv4_block7, 256)
        self.conv4_block9 = bottleneck_block("conv4_block9", self.conv4_block8, 256)
        self.conv4_block10 = bottleneck_block("conv4_block10", self.conv4_block9, 256)
        self.conv4_block11 = bottleneck_block("conv4_block11", self.conv4_block10, 256)
        self.conv4_block12 = bottleneck_block("conv4_block12", self.conv4_block11, 256)
        self.conv4_block13 = bottleneck_block("conv4_block13", self.conv4_block12, 256)
        self.conv4_block14 = bottleneck_block("conv4_block14", self.conv4_block13, 256)
        self.conv4_block15 = bottleneck_block("conv4_block15", self.conv4_block14, 256)
        self.conv4_block16 = bottleneck_block("conv4_block16", self.conv4_block15, 256)
        self.conv4_block17 = bottleneck_block("conv4_block17", self.conv4_block16, 256)
        self.conv4_block18 = bottleneck_block("conv4_block18", self.conv4_block17, 256)
        self.conv4_block19 = bottleneck_block("conv4_block19", self.conv4_block18, 256)
        self.conv4_block20 = bottleneck_block("conv4_block20", self.conv4_block19, 256)
        self.conv4_block21 = bottleneck_block("conv4_block21", self.conv4_block20, 256)
        self.conv4_block22 = bottleneck_block("conv4_block22", self.conv4_block21, 256)
        self.conv4_block23 = bottleneck_block("conv4_block23", self.conv4_block22, 256)
        print(np.shape(self.conv4_block23))
        #conv5_x  3 blocks
        self.conv5_block1 = bottleneck_block("conv5_block1", self.conv4_block23, 512, change_stride=True)
        self.conv5_block2 = bottleneck_block("conv5_block2", self.conv5_block1, 512)
        self.conv5_block3 = bottleneck_block("conv5_block3", self.conv5_block2, 512)
        print(np.shape(self.conv5_block3))

        #1/32
        self.score_32 = _conv_layer_without_relu(self.conv5_block3, [1,1,4*512,class_num], "1x1conv_1", wd=wd)
        #1/16
        self.score_16 = _conv_layer_without_relu(self.conv4_block23, [1,1,4*256,class_num], "1x1conv_2", wd=wd)
        #1/8
        self.score_8 = _conv_layer_without_relu(self.conv3_block4, [1,1,4*128,class_num], "1x1conv_3", wd=wd)

        self.upscore1 = _upscore_layer(self.score_32, shape=tf.shape(self.score_16), ksize=4, stride=2, num_class=class_num, name="upscore1")
        self.fuse1 = tf.add(self.score_16, self.upscore1, name="fuse1")
        self.upscore2 = _upscore_layer(self.fuse1, shape=tf.shape(self.score_8), ksize=4, stride=2, num_class=class_num, name="upscore2")
        self.fuse2 = tf.add(self.score_8, self.upscore2, name="fuse2")
        self.upscore = _upscore_layer(self.fuse2, shape=tf.shape(x), ksize=16, stride=8, num_class=class_num, name="final_upscore")

        return self.upscore