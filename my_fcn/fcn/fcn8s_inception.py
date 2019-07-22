from layers.layer_random_init import _conv_layer
from layers.layer_random_init import _conv_layer_without_relu
from layers.layer_random_init import _pooling_layer
from layers.layer_random_init import _upscore_layer
from layers.layer_random_init import _inception_layer
import tensorflow as tf

#fcn8s的结构中引入inception
class fcn8s_inception():
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
        self.score_fr = _conv_layer_without_relu(self.conv7, [1, 1, 4096, class_num], 'conv7_1x1conv', wd=wd)

        self.pooling4_conv = _conv_layer_without_relu(self.pooling4, [1, 1, 512, class_num], 'pool4_1x1conv', wd=wd)

        self.pooling3_conv = _conv_layer_without_relu(self.pooling3, [1, 1, 256, class_num], 'pool3_1x1conv', wd=wd)

        self.upscore = _upscore_layer(self.score_fr, shape=tf.shape(self.pooling4_conv), ksize=4, stride=2
                                      , num_class=class_num, name='upscore')

        self.fuse_pool4 = tf.add(self.pooling4_conv, self.upscore, name='fuse_pool4')

        self.fuse_pool4_upscore = _upscore_layer(self.fuse_pool4, shape=tf.shape(self.pooling3_conv),
                                                 ksize=4, stride=2, num_class=class_num, name='fuse_pool4_upscore')

        self.fuse_pool3 = tf.add(self.pooling3_conv, self.fuse_pool4_upscore, name='fuse_pool3')

        self.upscore = _upscore_layer(self.fuse_pool3, shape=tf.shape(x), ksize=16, stride=8,
                                      num_class=class_num, name='final_upscore')

        return self.upscore
