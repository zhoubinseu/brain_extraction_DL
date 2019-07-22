import tensorflow as tf
import numpy as np
from layers.layers import Conv_BN_Relu, Upsampling
from layers.layer_random_init import _upscore_layer

class region_conv():
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
        # region conv stage1   output: 1/4
        # 先进行两次卷积然后softmax
        self.s1_conv1 = Conv_BN_Relu(self.pooling2, [3, 3, 128, 64], "s1_conv1")
        self.s1_conv2 = Conv_BN_Relu(self.s1_conv1, [3, 3, 64, class_num], "s1_conv2")#need return
        self.softmax1 = tf.nn.softmax(self.s1_conv2)
        print(self.softmax1.shape)
        self.max1 = tf.reduce_max(self.softmax1, -1)#沿着最后一个维度获取每个位置最大值
        print(self.max1.shape)
        self.mask1 = np.where(self.max1>0.95, np.zeros((np.shape(self.max1)[0], np.shape(self.max1)[1], np.shape(self.max1)[2]), dtype=int), np.ones((np.shape(self.max1)[0], np.shape(self.max1)[1], np.shape(self.max1)[2]), dtype=int))#概率大于95%的像素设为0 否则为1
        # self.mask1 = tf.where(self.max1-0.95>0, np.zeros((np.shape(self.max1)[0], np.shape(self.max1)[1], np.shape(self.max1)[2])), np.ones((np.shape(self.max1)[0], np.shape(self.max1)[1], np.shape(self.max1)[2])))#概率大于95%的像素设为0 否则为1
        self.expand_mask1 = np.expand_dims(self.mask1, -1)#将mask扩展一个维度
        self.repeat_mask1 = np.repeat(self.expand_mask1, np.shape(self.pooling2)[-1], -1)#沿着最后一个维度复制，使得与要处理的feature的channel相同
        self.masked_feature1 = self.pooling2*self.repeat_mask1
        #stage1 的分割结果：只取mask1=0的区域
        self.invert_mask1 = 1-self.repeat_mask1
        self.pred_stage1 = self.s1_conv2*self.invert_mask1

        # encoder 3
        self.conv3_1 = Conv_BN_Relu(self.masked_feature1, [3, 3, 128, 256], "conv3_1")
        self.conv3_2 = Conv_BN_Relu(self.conv3_1, [3, 3, 256, 256], "conv3_2")
        self.conv3_3 = Conv_BN_Relu(self.conv3_2, [3, 3, 256, 256], "conv3_3")
        self.pooling3 = tf.nn.max_pool(self.conv3_3, self.pool_size, self.pool_stride, padding="SAME", name="pooling3")
        # encoder 4
        self.conv4_1 = Conv_BN_Relu(self.pooling3, [3, 3, 256, 512], "conv4_1")
        self.conv4_2 = Conv_BN_Relu(self.conv4_1, [3, 3, 512, 512], "conv4_2")
        self.conv4_3 = Conv_BN_Relu(self.conv4_2, [3, 3, 512, 512], "conv4_3")
        self.pooling4 = tf.nn.max_pool(self.conv4_3, self.pool_size, self.pool_stride, padding="SAME", name="pooling4")
        # region conv stage2  output: 1/16
        self.s2_conv1 = Conv_BN_Relu(self.pooling4, [3, 3, 512, 64], "s2_conv1")
        self.s2_conv2 = Conv_BN_Relu(self.s2_conv1, [3, 3, 64, class_num], "s2_conv2")
        self.softmax2 = tf.nn.softmax(self.s2_conv2)
        self.max2 = np.max(self.softmax2, -1)
        self.pooled_mask1 = tf.nn.max_pool(self.expand_mask1, self.pool_size, self.pool_stride, padding="SAME", name="mask1_pool1")#将mask调整至相同尺寸
        self.pooled_mask1 = tf.nn.max_pool(self.pooled_mask1, self.pool_size, self.pool_stride, padding="SAME", name="mask1_pool2")
        self.pooled_mask1 = np.repeat(self.pooled_mask1, class_num, -1)
        self.max2_after_mask1 = self.max2*self.pooled_mask1  #mask2=0的部分包含mask1=0的区域，先将mask1=0的区域设置为0
        self.mask2 = np.where(self.max2_after_mask1>0.95, 0, 1)
        self.expand_mask2 = np.expand_dims(self.mask2, -1)
        self.repeat_mask2 = np.repeat(self.expand_mask2, np.shape(self.pooling4)[-1], -1)
        self.masked_feature2 = self.pooling4*self.repeat_mask2
        #stage2 的分割结果：取mask1=1并且mask2=0的区域
        self.pred_stage2 = self.s2_conv2*self.pooled_mask1
        self.invert_mask2 = 1-np.repeat(self.expand_mask2, class_num, -1)
        self.pred_stage2 = self.pred_stage2*self.invert_mask2


        # encoder 5
        self.conv5_1 = Conv_BN_Relu(self.masked_feature2, [3, 3, 512, 512], "conv5_1")
        self.conv5_2 = Conv_BN_Relu(self.conv5_1, [3, 3, 512, 512], "conv5_2")
        self.conv5_3 = Conv_BN_Relu(self.conv5_2, [3, 3, 512, 512], "conv5_3")
        self.pooling5 = tf.nn.max_pool(self.conv5_3, self.pool_size, self.pool_stride, padding="SAME", name="pooling5")
        # region conv stage3  output: 1/32
        self.s3_conv1 = Conv_BN_Relu(self.pooling5, [3, 3, 512, 64], "s3_conv1")
        self.s3_conv2 = Conv_BN_Relu(self.s3_conv1, [3, 3, 64, class_num], "s3_conv2")
        #stage3 的分割结果，取mask1=1并且mask2=1的区域(等同于mask2=1的区域)
        self.pooled_mask2 = tf.nn.max_pool(self.expand_mask2, self.pool_size, self.pool_stride, padding="SAME", name="mask2_pool1")
        self.pred_stage3 = self.s3_conv2*np.repeat(self.pooled_mask2, class_num, -1)

        #[1]stage1(1/4) stage2(1/16) stage3(1/32)的分割结果先统一size，合并，再进行上采样  or  [2]采用FCN的方式一边上采样一边合并各阶段的结果？
        #先将stage3进行2x上采样，与stage2合并，再进行4x上采样，与stage1合并，最后进行4x上采样获得输出size的结果
        self.up_stage3 = _upscore_layer(self.pred_stage3, shape=tf.shape(self.pred_stage2), ksize=4, stride=2, name="up_stage3", num_class=class_num)
        self.fuse_s2_s3 = tf.add(self.up_stage3, self.pred_stage2)
        self.up_stage2 = _upscore_layer(self.fuse_s2_s3, shape=tf.shape(self.pred_stage1), ksize=8, stride=4, name="up_stage2", num_class=class_num)
        self.fuse_s1_s2 = tf.add(self.up_stage2, self.pred_stage1)
        self.fuse_pred = _upscore_layer(self.fuse_s1_s2, shape=tf.shape(input), ksize=8, stride=4, name="fuse_pred", num_class=class_num)

        return self.fuse_pred