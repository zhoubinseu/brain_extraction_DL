import tensorflow as tf
import numpy as np
from layers.layer_random_init import get_bn_result

#data format = NHWC

def resnet_shortcut(input, residual):
    #make height width channle are equal between input and residual
    input_shape = np.shape(input)
    residual_shape = np.shape(residual)
    euqal_height = input_shape[1]==residual_shape[1]
    equal_width = input_shape[2]==residual_shape[2]
    euqal_channels = input_shape[3]==residual_shape[3]
    if equal_width or euqal_height:
        stride = 1
    else:
        stride = 2
    shortcut = input
    #如果存在维度不一致，对shortcut使用1×1的卷积调整维度
    if not euqal_height or not equal_width or not euqal_channels:
        filter = tf.Variable(initial_value=tf.truncated_normal(shape=[1,1,int(input_shape[3]), int(residual_shape[3])], stddev=5e-2))
        shortcut = tf.nn.conv2d(input=shortcut, filter=filter, strides=[1, stride, stride, 1], padding="VALID", name="shortcut_conv")
    
    return tf.add(residual, shortcut)

#根据Identity Mappings in Deep Residual Networks提出的bn-relu-conv
def bn_relu_conv(input, filter_shape, stride):
    input_channel = np.shape(input)[3]
    bn_layer = get_bn_result(input, input_channel)
    relu_layer = tf.nn.relu(bn_layer)
    filter = tf.get_variable("conv", shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=5e-2))
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding="SAME")
    return conv_layer

def conv_bn_relu(input, filter_shape, stride, name="conv"):
    with tf.variable_scope(name):
        filter = tf.get_variable("conv", shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=5e-2))
        conv_layer = tf.nn.conv2d(input, filter, strides=[1, stride, stride, 1], padding="SAME")
        bn_layer = get_bn_result(conv_layer, np.shape(filter)[3])
        relu_layer = tf.nn.relu(bn_layer)
        return relu_layer

#残差模块层
def residual_block(name, input, output_channel, first_block=False):
    with tf.variable_scope(name):
        input_channel = np.shape(input)[3]
        #输入和输出的维度不同时，残差模块的第一个卷积的stride为2
        if input_channel*2 == output_channel:
            stride = 2
        elif input_channel == output_channel:
            stride = 1
        else:
            raise ValueError("output and input channel does not match in residual block!")
        
        #残差模块的第一个卷积
        with tf.variable_scope("conv1_in_block"):
            #如果当前模块是整个网络的第一个残差模块
            if first_block:
                filter = tf.get_variable("conv", shape=[3, 3, input_channel, output_channel], initializer=tf.truncated_normal_initializer(stddev=5e-2))
                conv1 = tf.nn.conv2d(input, filter, [1, 1, 1, 1], padding="SAME")
            else:
                conv1 = bn_relu_conv(input, [3, 3, input_channel, output_channel], stride)
        
        with tf.variable_scope("conv2_in_block"):
            conv2 = bn_relu_conv(conv1, [3, 3, output_channel, output_channel], 1)
        
        output = resnet_shortcut(input, conv2)
        return output

        


def bottleneck_block(name, input, main_channel, first_block=False, change_stride=False):       
    with tf.variable_scope(name):
        input_channel = np.shape(input)[3]

        if change_stride:
            stride = 2
        else:
            stride = 1

        with tf.variable_scope("conv1_in_bottleneck"):
            if first_block:
                filter = tf.get_variable("conv", shape=[1, 1, input_channel, main_channel], initializer=tf.truncated_normal_initializer(stddev=5e-2))
                conv1 = tf.nn.conv2d(input, filter, [1, 1, 1, 1], padding="SAME")
            else:
                conv1 = bn_relu_conv(input, [1, 1, input_channel, main_channel], stride)
        
        with tf.variable_scope("conv2_in_bottleneck"):
            conv2 = bn_relu_conv(conv1, [3,3,main_channel, main_channel], 1)
        
        with tf.variable_scope("conv3_in_bottleneck"):
            conv3 = bn_relu_conv(conv2, [1,1,main_channel, 4*main_channel], 1)
        
        output = resnet_shortcut(input, conv3)
        return output



def BatchNorm(input):
    mean, var = tf.nn.moments(input, axes=[0, 1, 2])
    output_channel = np.shape(input)[3]
    scale = tf.Variable(tf.ones([output_channel]))
    shift = tf.Variable(tf.zeros([output_channel]))
    epsilon = 0.001
    return tf.nn.batch_normalization(input, mean, var, shift, scale, epsilon)

def Conv_BN_Relu(input, filter_shape, name=None):
    with tf.variable_scope(name):
        filter = tf.get_variable("conv", filter_shape, initializer=tf.truncated_normal_initializer(stddev=5e-2))
        conv = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding="SAME")
        # bias = tf.get_variable("bias", filter_shape[3], initializer=tf.constant_initializer(0.0))
        # conv = tf.nn.bias_add(conv, bias)
        bn = BatchNorm(conv)
        output = tf.nn.relu(bn)
        return output

#transpose convolution
def Upsampling(input, shape, ksize, stride, output_channel, name=None):
    with tf.variable_scope(name):
        input_shape = np.shape(input)
        filter = tf.get_variable("deconv", [ksize, ksize, output_channel, input_shape[3]], initializer=tf.truncated_normal_initializer(stddev=5e-4))
        output_shape = [shape[0], shape[1], shape[2], output_channel]
        output_shape = tf.stack(output_shape)
        filter = tf.cast(filter, tf.float32)
        output = tf.nn.conv2d_transpose(input, filter, output_shape=output_shape, strides=[1, stride, stride, 1], padding="SAME")
        return output

def Conv_Pool_Conv(input, name=None):
    with tf.variable_scope(name):
        D = np.shape(input)[3]
        outputs = Conv_BN_Relu(input, [3, 3, D, D], 'conv1')
        outputs = tf.nn.max_pool(outputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling')
        outputs = Conv_BN_Relu(outputs, [3, 3, D, D//4], 'conv2')
        return outputs

def generation_stage(input, name=None):
    with tf.variable_scope(name):
        D = np.shape(input)[3]
        x = Conv_BN_Relu(input, [1, 1, D, D//2], 'conv1')
        x = Conv_BN_Relu(x, [3, 3, D//2, D//2], 'conv2')
        x = Conv_BN_Relu(x, [1, 1, D//2, D], 'conv3')
        outputs = tf.add(input, x)
        return outputs

def Dense_decoder_feature_generation(input, name=None):
    with tf.variable_scope(name):
        branch1 = Conv_Pool_Conv(input, 'b1')
        branch2 = Conv_Pool_Conv(input, 'b2')
        branch3 = Conv_Pool_Conv(input, 'b3')
        branch4 = Conv_Pool_Conv(input, 'b4')

        features = tf.concat([branch1, branch2, branch3, branch4], axis=3)
        D = np.shape(input)[3]
        features = Conv_BN_Relu(features, [3, 3, D, D], 'conv1')

        for i in range(4):
            features = generation_stage(features, name='bottle'+str(i))
        features = Upsampling(features, tf.shape(input), ksize=4, stride=2, output_channel=D, name="feature")
        features = Conv_BN_Relu(features, [3, 3, D, D], 'conv2')
        return features