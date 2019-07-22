import tensorflow as tf
import numpy as np

# 网络各层的实现及初始化

# 卷积层实现
def _conv_layer(input, filter_shape, name, stddev=5e-2, wd=0):
    with tf.variable_scope(name) as scope:
        filter = get_conv_filter(filter_shape, stddev,wd)
        conv = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME', name='conv')
        baises = get_bias(filter_shape[3])
        output = tf.nn.bias_add(conv, baises)
        #output = get_bn_result(output, filter_shape[3])
        relu = tf.nn.relu(output)
        tf.summary.histogram('conv_layer', relu)
        return relu

# branch1: 1x1 conv
# branch2: 1x1 conv  3x3 conv
# branch3: 1x1 conv  3x3 conv  3x3 conv
# branch4: 1x1 conv  3x3 conv  3x3 conv
# relu(concat(branch1,branch2,branch3,branch4))
# output channel = out_1+out_22+out_33+out_43
def _inception_layer(name, input, in_c, out_1, out_21, out_22, out_31, out_32, out_33, out_41, out_42, out_43):
    with tf.variable_scope(name) as scope:
        branch1 = _conv_layer_for_inception(input, [1, 1, in_c, out_1], "branch1")
        branch1 = get_bn_result(branch1, out_1)
        branch2_1 = _conv_layer_for_inception(input, [1, 1, in_c, out_21], "branch2_1")
        branch2_2 = _conv_layer_for_inception(branch2_1, [3, 3, out_21, out_22], "branch2_2")
        branch2_2 = get_bn_result(branch2_2, out_22)
        branch3_1 = _conv_layer_for_inception(input, [1, 1, in_c, out_31], "branch3_1")
        branch3_2 = _conv_layer_for_inception(branch3_1, [3, 3, out_31, out_32], "branch3_2")
        branch3_3 = _conv_layer_for_inception(branch3_2, [3, 3, out_32, out_33], "branch3_3")
        branch3_3 = get_bn_result(branch3_3, out_33)
        branch4_1 = _conv_layer_for_inception(input, [1, 1, in_c, out_41], "branch4_1")
        branch4_2 = _conv_layer_for_inception(branch4_1, [3, 3, out_41, out_42], "branch4_2")
        branch4_3 = _conv_layer_for_inception(branch4_2, [3, 3, out_42, out_43], "branch4_3")
        branch4_3 = get_bn_result(branch4_3, out_43)
        concat_feature = tf.concat([branch1, branch2_2, branch3_3, branch4_3], axis=3)  # input 4个维度[batchsize,width,height,channel]
        output = tf.nn.relu(concat_feature)
        return output

def _conv_layer_for_inception(input, filter_shape, name, stddev=5e-2, wd=0):
    with tf.variable_scope(name) as scope:
        filter = get_conv_filter(filter_shape, stddev, wd)
        conv = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME', name='conv')
        baises = get_bias([filter_shape[3]])
        output = tf.nn.bias_add(conv, baises)
        # output = get_bn_result(output, filter_shape[3])
        tf.summary.histogram('conv_layer_without_relu', output)
        return output

def _conv_layer_without_relu(input, filter_shape, name, stddev=5e-2, wd=0):
    with tf.variable_scope(name, reuse=None) as scope:
        filter = get_conv_filter(filter_shape, stddev,wd)
        conv = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME', name='conv')
        baises = get_bias(filter_shape[3])
        output = tf.nn.bias_add(conv, baises)
        #output = get_bn_result(output, filter_shape[3])
        tf.summary.histogram('conv_layer_without_relu', output)
        return output

# 最大池化层实现
def _pooling_layer(input, name):
    with tf.variable_scope(name) as scope:
        pool = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name='pooling')
        tf.summary.histogram('pooling_layer', pool)
        return pool


# 上采样(Upsampling)层实现
def _upscore_layer(input, shape, ksize, stride, name, num_class, stddev=5e-4, wd=0):
    with tf.variable_scope(name, reuse=True) as scope:
        # 最后的上采样最后一维为类别数 X 上采样的最后一维都为类别数，做融合之前的那次也是一样
        in_channels = input.get_shape()[3].value
        out_channels = num_class

        f_shape = [ksize, ksize, out_channels, in_channels]

        filter = get_conv_filter(f_shape,stddev,wd)
        #filter = get_deconv_filter(f_shape, wd)

        output_shape = [shape[0], shape[1], shape[2], out_channels]
        output_shape = tf.stack(output_shape)

        filter = tf.cast(filter, tf.float32)

        upscore = tf.nn.conv2d_transpose(input, filter, output_shape=output_shape,
                                         strides=[1, stride, stride, 1],
                                         padding='SAME', name='upscore')
        #tf.summary.histogram('upscore_layer', upscore)
        return upscore


# 获取卷积参数
def get_conv_filter(shape, stddev, wd):
    # init = tf.truncated_normal_initializer(stddev=stddev)
    # weights = tf.get_variable(name='conv_weights',shape=shape,
    #                           initializer=init)
    weights = tf.Variable(initial_value=tf.truncated_normal(shape=shape,stddev=stddev, name='conv_weights'))
    #adding weight decay
    if not tf.get_variable_scope().reuse:
         weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
         tf.add_to_collection('losses', weight_decay)
    #tf.summary.histogram('conv_weights',weights)
    return weights


#加入 batch_normalization
def get_bn_result(input_layer, output_channel):
    mean, var = tf.nn.moments(input_layer, axes=[0, 1, 2])
    scale = tf.Variable(tf.ones([output_channel]))
    shift = tf.Variable(tf.zeros([output_channel]))
    epsilon = 0.001
    return tf.nn.batch_normalization(input_layer, mean, var, shift, scale, epsilon)

# 获取偏置
def get_bias(shape):
    initializer = tf.constant_initializer(0.0)
    biases = tf.get_variable(name='biases', shape=shape,
                            initializer=initializer)
    return biases

def get_bias_v2(shape, name):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name=name)

# 获取deconv_filter的权重
def get_deconv_filter(input_shape,wd):
    # width = input_shape[0]
    # height = input_shape[1]
    #
    # f = ceil(width / 2.0)
    # c = (2 * f - 1 - f % 2) / (2.0 * f)
    # bilinear = np.zeros([input_shape[0], input_shape[1]])
    # for x in range(width):
    #     for y in range(height):
    #         value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
    #         bilinear[x, y] = value
    #
    # weights = np.zeros(input_shape)
    size = input_shape[0]
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    weights = np.zeros(shape=input_shape)
    bilinear = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

    for i in range(input_shape[2]):
        for j in range(input_shape[3]):
            weights[:, :, i, j] = bilinear

    # init = tf.constant_initializer(value=weights,dtype=tf.float32)
    # weights = tf.get_variable(name='deconv_filter', initializer=init, shape=weights.shape)
    weights = tf.Variable(initial_value=weights,name='deconv_filter',trainable=False)

    if not tf.get_variable_scope().reuse:
         weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
         tf.add_to_collection('losses', weight_decay)

    #tf.summary.histogram('deconv_weights', weights)

    # 参数需要返回的是tf.variable
    return weights


