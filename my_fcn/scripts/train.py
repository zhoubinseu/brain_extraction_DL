import tensorflow as tf
import numpy as np
from utils.util import shuffle_data
from fcn.fcn8s import fcn8s
from fcn.fcn16s import fcn16s
from fcn.fcn32s import fcn32s
from fcn.fcn8s_inception import fcn8s_inception
from fcn.fcn8s_lstm import fcn8s_lstm
from fcn.resnet_34 import resnet_34
from fcn.res101 import res101
from fcn.fcn8s_ddsc import fcn8s_ddsc
from fcn.fcn8s_inception_ddsc import fcn8s_inception_ddsc
from fcn.segnet import segnet
from fcn.segnet_inception import segnet_inception
from fcn.segnet_ddsc import segnet_ddsc
from fcn.segnet_inception_ddsc import segnet_inception_ddsc
from fcn.segnet_ddsc_v2 import segnet_ddsc_v2
from fcn.segnet_inception_ddsc_v2 import segnet_inception_ddsc_v2
from fcn.region_conv import region_conv
import logging
from tensorflow.python.framework import dtypes
from utils.util import dice_coefficient


def train(model_name, x, y, num_class=21, lr=1e-4, wd=5e-4
          , batch_size=256, epochs=10, display_step=10,
          sparse=False, conti=False, saved_model_name='model'):
    input_shape = np.shape(x)
    channel = np.shape(x)[3]

    data = tf.placeholder(tf.float32, [None, input_shape[1], input_shape[2], channel])
    if sparse:
        label = tf.placeholder(tf.int32, [None, input_shape[1], input_shape[2]])
    else:
        label = tf.placeholder(tf.int32, [None, input_shape[1], input_shape[2], num_class])

    if model_name == 'fcn8s':
        model = fcn8s()
        pred = model.build(data, class_num=num_class, channel=channel, train=True, wd=wd)
    elif model_name == 'fcn16s':
        model = fcn16s()
        pred = model.build(data, class_num=num_class, channel=channel, train=True, wd=wd)
    elif model_name == 'fcn32s':
        model = fcn32s()
        pred = model.build(data, class_num=num_class, channel=channel, train=True, wd=wd)
    elif model_name == 'fcn8s_inception':
        model = fcn8s_inception()
        pred = model.build(data, class_num=num_class, channel=channel, train=True, wd=wd)
    elif model_name == 'fcn8s_lstm':
        model = fcn8s_lstm()
        pred = model.build(data, class_num=num_class, channel=channel, train=True, wd=wd)
    elif model_name == 'resnet_34':
        model = resnet_34()
        pred = model.build(data, class_num=num_class, channel=channel, train=True, wd=wd)
    elif model_name == 'res101':
        model = res101()
        pred = model.build(data, class_num=num_class, channel=channel, train=True, wd=wd)
    elif model_name == 'segnet':
        model = segnet()
        pred = model.build(data, input_channel=channel, class_num=num_class)
    elif model_name == 'segnet_inception':
        model = segnet_inception()
        pred = model.build(data, input_channel=channel, class_num=num_class)
    elif model_name == 'segnet_ddsc':
        model = segnet_ddsc()
        pred = model.build(data, input_channel=channel, class_num=num_class)
    elif model_name == 'segnet_inception_ddsc':
        model = segnet_inception_ddsc()
        pred = model.build(data, input_channel=channel, class_num=num_class)
    elif model_name == 'segnet_ddsc_v2':
        model = segnet_ddsc_v2()
        pred = model.build(data, input_channel=channel, class_num=num_class)
    elif model_name == 'segnet_inception_ddsc_v2':
        model = segnet_inception_ddsc_v2()
        pred = model.build(data, input_channel=channel, class_num=num_class)
    elif model_name == 'fcn8s_ddsc':
        model = fcn8s_ddsc()
        pred = model.build(data, class_num=num_class, channel=channel, train=True, wd=wd)
    elif model_name == 'fcn8s_inception_ddsc':
        model = fcn8s_inception_ddsc()
        pred = model.build(data, class_num=num_class, channel=channel, train=True, wd=wd)
    elif model_name == 'region_conv':
        model = region_conv()
        pred = model.build(data, input_channel=channel, class_num=num_class)
    else:
        logging.info('Model name is wrong !')

    # 定义损失函数和优化器
    # cost = tf.reduce_mean(dice_coef_loss2(label, pred))
    if sparse:
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label))
        correct_pred = tf.equal(tf.argmax(pred, axis=-1, output_type=dtypes.int32), label)
    else:
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label))
        correct_pred = tf.equal(tf.argmax(pred, axis=-1), tf.argmax(label, axis=-1))
    tf.summary.scalar('loss', cost)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    dice = dice_coefficient(label, tf.argmax(pred, axis=-1, output_type=dtypes.int32))

    tf.summary.scalar('acc', accuracy)

    with tf.Session() as sess:
        # init all variables
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        if conti:
            ckpt = tf.train.get_checkpoint_state('./model')
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)

        # write log
        merged_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./log', sess.graph)

        sess.run(init)
        step = 1
        epoch = 1
        while epoch <= epochs:
            x, y = shuffle_data(x, y)
            i = 0
            while (i * batch_size + batch_size) <= input_shape[0]:
                batch_x = np.array(x[i * batch_size: i * batch_size + batch_size])
                batch_y = np.array(y[i * batch_size: i * batch_size + batch_size])
                sess.run(optimizer, feed_dict={data: batch_x, label: batch_y})
                if step % display_step == 0:
                    # 计算损失值和准确度,输出
                    summary_str, batch_pred, loss, acc, dice_coef = sess.run([merged_op, pred, cost, accuracy, dice],
                                                                  feed_dict={data: batch_x, label: batch_y})

                    result = 'Epoch: ' + str(epoch) + ", " + \
                             'Step: ' + str(step) + ", " +\
                             "Iter: " + str(step * batch_size) + ", " +\
                             "loss: " + "{:.6f}".format(loss) + ", " +\
                             "Training Accuracy: " + "{:.5f}".format(acc)+", " + \
                             "Training Dice: " + "{:.5f}".format(dice_coef)
                    print(result)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                step = step + 1
                i = i + 1
            epoch = epoch + 1

        summary_writer.close()

        # 模型存储
        saver.save(sess, 'E:/Zhou/brain extraction/my_fcn/saved_model/%s/model_ckpt_' % saved_model_name, global_step=step - 1)
        print('Optimization Finished!')
