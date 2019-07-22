import tensorflow as tf
import numpy as np
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
import logging
from tensorflow.python.framework import dtypes
from utils.util import dice_coefficient, specificity, sensitivity


def predict(model_name, x, y, num_class=21, batch_size=16, wd=5e-4,
            sparse=False, saved_model_name='model'):
    input_shape = np.shape(x)
    channel = np.shape(x)[3]
    data = tf.placeholder(tf.float32, [None, input_shape[1], input_shape[2], channel])
    if sparse:
        label = tf.placeholder(tf.int32, [None, input_shape[1], input_shape[2]])
    else:
        label = tf.placeholder(tf.float32, [None, input_shape[1], input_shape[2], num_class])

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
    else:
        logging.info('Model name is wrong !')

    if sparse:
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label))
        correct_pred = tf.equal(tf.argmax(pred, axis=-1, output_type=dtypes.int32), label)
    else:
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label))
        correct_pred = tf.equal(tf.argmax(pred, axis=-1), tf.argmax(label, axis=-1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    dice = dice_coefficient(label, tf.argmax(pred, axis=-1, output_type=dtypes.int32))
    spec = specificity(label, tf.argmax(pred, axis=-1, output_type=dtypes.int32))
    sens = sensitivity(label, tf.argmax(pred, axis=-1, output_type=dtypes.int32))
    prob = tf.nn.softmax(logits=pred, dim=-1)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state('E:/Zhou/brain extraction/my_fcn/saved_model/%s' % saved_model_name)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)
        accs = 0
        losses = 0
        dices = 0
        total_spec = 0
        total_sens = 0
        batch_count = 0
        prediction = np.zeros(shape=[input_shape[0], input_shape[1], input_shape[2], num_class])
        probability = np.zeros(shape=[input_shape[0], input_shape[1], input_shape[2], num_class])

        while batch_count * batch_size < input_shape[0]:
            batch_test_data = np.array(
                x[batch_count * batch_size: batch_count * batch_size + batch_size])
            batch_test_label = np.array(
                y[batch_count * batch_size: batch_count * batch_size + batch_size])
            dice_coef, spec_batch, sens_batch, acc, loss, prediction_batch, probability_batch = sess.run([dice, spec, sens, accuracy, cost, pred, prob],
                                                   feed_dict={data: batch_test_data, label: batch_test_label})
            prediction[batch_count * batch_size: batch_count * batch_size + batch_size] = prediction_batch
            probability[batch_count * batch_size: batch_count * batch_size + batch_size] = probability_batch

            print("test sample " + str(batch_count) + ": ", 'Loss: ', loss, 'Accuracy: ', acc, 'Dice: ', dice_coef, 'Specificity: ', spec_batch, 'Sensitivity: ', sens_batch)

            accs += acc
            losses += loss
            dices += dice_coef
            total_spec += spec_batch
            total_sens += sens_batch
            batch_count += 1

            # print(loss)
            # print(acc)
            # print(dice_coef)
            # print(batch_count)

        print('Prediction over !')
        print('Loss: ', losses / batch_count)
        print('Accuracy: ', accs / batch_count)
        print('Dice: ', dices / batch_count)
        print('Specificity: ', total_spec/batch_count)
        print('Sensitivity: ', total_sens/batch_count)

        # prediction of voc
        # prediction = np.argmax(prediction, axis=-1)
        # show_images(prediction[:6], y[:6], 6, 2, 21, figsize=(60, 60))

        # prediction of brain voc
        prediction = np.argmax(prediction, axis=-1)
        return prediction, probability