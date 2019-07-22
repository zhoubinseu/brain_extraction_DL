import tensorflow as tf
import numpy as np
import rcf
import rcf_utils

def test(x, y, batchsize=16, saved_model_name="model"):
    input_shape = np.shape(x)

    input_shape = np.shape(x)
    data = tf.placeholder(tf.float32, [None, input_shape[1], input_shape[2], input_shape[3]])
    label =  tf.placeholder(tf.float32, [None, input_shape[1], input_shape[2], 1])
    model = rcf.crf_vgg16()
    score1, deconv2, deconv3, deconv4, deconv5, pred = model.build(data, input_shape[3])

    cost_stage1 = rcf_utils.class_balanced_loss_function("cost_stage1", score1, label)
    cost_stage2 = rcf_utils.class_balanced_loss_function("cost_stage2", deconv2, label)
    cost_stage3 = rcf_utils.class_balanced_loss_function("cost_stage3", deconv3, label)
    cost_stage4 = rcf_utils.class_balanced_loss_function("cost_stage4", deconv4, label)
    cost_stage5 = rcf_utils.class_balanced_loss_function("cost_stage5", deconv5, label)
    cost_fusion = rcf_utils.class_balanced_loss_function("cost_fusion", pred, label)
    cost_sum = cost_stage1+cost_stage2+cost_stage3+cost_stage4+cost_stage5+cost_fusion

    correct_pred = tf.equal(pred, label)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    result = tf.nn.sigmoid(pred)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state('E:/Zhou/brain extraction/my_fcn/edge_detector/rcf_model/%s' % saved_model_name)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)

        losses = 0
        accs = 0
        batch_count = 0

        prediction = np.zeros(shape=[input_shape[0], input_shape[1], input_shape[2], input_shape[3]])

        while batch_count * batchsize < input_shape[0]:
            batch_x = np.array(x[batch_count * batchsize: batch_count * batchsize + batchsize])
            batch_y = np.array(y[batch_count * batchsize: batch_count * batchsize + batchsize])
            loss, acc, batch_prediction = sess.run([cost_sum, accuracy, result], feed_dict={data: batch_x, label: batch_y})
            prediction[batch_count * batchsize: batch_count * batchsize + batchsize] = batch_prediction
            
            losses += loss
            accs += acc
            batch_count += 1
        
        print("Test over!")
        print("loss: ", losses/batch_count)
        print("accuracy: ", accs/batch_count)

        #sigmoid产生的值在0~1之间，对其进行四舍五入得到离散的二分类
        prediction = np.around(prediction)
        # prediction = tf.cast(prediction, tf.int32)
        return prediction

