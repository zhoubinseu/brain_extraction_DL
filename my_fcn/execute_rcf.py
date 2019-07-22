import tensorflow as tf
import numpy as np
import rcf

def edge_detect(x, batchsize=16, saved_model_name="model"):
    input_shape = np.shape(x)
    data = tf.placeholder(tf.float32, [None, input_shape[1], input_shape[2], input_shape[3]])
    model = rcf.crf_vgg16()
    score1, deconv2, deconv3, deconv4, deconv5, pred = model.build(data, input_shape[3])
    result = tf.nn.sigmoid(pred)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state('E:/Zhou/brain extraction/my_fcn/edge_detector/rcf_model/%s' % saved_model_name)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)
        batch_count = 0

        prediction = np.zeros(shape=[input_shape[0], input_shape[1], input_shape[2], input_shape[3]])

        while batch_count * batchsize < input_shape[0]:
            batch_x = np.array(x[batch_count * batchsize: batch_count * batchsize + batchsize])
            batch_prediction = sess.run(result, feed_dict={data: batch_x})
            prediction[batch_count * batchsize: batch_count * batchsize + batchsize] = batch_prediction
            
            batch_count += 1

        #sigmoid产生的值在0~1之间，对其进行四舍五入得到离散的二分类
        prediction = np.around(prediction)
        # prediction = tf.cast(prediction, tf.int32)
        print("提取边缘成功")
        return prediction