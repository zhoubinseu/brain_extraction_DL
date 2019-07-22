import tensorflow as tf
import numpy as np
import rcf
import rcf_utils

def train(x, y, lr=1e-4, batchsize=32, epochs=10, display_step=10, saved_model_name="model"):
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
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_sum)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(init)
        step = 1
        epoch = 1
        while epoch <= epochs:
            x, y = rcf_utils.shuffle_data(x, y)
            i = 0
            while (i*batchsize+batchsize) <= input_shape[0]:
                batch_x = np.array(x[i * batchsize: i * batchsize + batchsize])
                batch_y = np.array(y[i * batchsize: i * batchsize + batchsize])
                sess.run(optimizer, feed_dict={data: batch_x, label: batch_y})
                if step % display_step == 0:
                    batch_pred, loss, acc = sess.run([pred, cost_sum, accuracy], feed_dict={data: batch_x, label: batch_y})

                    result = 'Epoch: ' + str(epoch) + ", " + \
                             'Step: ' + str(step) + ", " +\
                             "Iter: " + str(step * batchsize) + ", " +\
                             "loss: " + "{:.6f}".format(loss) + ", " +\
                             "Training Accuracy: " + "{:.5f}".format(acc)
                    print(result)
            
                step = step + 1
                i = i + 1
            epoch = epoch + 1

        saver.save(sess, "E:/Zhou/brain extraction/my_fcn/edge_detector/rcf_model/%s/model_ckpt_" % saved_model_name, global_step=step - 1)
        print('Optimization Finished!')

