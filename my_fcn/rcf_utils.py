import tensorflow as tf
import numpy as np
from os.path import join
import nibabel
import cv2

def shuffle_data(imgs, labels):
    imgs = np.asarray(imgs)
    labels = np.asarray(labels)
    r = np.random.permutation(len(imgs))
    return imgs[r], labels[r]

def get_bn_result(input, output_channel):
    mean, var = tf.nn.moments(input, axes=[0, 1, 2])
    scale = tf.Variable(tf.ones([output_channel]))
    shift = tf.Variable(tf.zeros([output_channel]))
    epsilon = 0.001
    return tf.nn.batch_normalization(input, mean, var, shift, scale, epsilon)

def _conv_layers(name, input, filter_shape, stride=1, stddev=5e-2, bias=True, bn=False, relu=True):
    with tf.variable_scope(name):
        filter = tf.get_variable(name="conv_weight", shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, filter, [1, stride, stride, 1], padding="SAME", name="conv")
        if bias:
            b = tf.get_variable(name="bias", shape=[filter_shape[3]], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, b)
        if bn:
            conv = get_bn_result(conv, filter_shape[3])
        if relu:
            conv = tf.nn.relu(conv)
        return conv

def _max_pooling(name, input, ksize=2, stride=2):
    with tf.variable_scope(name):
        pool = tf.nn.max_pool(input, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding="SAME", name="pooling")
        return pool

def _up_sample(name, input, shape, ksize, stride, output_channel, stddev=5e-4):
    with tf.variable_scope(name):
        in_channel = np.shape(input)[3]
        f_shape = [ksize, ksize, output_channel, in_channel]
        filter = tf.get_variable(name="conv_weight", shape=f_shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
        output_shape = [shape[0], shape[1], shape[2], output_channel]
        output_shape = tf.stack(output_shape)
        filter = tf.cast(filter, tf.float32)

        upscore = tf.nn.conv2d_transpose(input, filter, output_shape=output_shape,
                                         strides=[1, stride, stride, 1],
                                         padding='SAME', name='upscore')

        return upscore
    

def class_balanced_loss_function(name, pred, label):
    with tf.variable_scope(name):
        pred = tf.cast(pred, tf.float32)
        label = tf.cast(label, tf.float32)
        num_pos = tf.reduce_sum(label)
        num_neg = tf.reduce_sum(1.0-label)

        beta = num_neg/(num_neg+num_pos)
        pos_weight = beta/(1-beta)
        cost = tf.nn.weighted_cross_entropy_with_logits(targets=label, logits=pred, pos_weight=pos_weight)

        loss = tf.reduce_mean((1-beta)*cost)
        return tf.where(tf.equal(beta, 1.0), 0.0, loss)

LPBA40_DIR = "F:/LPBA40/native_space/"
IBSR_DIR = "F:/IBSR/IBSR_nifti_stripped/"
OASIS_DISC1_DIR = "F:/oasis/oasis/disc1/"  #39
OASIS_DISC2_DIR = "F:/oasis/oasis/disc2/"  #38
CANNY_MINVAL = 100
CANNY_MAXVAL = 200

def normalization(data):
    max = np.max(data)
    min = np.min(data)
    data = (data - min) / (max - min)
    return data

# train_images 为MRI图片， train_labels为对应label的边缘， 使用canny提取
def load_lpba40_withedge_train():
    images = []
    labels = []

    for idx in range(1, 22):
        if idx < 9:
            imagename = LPBA40_DIR+"S0" + str(idx) + "/S0" + str(idx) + ".native.mri.hdr"
            labelname = LPBA40_DIR+"S0" + str(idx) + "/S0" + str(idx) + ".native.brain.mask.hdr"
        elif idx == 9 or idx == 10:
            continue
        elif idx > 10:
            imagename = LPBA40_DIR+"S" + str(idx) + "/S" + str(idx) + ".native.mri.hdr"
            labelname = LPBA40_DIR+"S" + str(idx) + "/S" + str(idx) + ".native.brain.mask.hdr"
        img = nibabel.load(imagename)
        label = nibabel.load(labelname)
        img_data = img.get_data()
        label_data = label.get_data()
        images.append(img_data[:, :, :, 0])
        labels.append(label_data[:, :, :, 0])

    train_images = np.asarray(images, dtype=np.int32)
    train_labels = np.asarray(labels, dtype=np.int32)

    train_images = np.reshape(train_images, [train_images.shape[0]*train_images.shape[1], train_images.shape[2], train_images.shape[3]])
    train_labels = np.reshape(train_labels, [train_labels.shape[0]*train_labels.shape[1], train_labels.shape[2], train_labels.shape[3]])
    train_images = np.expand_dims(train_images, axis=-1)
    sequence_len = train_images.shape[0]
    edges = []
    for i in range(0, sequence_len):
        train_labels_slice = train_labels[i]
        slice_edge = cv2.Canny(np.uint8(train_labels_slice), CANNY_MINVAL, CANNY_MAXVAL)
        edges.append(slice_edge)

    edge_labels = np.asarray(edges, dtype=np.int32)
    edge_labels = normalization(edge_labels)
    edge_labels = np.expand_dims(edge_labels, axis=-1)

    return train_images, edge_labels

def load_lpba40_withedge_test():
    images = []
    labels = []

    for idx in range(22, 41):
        if idx < 9:
            imagename = LPBA40_DIR+"S0" + str(idx) + "/S0" + str(idx) + ".native.mri.hdr"
            labelname = LPBA40_DIR+"S0" + str(idx) + "/S0" + str(idx) + ".native.brain.mask.hdr"
        elif idx == 9 or idx == 10:
            continue
        elif idx > 10:
            imagename = LPBA40_DIR+"S" + str(idx) + "/S" + str(idx) + ".native.mri.hdr"
            labelname = LPBA40_DIR+"S" + str(idx) + "/S" + str(idx) + ".native.brain.mask.hdr"
        img = nibabel.load(imagename)
        # print(img.shape)
        label = nibabel.load(labelname)
        label_affine = label.affine
        img_data = img.get_data()
        label_data = label.get_data()
        images.append(img_data[:, :, :, 0])
        labels.append(label_data[:, :, :, 0])

    test_images = np.asarray(images, dtype=np.int32)
    test_labels = np.asarray(labels, dtype=np.int32)

    test_images = np.reshape(test_images, [test_images.shape[0] * test_images.shape[1], test_images.shape[2],
                                            test_images.shape[3]])
    test_labels = np.reshape(test_labels, [test_labels.shape[0] * test_labels.shape[1], test_labels.shape[2],
                                            test_labels.shape[3]])

    test_images = np.expand_dims(test_images, axis=-1)

    sequence_len = test_images.shape[0]
    edges = []
    for i in range(0, sequence_len):
        test_labels_slice = test_labels[i]
        slice_edge = cv2.Canny(np.uint8(test_labels_slice), CANNY_MINVAL, CANNY_MAXVAL)
        edges.append(slice_edge)
    
    test_edge_labels = np.asarray(edges, dtype=np.int32)
    test_edge_labels = normalization(test_edge_labels)
    test_edge_labels = np.expand_dims(test_edge_labels, axis=-1)

    return test_images, test_edge_labels


def load_ibsr18_withedge_train():
    images = []
    labels = []

    for idx in range(1, 10):
        if idx < 10:
            imagename = IBSR_DIR+"IBSR_0" + str(idx) + "/IBSR_0" + str(idx) + "_ana.nii.gz"
            labelname = IBSR_DIR+"IBSR_0" + str(idx) + "/IBSR_0" + str(idx) + "_ana_brainmask.nii.gz"
        elif idx >= 10:
            imagename = IBSR_DIR+"IBSR_" + str(idx) + "/IBSR_" + str(idx) + "_ana.nii.gz"
            labelname = IBSR_DIR+"IBSR_" + str(idx) + "/IBSR_" + str(idx) + "_ana_brainmask.nii.gz"
        img = nibabel.load(imagename)
        label = nibabel.load(labelname)
        img_data = img.get_data()
        label_data = label.get_data()
        # 数据集中brain为256*256*128*1 label为256*128*256*1，维度不一致
        label_data = label_data.transpose(0,2,1,3)
        img_data = img_data[:, :, :, 0]
        label_data = label_data[:, :, :, 0]
        images.append(img_data)
        labels.append(label_data)

    train_images = np.asarray(images, dtype=np.int32)
    train_labels = np.asarray(labels, dtype=np.int32)


    train_images = np.reshape(train_images, [train_images.shape[0]*train_images.shape[1], train_images.shape[2], train_images.shape[3]])
    train_labels = np.reshape(train_labels, [train_labels.shape[0]*train_labels.shape[1], train_labels.shape[2], train_labels.shape[3]])
    train_images = np.expand_dims(train_images, axis=-1)
    #对每个index的label提取edge，作为该index位置的image的第二个chennel
    sequence_len = train_images.shape[0]
    edges = []
    for i in range(0, sequence_len):
        train_labels_slice = train_labels[i]
        slice_edge = cv2.Canny(np.uint8(train_labels_slice), 1, 2)
        edges.append(slice_edge)

    edge_labels = np.asarray(edges, dtype=np.int32)
    edge_labels = normalization(edge_labels)
    edge_labels = np.expand_dims(edge_labels, axis=-1)

    return train_images, edge_labels

def load_ibsr18_withedge_test():
    images = []
    labels = []

    for idx in range(10, 19):
        if idx < 10:
            imagename = IBSR_DIR+"IBSR_0" + str(idx) + "/IBSR_0" + str(idx) + "_ana.nii.gz"
            labelname = IBSR_DIR+"IBSR_0" + str(idx) + "/IBSR_0" + str(idx) + "_ana_brainmask.nii.gz"
        elif idx >= 10:
            imagename = IBSR_DIR+"IBSR_" + str(idx) + "/IBSR_" + str(idx) + "_ana.nii.gz"
            labelname = IBSR_DIR+"IBSR_" + str(idx) + "/IBSR_" + str(idx) + "_ana_brainmask.nii.gz"
        img = nibabel.load(imagename)
        
        label = nibabel.load(labelname)
        label_affine = label.affine
        img_data = img.get_data()
        label_data = label.get_data()
        #数据集中brain为256*256*128*1 label为256*128*256*1，维度不一致
        label_data = label_data.transpose(0,2,1,3)
        images.append(img_data[:, :, :, 0])
        labels.append(label_data[:, :, :, 0])

    test_images = np.asarray(images, dtype=np.int32)
    test_labels = np.asarray(labels, dtype=np.int32)
    test_images = np.reshape(test_images, [test_images.shape[0] * test_images.shape[1], test_images.shape[2],
                                           test_images.shape[3]])
    test_labels = np.reshape(test_labels, [test_labels.shape[0] * test_labels.shape[1], test_labels.shape[2],
                                           test_labels.shape[3]])
    
    test_images = np.expand_dims(test_images, axis=-1)

    sequence_len = test_images.shape[0]
    edges = []
    for i in range(0, sequence_len):
        test_labels_slice = test_labels[i]
        slice_edge = cv2.Canny(np.uint8(test_labels_slice), 1, 2)
        edges.append(slice_edge)
    
    test_edge_labels = np.asarray(edges, dtype=np.int32)
    test_edge_labels = normalization(test_edge_labels)
    test_edge_labels = np.expand_dims(test_edge_labels, axis=-1)

    # print(test_images.shape)
    # print(test_labels.shape)

    return test_images, test_edge_labels

def load_oasis_withedge_train():
    images = []
    labels = []

    for idx in range(1, 22):
        if idx == 8 or idx == 24 or idx == 36:
            continue
        elif idx == 7:
            imagename = OASIS_DISC1_DIR+"OAS1_0007_MR1/PROCESSED/MPRAGE/T88_111/OAS1_0007_MR1_mpr_n3_anon_111_t88_gfc.hdr"
            labelname = OASIS_DISC1_DIR+"OAS1_0007_MR1/PROCESSED/MPRAGE/T88_111/OAS1_0007_MR1_mpr_n3_anon_111_t88_masked_gfc.hdr"
        elif idx in (15, 16, 20, 26, 34, 38, 39):
            imagename = OASIS_DISC1_DIR+"OAS1_00"+str(idx)+"_MR1/PROCESSED/MPRAGE/T88_111/OAS1_00"+str(idx)+"_MR1_mpr_n3_anon_111_t88_gfc.hdr"
            labelname = OASIS_DISC1_DIR+"OAS1_00"+str(idx)+"_MR1/PROCESSED/MPRAGE/T88_111/OAS1_00"+str(idx)+"_MR1_mpr_n3_anon_111_t88_masked_gfc.hdr"
        elif idx < 10:
            imagename = OASIS_DISC1_DIR+"OAS1_000"+str(idx)+"_MR1/PROCESSED/MPRAGE/T88_111/OAS1_000"+str(idx)+"_MR1_mpr_n4_anon_111_t88_gfc.hdr"
            labelname = OASIS_DISC1_DIR+"OAS1_000"+str(idx)+"_MR1/PROCESSED/MPRAGE/T88_111/OAS1_000"+str(idx)+"_MR1_mpr_n4_anon_111_t88_masked_gfc.hdr"
        else:
            imagename = OASIS_DISC1_DIR+"OAS1_00"+str(idx)+"_MR1/PROCESSED/MPRAGE/T88_111/OAS1_00"+str(idx)+"_MR1_mpr_n4_anon_111_t88_gfc.hdr"
            labelname = OASIS_DISC1_DIR+"OAS1_00"+str(idx)+"_MR1/PROCESSED/MPRAGE/T88_111/OAS1_00"+str(idx)+"_MR1_mpr_n4_anon_111_t88_masked_gfc.hdr"
        img = nibabel.load(imagename)
        label = nibabel.load(labelname)
        img_data = img.get_data()
        label_data = label.get_data()
        label_data[label_data>0] = 1
        images.append(img_data[:, :, :, 0])
        labels.append(label_data[:, :, :, 0])
    
    train_images = np.asarray(images, dtype=np.int32)
    train_labels = np.asarray(labels, dtype=np.int32)
    train_images = np.reshape(train_images, [train_images.shape[0]*train_images.shape[1], train_images.shape[2], train_images.shape[3]])
    train_labels = np.reshape(train_labels, [train_labels.shape[0]*train_labels.shape[1], train_labels.shape[2], train_labels.shape[3]])
    train_images = np.expand_dims(train_images, axis=-1)
    sequence_len = train_images.shape[0]
    edges = []
    for i in range(0, sequence_len):
        train_labels_slice = train_labels[i]
        slice_edge = cv2.Canny(np.uint8(train_labels_slice), 1, 2)
        edges.append(slice_edge)
    
    edge_labels = np.asarray(edges, dtype=np.int32)
    edge_labels = normalization(edge_labels)
    edge_labels = np.expand_dims(edge_labels, axis=-1)

    return train_images, edge_labels

def load_oasis_withedge_test():
    images = []
    labels = []

    for idx in range(22, 43):
        if idx == 8 or idx == 24 or idx == 36:
            continue
        elif idx == 7:
            imagename = OASIS_DISC1_DIR+"OAS1_0007_MR1/PROCESSED/MPRAGE/T88_111/OAS1_0007_MR1_mpr_n3_anon_111_t88_gfc.hdr"
            labelname = OASIS_DISC1_DIR+"OAS1_0007_MR1/PROCESSED/MPRAGE/T88_111/OAS1_0007_MR1_mpr_n3_anon_111_t88_masked_gfc.hdr"
        elif idx in (15, 16, 20, 26, 34, 38, 39):
            imagename = OASIS_DISC1_DIR+"OAS1_00"+str(idx)+"_MR1/PROCESSED/MPRAGE/T88_111/OAS1_00"+str(idx)+"_MR1_mpr_n3_anon_111_t88_gfc.hdr"
            labelname = OASIS_DISC1_DIR+"OAS1_00"+str(idx)+"_MR1/PROCESSED/MPRAGE/T88_111/OAS1_00"+str(idx)+"_MR1_mpr_n3_anon_111_t88_masked_gfc.hdr"
        elif idx < 10:
            imagename = OASIS_DISC1_DIR+"OAS1_000"+str(idx)+"_MR1/PROCESSED/MPRAGE/T88_111/OAS1_000"+str(idx)+"_MR1_mpr_n4_anon_111_t88_gfc.hdr"
            labelname = OASIS_DISC1_DIR+"OAS1_000"+str(idx)+"_MR1/PROCESSED/MPRAGE/T88_111/OAS1_000"+str(idx)+"_MR1_mpr_n4_anon_111_t88_masked_gfc.hdr"
        else:
            imagename = OASIS_DISC1_DIR+"OAS1_00"+str(idx)+"_MR1/PROCESSED/MPRAGE/T88_111/OAS1_00"+str(idx)+"_MR1_mpr_n4_anon_111_t88_gfc.hdr"
            labelname = OASIS_DISC1_DIR+"OAS1_00"+str(idx)+"_MR1/PROCESSED/MPRAGE/T88_111/OAS1_00"+str(idx)+"_MR1_mpr_n4_anon_111_t88_masked_gfc.hdr"
        img = nibabel.load(imagename)
        label = nibabel.load(labelname)
        label_affine = label.affine
        img_data = img.get_data()
        label_data = label.get_data()
        label_data[label_data>0] = 1
        images.append(img_data[:, :, :, 0])
        labels.append(label_data[:, :, :, 0])

    test_images = np.asarray(images, dtype=np.int32)
    test_labels = np.asarray(labels, dtype=np.int32)
    test_images = np.reshape(test_images, [test_images.shape[0] * test_images.shape[1], test_images.shape[2], test_images.shape[3]])
    test_labels = np.reshape(test_labels, [test_labels.shape[0] * test_labels.shape[1], test_labels.shape[2], test_labels.shape[3]])
    test_images = np.expand_dims(test_images, axis=-1)
    
    sequence_len = test_images.shape[0]
    edges = []
    for i in range(0, sequence_len):
        test_labels_slice = test_labels[i]
        slice_edge = cv2.Canny(np.uint8(test_labels_slice), 1, 2)
        edges.append(slice_edge)
    
    test_edge_labels = np.asarray(edges, dtype=np.int32)
    test_edge_labels = normalization(test_edge_labels)
    test_edge_labels = np.expand_dims(test_edge_labels, axis=-1)

    return test_images, test_edge_labels