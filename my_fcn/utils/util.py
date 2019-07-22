import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils.NewDataLoader import load_ibsr_2d_test, load_ibsr_2d_test_sv_seg, load_lpba40_2d_test, load_lpba40_2d_test_sv_seg


def shuffle_data(imgs, labels):
    imgs = np.asarray(imgs)
    labels = np.asarray(labels)
    r = np.random.permutation(len(imgs))
    return imgs[r], labels[r]


def dice_coefficient(true, pred):
    intersection = tf.reduce_sum(true*pred)
    return (2*intersection)/(tf.reduce_sum(true)+tf.reduce_sum(pred))

def dice_coefficient_numpy(arg1, arg2):
    intersection = np.sum(arg1 * arg2)
    return (2 * intersection) / (np.sum(arg1) + np.sum(arg2))

def sensitivity(true, pred):
    data_shape = tf.shape(true) #dim = 3
    total = data_shape[0]*data_shape[1]*data_shape[2]
    TP = tf.reduce_sum(true*pred)
    FP = tf.reduce_sum(pred)-TP
    FN = tf.reduce_sum(true)-TP
    TN = total-FP-TP-FN
    return TP/(TP+FN)

def specificity(true, pred):
    data_shape = tf.shape(true)  # dim = 3
    total = data_shape[0] * data_shape[1] * data_shape[2]
    TP = tf.reduce_sum(true * pred)
    FP = tf.reduce_sum(pred) - TP
    FN = tf.reduce_sum(true) - TP
    TN = total - FP - TP - FN
    return TN / (TN + FP)

#计算超体素方法处理后的dice
def dice_after_sv(data_set, test_num):
    if data_set=='IBSR':
        # shape = [test_num*256, 256, 128]
        _, true_seg, _ = load_ibsr_2d_test()
        sv_seg = load_ibsr_2d_test_sv_seg()
    elif data_set=='LPBA40':
        # shape = [test_num*256, 124, 256]
        _, true_seg, _ = load_lpba40_2d_test()
        sv_seg = load_lpba40_2d_test_sv_seg()

    dices = 0
    for i in range(0, test_num):
        true_data = true_seg[i*256:i*256+256]
        sv_data = sv_seg[i*256:i*256+256]
        dice = dice_coefficient_numpy(true_data, sv_data)
        print("test sample "+str(i)+" dice: ", dice)
        dices += dice

    print("test set average dice: ", dices/test_num)