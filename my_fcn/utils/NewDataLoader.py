import re
import numpy as np
import scipy.misc as misc
from os.path import join
import nibabel
import matplotlib.pyplot as plt
import cv2

#OASIS数据集 使用PROCESSED/MPRAGE/T88_111中的数据，将masked处理成label

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

#S09和S10的shape为（256,120,256）其余为（256,124,256）
def load_lpba40_2d_train():
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

    # train_images = normalization(train_images)
    train_labels = normalization(train_labels)

    # print(train_images.shape)
    # print(train_labels.shape)

    return train_images, train_labels


def load_lpba40_2d_test():
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

    # test_images = normalization(test_images)
    test_labels = normalization(test_labels)

    # print(test_images.shape)
    # print(test_labels.shape)

    return test_images, test_labels, label_affine


def load_ibsr_2d_test():
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

    # test_images = normalization(test_images)
    test_labels = normalization(test_labels)

    # print(test_images.shape)
    # print(test_labels.shape)

    return test_images, test_labels, label_affine

#超体素处理后的测试集分割结果 shape [256, 256, 128]
def load_ibsr_2d_test_sv_seg():
    labels = []
    for idx in range(0, 9):
        labelname = "D:/DL/my_fcn/results/ibsr/probability/ibsr_svpooling_" + str(idx) + ".nii.gz"
        label = nibabel.load(labelname)
        label_data = label.get_data()
        # print(label_data.shape)
        labels.append(label_data)
    sv_seg = np.asarray(labels, dtype=np.int32)
    # shape [test_num*256, 256, 128]
    sv_seg = np.reshape(sv_seg, [sv_seg.shape[0] * sv_seg.shape[1], sv_seg.shape[2],
                                 sv_seg.shape[3]])
    return sv_seg


def load_lpba40_2d_test_sv_seg():
    labels = []
    for idx in range(0, 28):
        labelname = "D:/DL/my_fcn/results/lpba40/probability/lpba40_svpooling_" + str(idx) + ".nii.gz"
        label = nibabel.load(labelname)
        label_data = label.get_data()
        # print(label_data.shape)
        labels.append(label_data)
    sv_seg = np.asarray(labels, dtype=np.int32)
    # shape [test_num*256, 124, 256]
    sv_seg = np.reshape(sv_seg, [sv_seg.shape[0] * sv_seg.shape[1], sv_seg.shape[2],
                                 sv_seg.shape[3]])
    return sv_seg


def load_ibsr_2d_train():
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
        images.append(img_data[:, :, :, 0])
        labels.append(label_data[:, :, :, 0])

    train_images = np.asarray(images, dtype=np.int32)
    train_labels = np.asarray(labels, dtype=np.int32)
    # print(train_images.shape)
    # print(train_labels.shape)

    train_images = np.reshape(train_images, [train_images.shape[0]*train_images.shape[1], train_images.shape[2], train_images.shape[3]])
    train_labels = np.reshape(train_labels, [train_labels.shape[0]*train_labels.shape[1], train_labels.shape[2], train_labels.shape[3]])

    train_images = np.expand_dims(train_images, axis=-1)

    # train_images = normalization(train_images)
    train_labels = normalization(train_labels)

    # print(train_images.shape)
    # print(train_labels.shape)

    return train_images, train_labels

#tran_image的channel=2， channel1为磁共振图像，channel2为对应标签提取的边缘
def load_ibsr_withedge_train():
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
    # print(train_images.shape)
    # print(train_labels.shape)

    train_images = np.reshape(train_images, [train_images.shape[0]*train_images.shape[1], train_images.shape[2], train_images.shape[3]])
    train_labels = np.reshape(train_labels, [train_labels.shape[0]*train_labels.shape[1], train_labels.shape[2], train_labels.shape[3]])
    # print(train_images.shape)
    # print(train_labels.shape)
    #对每个index的label提取edge，作为该index位置的image的第二个chennel
    sequence_len = train_images.shape[0]
    train_images_with_edge = []
    for i in range(0, sequence_len):
        train_label_slice = train_labels[i]
        train_image_slice = train_images[i]
        label_edge = cv2.Canny(np.uint8(train_label_slice), 100, 200)
        image_slice_with_edge = np.stack((train_image_slice, label_edge), axis=2)
        train_images_with_edge.append(image_slice_with_edge)
    
    train_images = np.asarray(train_images_with_edge, dtype=np.int32)
    train_labels = normalization(train_labels)
    # print(train_images.shape)
    # print(train_labels.shape)

    return train_images, train_labels

def load_ibsr_withedge_test():
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
    
    sequence_len = test_images.shape[0]
    test_images_with_edge = []
    for i in range(0, sequence_len):
        test_label_slice = test_labels[i]
        test_image_slice = test_images[i]
        label_edge = cv2.Canny(np.uint8(test_label_slice), 100, 200)
        image_slice_with_edge = np.stack((test_image_slice, label_edge), axis=2)
        test_images_with_edge.append(image_slice_with_edge)
    
    test_images = np.asarray(test_images_with_edge, dtype=np.int32)
    test_labels = normalization(test_labels)

    # print(test_images.shape)
    # print(test_labels.shape)

    return test_images, test_labels, label_affine

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

    #对每个index的label提取edge，作为该index位置的image的第二个chennel
    sequence_len = train_images.shape[0]
    train_images_with_edge = []
    for i in range(0, sequence_len):
        # train_label_slice = train_labels[i]
        train_image_slice = train_images[i]
        # label_edge = cv2.Canny(np.uint8(train_label_slice), 100, 200)
        img_edge = cv2.Canny(np.uint8(train_image_slice), 200, 300)
        image_slice_with_edge = np.stack((train_image_slice, img_edge), axis=2)
        train_images_with_edge.append(image_slice_with_edge)
    
    train_images = np.asarray(train_images_with_edge, dtype=np.int32)
    train_labels = normalization(train_labels)

    # print(train_images.shape)
    # print(train_labels.shape)

    return train_images, train_labels

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

    sequence_len = test_images.shape[0]
    test_images_with_edge = []
    for i in range(0, sequence_len):
        # test_label_slice = test_labels[i]
        test_image_slice = test_images[i]
        # label_edge = cv2.Canny(np.uint8(test_label_slice), 100, 200)
        img_edge = cv2.Canny(np.uint8(test_image_slice), 200, 300)
        image_slice_with_edge = np.stack((test_image_slice, img_edge), axis=2)
        test_images_with_edge.append(image_slice_with_edge)
    
    test_images = np.asarray(test_images_with_edge, dtype=np.int32)
    test_labels = normalization(test_labels)

    # print(test_images.shape)
    # print(test_labels.shape)

    return test_images, test_labels, label_affine

def load_oasis_2d_train():
    #data shape: [176,208,176,1]
    #某些数据存在特殊情况
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
    train_labels = normalization(train_labels)

    # print(train_images.shape)
    # print(train_labels.shape)
    # img_slice = img_data[88,:,:,0]
    # label_slice = label_data[88,:,:,0]
    # fig = plt.figure()
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(img_slice, cmap='gray')
    # fig.add_subplot(1, 2, 2)
    # plt.imshow(label_slice, cmap='gray')
    # plt.show()
    return train_images, train_labels

def load_oasis_2d_test():
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
    test_labels = normalization(test_labels)

    # print(test_images.shape)
    # print(test_labels.shape)

    return test_images, test_labels, label_affine

def testloaddata():
    imagename = LPBA40_DIR+"S01/S01.native.mri.hdr"
    labelname = LPBA40_DIR+"S01/S01.native.brain.mask.hdr"
    img = nibabel.load(imagename)
    label = nibabel.load(labelname)
    label_affine = label.affine
    img_data = img.get_data()
    label_data = label.get_data()
    print(img_data.shape)
    print(label_data.shape)
    img_data = img_data[:, :, :, 0]
    label_data = label_data[:, :, :, 0]
    print(img_data.shape)
    print(label_data.shape)
    # print(type(img_data))
    # print(type(label_data))
    # slice1
    img_slice1 = img_data[128, :, :]
    label_slice1 = label_data[128, :, :]
    # # slice2
    # img_slice2 = img_data[:, 62, :]
    # label_slice2 = label_data[:, 62, :]
    imgedges_slice2 = cv2.Canny(np.uint8(img_slice1), 200, 300)
    labeledges_slice2 = cv2.Canny(label_slice1,100,200)
    # three_channel_image = np.stack((img_slice2, imgedges_slice2, labeledges_slice2),axis=2)
    print(img_slice1.shape)
    print(label_slice1.shape)
    # print(three_channel_image.shape)
    # # slice3
    # img_slice3 = img_data[:, :, 128]
    # label_slice3 = label_data[:, :, 128]

    # print(type(img_slice2))
    # print(type(label_slice2))

    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.imshow(img_slice1, cmap='gray')
    fig.add_subplot(1, 3, 2)
    plt.imshow(label_slice1, cmap='gray')
    fig.add_subplot(1, 3, 3)
    plt.imshow(labeledges_slice2, cmap='gray')
    # fig.add_subplot(2, 2, 3)
    # plt.imshow(imgedges_slice2, cmap='gray')
    # fig.add_subplot(2, 2, 4)
    # plt.imshow(labeledges_slice2, cmap='gray')
    plt.show()

# testloaddata()