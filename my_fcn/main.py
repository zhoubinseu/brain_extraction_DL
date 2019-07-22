from scripts.train import train
from scripts.predict import predict
import scipy.io as sio
import numpy as np
from utils.NewDataLoader import load_lpba40_2d_train, load_lpba40_withedge_train, load_lpba40_2d_test, load_lpba40_withedge_test, load_ibsr_2d_train, load_ibsr_withedge_train, load_ibsr_2d_test, load_ibsr_withedge_test, load_oasis_2d_train, load_oasis_2d_test
import matplotlib.pyplot as plt
import nibabel
from utils.util import dice_after_sv
from execute_rcf import edge_detect

IBSR_PREDICTION_DIR = 'E:/Zhou/brain extraction/my_fcn/prediction/ibsr/'
LPBA_PREDICTION_DIR = 'E:/Zhou/brain extraction/my_fcn/prediction/lpba/'
OASIS_PREDICTION_DIR = 'E:/Zhou/brain extraction/my_fcn/prediction/oasis/'


if __name__ == '__main__':
    #读取IBSR
    # train_images, train_labels = load_ibsr_2d_train()
    # test_images, test_labels, label_affine = load_ibsr_2d_test()
    #读取LPBA40
    train_images, train_labels = load_lpba40_2d_train()
    test_images, test_labels, label_affine = load_lpba40_2d_test()
    #读取OASIS
    # train_images, train_labels = load_oasis_2d_train()
    # test_images, test_labels, label_affine = load_oasis_2d_test()


    train_images = np.asarray(train_images)
    train_labels = np.asarray(train_labels)
    test_images = np.asarray(test_images)
    test_labels = np.asarray(test_labels)

    # print(train_images.shape)
    # print(train_labels.shape)
    # print(test_images.shape)
    # print(test_labels.shape)

    #处理训练数据的Image
    #使用rcf提取image的脑组织轮廓  将mri_edges加入train_images,作为第二个channel
    # mri_edges = edge_detect(train_images, batchsize=176, saved_model_name="rcf_oasis")
    # print(mri_edges.shape)
    # train_images = np.squeeze(train_images, axis=-1)
    # mri_edges = np.squeeze(mri_edges, axis=-1)
    # sequence_len = train_images.shape[0]
    # train_images_with_edge = []
    # for i in range(0, sequence_len):
    #     train_images_slice = train_images[i]
    #     mri_edges_slice = mri_edges[i]
    #     image_edge_slice = np.stack((train_images_slice, mri_edges_slice), axis=2)
    #     train_images_with_edge.append(image_edge_slice)
    
    # train_images = np.asarray(train_images_with_edge, dtype=np.int32)
    
    #处理测试数据的Image，加入边缘信息
    mri_edges = edge_detect(test_images, batchsize=256, saved_model_name="rcf_lpba40")
    # print(mri_edges.shape)
    test_images = np.squeeze(test_images, axis=-1)
    mri_edges = np.squeeze(mri_edges, axis=-1)
    sequence_len = test_images.shape[0]
    test_images_with_edge = []
    for i in range(0, sequence_len):
        test_images_slice = test_images[i]
        mri_edges_slice = mri_edges[i]
        image_edge_slice = np.stack((test_images_slice, mri_edges_slice), axis=2)
        test_images_with_edge.append(image_edge_slice)
    
    test_images = np.asarray(test_images_with_edge, dtype=np.int32)

    print(train_images.shape)
    print(train_labels.shape)
    print(test_images.shape)
    print(test_labels.shape)

    #训练模型
    # train(model_name='region_conv', x=train_images, y=train_labels, num_class=2, lr=1e-4,
    #      wd=5e-4, batch_size=16, epochs=15, display_step=10,
    #      sparse=True, conti=False, saved_model_name='region_conv_lpba40_9train_9test_15epochs')

    #测试 保存测试集的分割结果
    preddata, probability = predict(model_name='segnet_inception_ddsc', x=test_images, y=test_labels, num_class=2, wd=5e-4,
			batch_size=128, sparse=True, saved_model_name='segnet_inception_ddsc_rcf_lpba40_19train_19test_15epochs')
    preddata = preddata.astype("uint8")
    for i in range(0, 19):
        data = np.expand_dims(preddata[i*256:i*256+256], axis=-1)
        img = nibabel.Nifti1Image(data, label_affine)
        nibabel.save(img, LPBA_PREDICTION_DIR+"lpba_seg"+str(i)+".nii.gz")

    print(preddata.shape)

    test_img_slice = test_images[128,:,:,0]
    test_label_slice = test_labels[128,:,:]
    pred_slice = preddata[128,:,:]
    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.imshow(test_img_slice, cmap='gray')
    fig.add_subplot(1, 3, 2)
    plt.imshow(test_label_slice, cmap='gray')
    fig.add_subplot(1, 3, 3)
    plt.imshow(pred_slice, cmap='gray')
    plt.show()

    

    #计算超体素方法处理后，测试集分割结果的dice
    # dice_after_sv(data_set='LPBA40', test_num=28)


    #save data
    # data = np.expand_dims(preddata, axis=-1)
    # print(data.shape)
    # img = nibabel.Nifti1Image(data, label_affine)
    # nibabel.save(img, "D:/DL/my_fcn/results/test/pred_0.nii.gz")