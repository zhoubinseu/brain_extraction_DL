import train_rcf
import test_rcf
import execute_rcf
import rcf_utils
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    train_images, edge_labels = rcf_utils.load_lpba40_withedge_train()
    test_images, test_edge_labels = rcf_utils.load_lpba40_withedge_test()
    # train_images, edge_labels = rcf_utils.load_ibsr18_withedge_train()
    # test_images, test_edge_labels = rcf_utils.load_ibsr18_withedge_test()
    # train_images, edge_labels = rcf_utils.load_oasis_withedge_train()
    # test_images, test_edge_labels = rcf_utils.load_oasis_withedge_test()
    train_images = np.asarray(train_images)
    edge_labels = np.asarray(edge_labels)
    test_images = np.asarray(test_images)
    test_edge_labels = np.asarray(test_edge_labels)
    # print(train_images.shape)
    # print(edge_labels.shape)
    # print(test_images.shape)
    # print(test_edge_labels.shape)

    # img_slice = train_images[384, :, :, 0]
    # label_slice = edge_labels[384, :, :, 0]

    # fig = plt.figure()
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(img_slice, cmap='gray')
    # fig.add_subplot(1, 2, 2)
    # plt.imshow(label_slice, cmap='gray')
    # plt.show()
    # train_rcf.train(train_images, edge_labels, lr=1e-4, batchsize=32, epochs=25, display_step=10, saved_model_name="rcf_oasis")

    # prediction = test_rcf.test(test_images, test_edge_labels, batchsize=256, saved_model_name="rcf_ibsr18_30epoch")
    
    prediction = execute_rcf.edge_detect(test_images, batchsize=256, saved_model_name="rcf_lpba40")
    
    # print(prediction.shape)
    test_img_slice = test_images[128,:,:,0]
    pred_slice = prediction[128,:,:,0]
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(test_img_slice, cmap='gray')
    fig.add_subplot(1, 2, 2)
    plt.imshow(pred_slice, cmap='gray')
    plt.show()
