import nibabel
import numpy as np

def normalization(data):
    max = np.max(data)
    min = np.min(data)
    data = (data - min) / (max - min)
    data = data.astype(np.uint8)
    # print(data.dtype)
    return data

def LoadData():
    seg_name = "D:/DL/my_fcn/results/lpba40/fcn_seg/lpba40_seg1.nii.gz"
    # seg_name = "D:/DL/my_fcn/results/lpba40/inception_seg/lpba40_seg1.nii.gz"
    truth_name = "E:/degree thesis/data/LPBA40/native_space/S23/S23.native.brain.mask.hdr"
    seg = nibabel.load(seg_name)
    truth = nibabel.load(truth_name)
    affine = truth.affine
    seg_data = seg.get_data()
    truth_data = truth.get_data()
    seg_data = seg_data[:, :, :, 0]
    truth_data = truth_data[:, :, :, 0]
    truth_data = normalization(truth_data)
    # print(seg_data.dtype, truth_data.dtype)
    return seg_data, truth_data, affine

#生成error map
def generate_error_map(seg, truth):
    error_map = np.bitwise_xor(seg, truth)
    return error_map


if __name__ == '__main__':
    segmentation, truth, affine = LoadData()
    error_map_data = generate_error_map(segmentation, truth)
    error_map = nibabel.Nifti1Image(error_map_data, affine)
    nibabel.save(error_map, "D:/DL/my_fcn/results/error_map/fcn_error.nii.gz")