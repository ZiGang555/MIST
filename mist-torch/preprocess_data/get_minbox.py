import SimpleITK as sitk
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import os

def crop_and_extract_nodules(input_path, mask_path, crop_size):
    # 加载NIfTI图像和mask
    image = sitk.ReadImage(input_path)
    mask = sitk.ReadImage(mask_path)
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)

    # 获取肺部结节的位置
    nodule_indices = np.where(mask_array > 0)


    # 计算肺部结节区域的中心
    center_z = int(np.mean(nodule_indices[0]))
    center_y = int(np.mean(nodule_indices[1]))
    center_x = int(np.mean(nodule_indices[2]))

    # 计算裁剪盒子的边界
    min_z = max(0, center_z - crop_size[0] // 2)
    max_z = min(image_array.shape[0], center_z + crop_size[0] // 2)
    min_y = max(0, center_y - crop_size[1] // 2)
    max_y = min(image_array.shape[1], center_y + crop_size[1] // 2)
    min_x = max(0, center_x - crop_size[2] // 2)
    max_x = min(image_array.shape[2], center_x + crop_size[2] // 2)
    print(min_z, max_z, min_y, max_y, min_x, max_x)
    # 检查边界情况
    if max_z - min_z < crop_size[0]:
        diff = crop_size[0] - (max_z - min_z)
        min_z = max(0, min_z - diff // 2)
        max_z = min(image_array.shape[0], max_z + diff // 2)
    if max_y - min_y < crop_size[1]:
        diff = crop_size[1] - (max_y - min_y)
        min_y = max(0, min_y - diff // 2)
        max_y = min(image_array.shape[1], max_y + diff // 2)
    if max_x - min_x < crop_size[2]:
        diff = crop_size[2] - (max_x - min_x)
        min_x = max(0, min_x - diff // 2)
        max_x = min(image_array.shape[2], max_x + diff // 2)

    # 裁剪图像和mask
    cropped_image = image_array[min_z:max_z, min_y:max_y, min_x:max_x]
    cropped_mask = mask_array[min_z:max_z, min_y:max_y, min_x:max_x]

    # 创建一个新的SimpleITK图像对象
    cropped_image = sitk.GetImageFromArray(cropped_image)
    cropped_mask = sitk.GetImageFromArray(cropped_mask)

    # 设置裁剪后的图像的元数据
    cropped_image.SetSpacing(image.GetSpacing())
    cropped_image.SetOrigin(image.GetOrigin())

    # 设置裁剪后的mask的元数据
    cropped_mask.SetSpacing(mask.GetSpacing())
    cropped_mask.SetOrigin(mask.GetOrigin())
    
    return cropped_image, cropped_mask

# 调用裁剪和提取肺部结节的函数
input_path = 'dataset/LUNA_Train/0002/image.nii.gz'
mask_path = 'dataset/LUNA_Train/0002/mask.nii.gz'
output_path = './'

cropped_image, cropped_mask= crop_and_extract_nodules(input_path, mask_path,(64,64,64))
# 对裁剪后的mask进行膨胀和腐蚀操作
dilated_mask = sitk.BinaryDilate(cropped_mask, [1,1,1])  # 使用3x3x3的结构元素进行膨胀操作
eroded_mask = sitk.BinaryErode(cropped_mask, [1,1,1])  # 使用3x3x3的结构元素进行腐蚀操作
# 保存裁剪后的图像和mask为NIfTI格式
cropped_path = os.path.join(output_path, "cropped_image.nii.gz")
sitk.WriteImage(cropped_image, cropped_path)
sitk.WriteImage(cropped_mask, cropped_path.replace(".nii.gz", "_mask.nii.gz"))
sitk.WriteImage(dilated_mask, cropped_path.replace(".nii.gz", "_dilatedMask.nii.gz"))
sitk.WriteImage(eroded_mask, cropped_path.replace(".nii.gz", "_erodedMask.nii.gz"))