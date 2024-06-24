import SimpleITK as sitk
import numpy as np

def remove_small_areas(input_path, output_path, size_threshold):
    """
    应用RSA技术处理三维肺部结节图像分割结果，去除小于size_threshold的噪点区域。
    
    参数:
    - input_path: 输入NIfTI图像的路径。
    - output_path: 处理后的图像保存路径。
    - size_threshold: 体积阈值，用于去除小区域。
    """
    
    # 读取NIfTI图像
    img = sitk.ReadImage(input_path)
    
    # 将图像转换为标签图
    img_labels = sitk.ConnectedComponent(img)
    
    # 去除小区域
    img_labels_clean = sitk.RelabelComponent(img_labels, minimumObjectSize=size_threshold, sortByObjectSize=True)
    
    # 将去除小区域后的标签图转换回二进制图像
    img_clean = img_labels_clean > 0
    
    # 保存处理后的图像
    sitk.WriteImage(img_clean, output_path)
    print(f"Processed image saved to {output_path}")

# 输入和输出图像路径
input_nifti_path = 'results/nnunet/dice_ce/predictions/test/0505.nii.gz'
output_nifti_path = 'postprocess_preds/after_RSA.nii'

# 定义去除体积小于1000体素的区域
size_threshold = 100

# 调用函数
remove_small_areas(input_nifti_path, output_nifti_path, size_threshold)
