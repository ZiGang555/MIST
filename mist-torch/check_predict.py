import SimpleITK as sitk
import numpy as np
import os

# 替换为你的NIfTI文件路径
nifti_file_path = 'results/predictions/train/raw/420.nii.gz'
def list_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(os.path.join(root, file))
# 读取NIfTI文件
img = sitk.ReadImage(nifti_file_path)

# 将图像转换为numpy数组
img_array = sitk.GetArrayFromImage(img)

# 统计数组中的唯一值及其出现的次数
unique_values = np.unique(img_array)

print(f"Value: {unique_values}")
