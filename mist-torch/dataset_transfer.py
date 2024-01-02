import os
import shutil
from tqdm import tqdm
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *

def convert_nnunet_to_msd(nnunet_root, msd_root):
    # 检查目录是否存在
    if not os.path.exists(nnunet_root):
        print(f"Error: Directory '{nnunet_root}' not found.")
        return
    
    # 创建 MSD 格式的目录结构
    msd_train_dir = os.path.join(msd_root, "LUNA_Train")
    msd_test_dir = os.path.join(msd_root, "LUNA_Test")

    os.makedirs(msd_train_dir, exist_ok=True)
    os.makedirs(msd_test_dir, exist_ok=True)

    # 复制训练集数据
    train_data_dir = os.path.join(nnunet_root, "nnUNet_raw_data", "Task501_Lung", "imagesTr")
    train_label_dir = os.path.join(nnunet_root, "nnUNet_raw_data", "Task501_Lung", "labelsTr")
    # i=0
    for subject_id in tqdm(os.listdir(train_data_dir), desc="Copying Train Data"):
        dest_path = os.path.join(msd_train_dir, subject_id[5:9])
        os.makedirs(dest_path, exist_ok=True)
        subject_data = os.path.join(train_data_dir, subject_id)
        subject_label = os.path.join(train_label_dir, subject_id[:9]+'.nii.gz')
        
        # 复制图像
        shutil.copy(subject_data, os.path.join(dest_path, 'image.nii.gz'))

        # 复制标签
        shutil.copy(subject_label, os.path.join(dest_path, 'mask.nii.gz'))
        # i +=1
        # if i==60:
        #     break


nnunet_root = "/home/hzg/data/users/hzg/project/bishe/Extended_nnUNet/DATASET/nnUNet_raw_base"  # 替换为实际的 nnU-Net 数据集路径
msd_root = "/home/hzg/data/users/hzg/project/bishe/MIST/mist-torch/dataset"  # 替换为实际的 MSD 格式数据集路径

convert_nnunet_to_msd(nnunet_root, msd_root)
# 创建json文件内容
json_dict = OrderedDict()
json_dict['task'] = "Lung_Seg"
json_dict['modality'] = "ct"
json_dict['train-data'] = "/home/hzg/data/users/hzg/project/bishe/MIST/mist-torch/dataset/LUNA_Train"
json_dict['test-data'] = "/home/hzg/data/users/hzg/project/bishe/MIST/mist-torch/dataset/LUNA_Test"
json_dict['mask'] = "mask.ni.gz"
json_dict['images'] = {"CT": ["image.nii.gz"]}

json_dict['labels'] = [0,1]
json_dict['final_classes'] = {"nodule": [1]}
# json_dict['numTraining'] = len(patient_names)
# json_dict['numTest'] = 0
# json_dict['training'] = [{'image': "./imagesTr/Lung_%s.nii.gz" % i, "label": "./labelsTr/Lung_%s.nii.gz" % i} for i in
#                             patient_names]
# json_dict['test'] = []

# 将字典写入json文件中
save_json(json_dict, join(msd_root, "dataset.json"))
