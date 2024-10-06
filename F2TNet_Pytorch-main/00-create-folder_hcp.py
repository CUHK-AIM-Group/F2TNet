from nilearn import datasets
import argparse
from imports import preprocess_data as Reader
import os
import shutil
import sys
import numpy as np



subject_id = np.loadtxt('/data/hzb/project/Brain_Predict_Score/ViTPre0219/F2TNet_Pytorch-main/data_hcp/subject_IDs.txt', delimiter=',')
for i in range(len(subject_id)):
    folder_name = str(int(subject_id[i]))
    os.makedirs('data_hcp/HCP_pcp/Gordon/filt_noglobal/'+folder_name)  # makedirs 创建文件时如果路径不存在会创建这个路径





