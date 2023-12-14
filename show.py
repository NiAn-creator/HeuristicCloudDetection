import os,shutil
import cv2
import numpy as np
from tqdm import tqdm

mask_path = '/Users/liuyang/Documents/VisInt/GF1_WFV_dataset/vis_bigpatch_mask/'
tiff_path = '/Users/liuyang/Documents/VisInt/GF1_WFV_dataset/vis_bigpatch_tiff/'

save_path = '/Users/liuyang/Documents/CLDiff-TGRS2023/GF1_CLDiff_map/'


for file in os.listdir(save_path):
   cc = file.replace('.png','').split('_')
   if len(cc) ==3:
       os.rename(save_path + file, save_path + file.replace('.png', '_0img.png'))





