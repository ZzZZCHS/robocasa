import os
import shutil
from tqdm import tqdm


ori_root = "/ailab/user/huanghaifeng/work/robocasa_exps/robocasa/robocasa/models/assets/objects/objaverse"
tgt_root = "/ailab/user/huanghaifeng/work/robocasa_exps/robocasa/robocasa/objaverse"

os.makedirs(tgt_root, exist_ok=True)

for obj_class in tqdm(os.listdir(ori_root)):
    ori_class_dir = os.path.join(ori_root, obj_class)
    tgt_class_dir = os.path.join(tgt_root, obj_class)
    os.makedirs(tgt_class_dir, exist_ok=True)
    for obj_name in os.listdir(ori_class_dir):
        if not obj_name.startswith(obj_class):
            continue
        ori_obj_dir = os.path.join(ori_class_dir, obj_name)
        tgt_obj_dir = os.path.join(tgt_class_dir, obj_name)
        ori_image_path = os.path.join(ori_obj_dir, 'combined_views.png')
        if not os.path.exists(ori_image_path):
            continue
        os.makedirs(tgt_obj_dir, exist_ok=True)
        shutil.copy(ori_image_path, tgt_obj_dir)

