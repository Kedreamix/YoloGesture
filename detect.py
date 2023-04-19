#-----------------------------------------------------------------------#
#   detect.py 是用来尝试利用小模型半自动化进行标注数据
#-----------------------------------------------------------------------#
import numpy as np
from PIL import Image
from get_yaml import get_config

from yolo import YOLO
from gen_annotation import GEN_Annotations

if __name__ == "__main__":# 配置文件
    # 配置文件
    config = get_config()
    yolo = YOLO()

    dir_detect_path = config['dir_detect_path']
    detect_save_path   = config['detect_save_path']
    
    import os
    from tqdm import tqdm
    
    img_names = os.listdir(dir_detect_path)
    for img_name in tqdm(img_names):
        
        if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            # if int(img_name.split('.')[0][-4:]) < 355:
            #     continue
            image_path  = os.path.join(dir_detect_path, img_name)
            image       = Image.open(image_path)
            boxes       = yolo.get_box(image)
            if not os.path.exists(detect_save_path):
                os.makedirs(detect_save_path)

            annotation        = GEN_Annotations(img_name)
            w,h = np.array(np.shape(image)[0:2])
            annotation.set_size(w,h,3)
            if boxes:
                for box in boxes:
                    label,ymin,xmin,ymax,xmax = box
                    annotation.add_pic_attr(label,xmin,ymin,xmax,ymax)
                annotation_path  = os.path.join(detect_save_path, img_name.split('.')[0])
                annotation.savefile("{}.xml".format(annotation_path ))
            # print(img_name,'已经半自动标注')