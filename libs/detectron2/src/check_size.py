import cv2
import os 
path = "/storageStudents/K2018/tiendv/tiennv/mfd_2021/detectron2/output/cfg2/Va01/imgs"

for file in os.listdir(path):
    img_file = os.path.join(path,file)
    img = cv2.imread(img_file)
    print(img.shape)
