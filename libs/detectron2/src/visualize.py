#visualize training data

import random
import argparse
import sys
import cv2 
import os

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog

sys.path.append('./'), 
from config import  ANNOT_TRAIN_JSON,  PATH_TRAIN_IMG, NAME_TRAIN, \
ANNOT_VALID_JSON, PATH_VALID_IMG, NAME_VALID, \
ANNOT_TEST_JSON, PATH_TEST_IMG, NAME_TEST, \
ANNOT_REF_JSON, PATH_REF_IMG, NAME_REF, \
ANNOT_REF_JSON_2, PATH_REF_IMG_2, NAME_REF_2, \
PATH_MODEL_ZOO,PATH_MODEL_FINAL 
import register_data


def visualize(mode, Num_img, output_path):
  if mode == 0 :
    name_visualization = NAME_TRAIN

  elif mode == 1:
    name_visualization = NAME_REF_2

  else:
    name_visualization = NAME_TEST

  my_dataset_metadata = MetadataCatalog.get(name_visualization)
  dataset_dicts = DatasetCatalog.get(name_visualization)
  print("Number random: ", Num_img)
  for d in random.sample(dataset_dicts, Num_img):
      img = cv2.imread(d["file_name"])
      visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_metadata, scale=0.5)
      vis = visualizer.draw_dataset_dict(d)
      print( d["file_name"].split("/")[-1])
      print(output_path)
      save_path = os.path.join(output_path, d["file_name"].split("/")[-1])
      cv2.imwrite(save_path, vis.get_image()[:, :, ::-1])
  
def main(args):
  visualize(args.mode, args.num_img, args.output)

if __name__ == "__main__" :
  parser = argparse.ArgumentParser(description='Config visualize train data')
  parser.add_argument('--mode', '-m', required=True,
                      type=bool, help='If you want visualize train data  mode = 1, otherwise mode = 2')
  parser.add_argument('--num_img', '-n', required=True, 
                      type=int, help='Number of image will visualize')
  parser.add_argument('--output', '-o', default = "./Visualize", 
                      type=str, help='The path output visualize file')
  args = parser.parse_args()
  main(args)


  '''
  CUDA_VISIBLE_DEVICES=6 python ./visualize.py \
      --mode 1 \
      --num_img 20\
      --output ./Visualize 
        
  '''