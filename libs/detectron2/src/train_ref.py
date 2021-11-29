# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog

import json
import os
import pandas as pd
import csv
import time

import detectron2.data.transforms as T
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, build_detection_train_loader   # the default mapper

#from detectron2.evaluation.coco_evaluation import COCOEvaluator
import os

import datetime 
import argparse
from config import  ANNOT_TRAIN_JSON,  PATH_TRAIN_IMG, NAME_TRAIN, \
ANNOT_VALID_JSON, PATH_VALID_IMG, NAME_VALID, \
ANNOT_TEST_JSON, PATH_TEST_IMG, NAME_TEST, \
ANNOT_REF_JSON, PATH_REF_IMG, NAME_REF,   \
ANNOT_REF_JSON_2, PATH_REF_IMG_2, NAME_REF_2,\
PATH_MODEL_ZOO,PATH_MODEL_FINAL 
from src import register_data



class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

def train(pretrain, no_iter, model_final, model_zoo_path, num_woker, batch_size, log_file):
  setup_logger(log_file)

  cfg = get_cfg()

  cfg.merge_from_file(model_zoo.get_config_file(model_zoo_path))
  cfg.DATASETS.TRAIN = (NAME_REF_2,)
  cfg.DATASETS.TEST = (NAME_REF_2,)

  # cfg.DATALOADER.NUM_WORKERS = num_woker
  if pretrain == True:
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_zoo_path)  # Let training initialize from model zoo
  else:
    cfg.MODEL.WEIGHTS = ""
  cfg.SOLVER.IMS_PER_BATCH = batch_size
  cfg.SOLVER.BASE_LR = 0.001


  cfg.SOLVER.WARMUP_ITERS = 1000
  print("Numbers Iter: ", no_iter)
  cfg.SOLVER.MAX_ITER = no_iter #adjust up if val mAP is still rising, adjust down if overfit
  cfg.SOLVER.STEPS = (1000, 1500)
  cfg.SOLVER.GAMMA = 0.05


  cfg.OUTPUT_DIR = model_final
  cfg.SOLVER.CHECKPOINT_PERIOD = 500



  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8
  # Train model Mathematical formula detection
  # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 # your number of classes + 1
  # cfg.MODEL.RETINANET.NUM_CLASSES = 2
  # Train model detect references layout
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # your number of classes + 1
  cfg.MODEL.RETINANET.NUM_CLASSES = 1
  cfg.TEST.EVAL_PERIOD = 1000
  # augmentation data 
  train_augmentations = [
    T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
    T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
    T.RandomLighting(0.5),
    T.RandomRotation(angle = [30,45]),
    T.RandomContrast(0,255)
    ]
  dataloader = build_detection_train_loader(cfg,
   mapper=DatasetMapper(cfg, is_train=True, augmentations=train_augmentations)
   )


  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
  trainer = CocoTrainer(cfg)
  trainer.resume_or_load(resume=True)
  trainer.train()

def main(args):
  train(args.pretrain, args.no_iter, args.model, args.model_zoo, args.woker, args.batch_size, args.log_file)

if __name__ == "__main__" :
  parser = argparse.ArgumentParser(description='Config train Faster RCNN')
  parser.add_argument('--pretrain', '-p', default = False, 
                      type=bool, help='pretrain p = 1 otherwise p = 0 ')
  parser.add_argument('--no_iter', '-n', required=True, 
                      type=int, help='Number of iter ')
  parser.add_argument('--model', '-m', default = './model', 
                      type=str, help='the path save model ')
  parser.add_argument('--model_zoo', '-mz', default = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml", 
                      type=str, help='the path model zoo ')            
  parser.add_argument('--woker', '-w', default = 4, 
                      type=int, help='The number worker ')
  parser.add_argument('--batch_size', '-bs', default = 4, 
                      type=int, help='Batch size  ')
  parser.add_argument('--log_file', '-lg', default = './log/config1', 
                      type=str, help='The path folder to save logfile')   
  args = parser.parse_args()
  print("INFO argument \n{}\n{}\n{}\n{}\n{}\n{}\n{}".format(args.pretrain, args.no_iter, args.model, args.model_zoo, args.woker, args.batch_size, args.log_file))
  main(args)
