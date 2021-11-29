# import some common libraries
import numpy as np
import cv2
import random
import json
import os
import pandas as pd
import csv
import time
import glob
import os
import datetime 
import argparse

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode

from config import  ANNOT_TRAIN_JSON,  PATH_TRAIN_IMG, NAME_TRAIN, \
ANNOT_VALID_JSON, PATH_VALID_IMG, NAME_VALID, \
ANNOT_TEST_JSON, PATH_TEST_IMG, NAME_TEST, \
PATH_MODEL_ZOO,PATH_MODEL_FINAL
import register_data


def predict(input_path, output_path, model_path, model_zoo_path ):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_zoo_path))
    cfg.DATASETS.TRAIN = (NAME_TRAIN,)
    cfg.DATASETS.TEST = (NAME_VALID,)
    cfg.SOLVER.IMS_PER_BATCH = 4

    cfg.OUTPUT_DIR = model_path

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 # your number of classes + 1
    cfg.MODEL.RETINANET.NUM_CLASSES = 2

    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # your number of classes + 1
    # cfg.MODEL.RETINANET.NUM_CLASSES = 1

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    test_metadata = MetadataCatalog.get(NAME_VALID).set(thing_classes = ['embedded', 'isolated'])
    print(input_path)
    for imageName in glob.glob(input_path + '/*jpg'):
        im = cv2.imread(imageName)
        outputs = predictor(im)
        # print('outputs: ', outputs)
        v = Visualizer(im[:, :, ::1],
                    metadata=test_metadata, 
                    scale=1
                        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        save_path = os.path.join(output_path,"imgs",imageName.split("/")[-1])
        cv2.imwrite(save_path,out.get_image()[:, :, ::-1])
        print("Saved image: ",save_path)
        # Save bbox in file
        # print("results file : ", imageName.split("/")[-1].split(".")[0] + '.txt')
        result_path = os.path.join(output_path,"results", imageName.split("/")[-1].split(".")[0] + '.txt')
        result_file = open(result_path, 'w')
        # print('bbox: ', outputs["instances"].to("cpu").pred_boxes)
        # print('scores: ', outputs["instances"].to("cpu").scores)
        # print("class: ",outputs["instances"].to("cpu").pred_classes)
        for id_bb in range(len(outputs["instances"].pred_boxes)):
            class_id = int(outputs["instances"].to('cpu').pred_classes[id_bb])
            bbox = [int(p) for p in outputs["instances"].to('cpu').pred_boxes.tensor.numpy().tolist()[id_bb] ]
            score = float(outputs["instances"].to('cpu').scores[id_bb])
            # print("class id {}, bbox {}, score {}".format(class_id, bbox, score))
            result_file.write("{},{},{},{},{}\n".format(bbox[0], bbox[1], bbox[2], bbox[3], class_id))
        result_file.close()
        print("Saved result: ",result_path )


def main(args):
    print("Hello world")
    predict(args.input_path, args.output_path, args.model, args.model_zoo)

def args_parse():
    
    parser = argparse.ArgumentParser(description="This argument predict detectron2")
    parser.add_argument('-i', '--input_path',  default="./Test",
                        help="This folder path of image that you want predict")
    parser.add_argument('-o', '--output_path', default = "./output",
                        help = "This path save predict result")
    parser.add_argument('-m', '--model', default = "./model",
                        help = "This path model that you want to choose")
    parser.add_argument('-mz', '--model_zoo', default = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
                        help = "This path deep network that you want to choose")

    return parser.parse_args()
if __name__ == "__main__":

    args = args_parse()
    print("Argument: \n{}\n{}\n{}\n{}".format(args.input_path, args.output_path,args.model, args.model_zoo))
    main(args)
