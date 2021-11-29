from detectron2.data.datasets import register_coco_instances
import datetime
from config import  ANNOT_TRAIN_JSON,  PATH_TRAIN_IMG, NAME_TRAIN, \
ANNOT_VALID_JSON, PATH_VALID_IMG, NAME_VALID, \
ANNOT_TEST_JSON, PATH_TEST_IMG, NAME_TEST, \
ANNOT_REF_JSON, PATH_REF_IMG, NAME_REF, \
ANNOT_TRAIN10_JSON, PATH_TRAIN10_IMG, NAME_TR10, \
ANNOT_REF_JSON_2, PATH_REF_IMG_2, NAME_REF_2 

register_coco_instances(NAME_TRAIN, {}, ANNOT_TRAIN_JSON, PATH_TRAIN_IMG)
register_coco_instances(NAME_VALID, {}, ANNOT_VALID_JSON, PATH_VALID_IMG)
register_coco_instances(NAME_TEST, {}, ANNOT_TEST_JSON, PATH_TEST_IMG)
register_coco_instances(NAME_REF, {}, ANNOT_REF_JSON, PATH_REF_IMG)
register_coco_instances(NAME_REF_2, {}, ANNOT_REF_JSON_2, PATH_REF_IMG_2)
register_coco_instances(NAME_TR10, {}, ANNOT_TRAIN10_JSON, PATH_TRAIN10_IMG)
