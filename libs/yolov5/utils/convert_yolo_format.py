"""
 Usage: python utils/convert_yolo_format.py \
                --data_dir /storageStudents/Datasets/ICDAR2021_MDF/Ts00 \
                --output_dir converted_dataset/
"""

import glob2
import os
import argparse
import ntpath
import cv2
from tqdm import tqdm
import numpy as np
from shutil import copyfile


LABEL2COLOR = {
    '0': (0, 0, 255),
    '1': (0, 255, 0),
}


def load_data_from_dir(path):
    img_paths = glob2.glob(os.path.join(path, '*.jpg'))
    annotation_paths = glob2.glob(os.path.join(path, '*.txt'))
    print("The number of images: ", len(img_paths))
    print("The number of groundtruth files: ", len(annotation_paths))
    return img_paths, annotation_paths
     

def read_data_gt(path):
    class_ids, bboxes = [], []
    with open(path, 'r') as f:
        data = f.readlines()
    for line in data:
        if line.startswith("#"):
            continue
        class_id = line.rstrip().split()[-1]
        bbox = [float(value) for value in line.rstrip().split()[0: -1]]
        class_ids.append(class_id)
        bboxes.append(bbox)

    return class_ids, (np.array(bboxes)/100).round(4)


def convert_to_structure(img_paths, annotation_paths, data_dir, output_dir):
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_folder_dir = os.path.join(output_dir, os.path.split(data_dir)[-1])
    if not os.path.exists(output_folder_dir):
        os.mkdir(output_folder_dir)

    print("[INFO] Converting groundtruth to yolo structure ...")
    print("\tCheck progress at {} directory".format(output_folder_dir))
    for i, img_path in enumerate(tqdm(img_paths)):

        dst_img_path = os.path.join(output_folder_dir, ntpath.basename(img_path))
        copyfile(src=img_path, dst=dst_img_path)

        # Read groundtruth file
        img_file = ntpath.basename(img_path)
        #gt_path = os.path.join(data_dir,img_file.split(".")[0] + '.txt')
        doc_id = img_file.split("-")[0]
        page_id = img_file.split("-")[1].split(".")[0][4:]
        gt_path = os.path.join(data_dir, doc_id + '-color_page' + page_id + '.txt')


        if not os.path.exists(gt_path):
            print("Can't get the {} groundtruth file".format(gt_path))
            continue
        class_ids, bboxes = read_data_gt(path=gt_path)

        out_gt_path = os.path.join(output_folder_dir, img_file.split(".")[0]+'.txt')        
        # print("Writing to file {} ...".format(out_gt_path))
        with open(out_gt_path, 'w') as f:
            for i, (class_id, bbox) in enumerate(zip(class_ids, bboxes)):
                # [x_center, y_center, width, height]
                bbox = [bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2, bbox[2], bbox[3]]
                bboxes[i] = bbox
                # Write to annotation output file
                line = "{} {} {} {} {}".format(class_id, bbox[0], bbox[1], bbox[2], bbox[3])
                f.write(line + '\n')
        
  
def main(args):
    
    # Read all image paths and annotation_paths
    img_paths, annotation_paths = load_data_from_dir(path=args['data_dir'])

    # Convert to structure ScanSSD
    convert_to_structure(img_paths=img_paths, annotation_paths=annotation_paths, data_dir=args['data_dir'], output_dir=args['output_dir'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert dataset MFD2021 to input format of ScanSSD')
    parser.add_argument('--data_dir', default='data/Tr00/',
                        type=str, help='The path container all images and groundtruths.')
    parser.add_argument('--output_dir', default='data/formated_yolov5/', 
                        type=str, help='The base directory to store converted format data')
    args = vars(parser.parse_args())

    main(args)