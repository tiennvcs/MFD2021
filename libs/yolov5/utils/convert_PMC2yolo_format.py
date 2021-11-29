import os
import numpy as np
import argparse
import tqdm
import glob2
import xml.etree.ElementTree as ET 



def read_data_from_dir(path):

    """
    The path should container 2 subfolders: images and annotaions

    -path
    |-------images
    |------------|image1.jpg
    |------------|image2.jpg
    |------------|......

    |-------annotations
    |------------|image1.xml
    |------------|image2.xml
    |------------|......

    """
    image_ids = None
    annotations = None 

    base_image_folder = os.path.join(path, 'images/')
    base_annotation_folder = os.path.join(path, 'annotations/')

    if not os.path.exists(base_image_folder):
        print("Invalid image folder. Can't not find the directory {}".format(base_image_folder))
        exit(0)

    if not os.path.exists(base_annotation_folder):
        print("Invalid annotation folder. Can't not find the directory {}".format(base_annotation_folder))
        exit(0)

    # Read image ids
    image_ids = sorted(glob2.glob(os.path.join(base_image_folder, '*.jpg')))

    # Read annotation files
    annotations = sorted(glob2.glob(os.path.join(base_annotation_folder, '*.xml')))

    return image_ids,  annotations


def convert_data2yolo_format(image_ids, annotations, output_dir):
    
    for i, annotation_file in enumerate(annotations):
        tree = ET.parse(annotation_file)
        root = tree.getroot() 
        references = root.findall("*/name")
        print(references)
        for reference in references:
            print(reference.text)
        
        break



def main(args):
    
    # Load the image ids and annotations files
    image_ids, annotations = read_data_from_dir(args['input_path'])
    print("The number of images: {}".format(len(image_ids)))
    print("The number of annotation files: {}".format(len(annotations)))
    
    # Convert to yolo format
    convert_data2yolo_format(image_ids=image_ids, annotations=annotations, output_dir=args['output_path'])


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert the PMC annotation dataset format to Yolo annotation format.')
    parser.add_argument('--input_path', default='../detectron2/PMC_dataset/', 
        help='The directory path container images and annotations folder.')
    parser.add_argument('--output_path', default='convert_dataset/PCM_dataset/',
        help='The output folder after convert original dataset to yolo format.')

    args = vars(parser.parse_args())
    print(args)

    main(args)