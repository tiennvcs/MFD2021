import argparse
import cv2
import os 
import json
from tqdm import tqdm

def convert_annotation(input_folder, output):

    '''
    + input_folder: The path of input_folder where save image and annotation for each image.
    + Output: The path of output that save new annotation.
    '''
    new_annotation = {
        "info":{
            "description": "Mathematical Formula Detection",
            "url": None,
            "version": "0.1.0",
            "year": 2021,
            "contributor": "ICDAR2021",
            "date_created": "2021-02-05-20:46"
        },
        "images": [],
        "annotations": [],
        "categories": [
            {
                  "supercategory": "Embedded",
                  "id": 1,
                  "name": "Embedded"
            },
            {
                  "supercategory": "Isolated",
                  "id": 2,
                  "name": "Isolated"
            },
        ]
    }
    print("Converted")
    j = 0
    for i,file in tqdm(enumerate(os.listdir(input_folder))):
        extend = file.split(".")[-1]
        if extend == "jpg":
            img_path = os.path.join(input_folder,file)
            img = cv2.imread(img_path)
            height_img, width_img, _ = img.shape
            # Processing for info image
            print("Processing image {}".format(file))
            annot_image = {}
            annot_image["file_name"] = file
            annot_image["height"] = height_img
            annot_image['width'] = width_img
            annot_image['id'] = i
            annot_image['street_id'] = i
            new_annotation['images'].append(annot_image)
            # print("file: ", file)
            # prefixes, suffixes = file.split(".")[0].split("-")
             
            # name_file = prefixes + "-color_" + suffixes + ".txt"
            name_file = file.split(".")[0] + ".txt"
            annot_path = os.path.join(input_folder, name_file)
            print("file annotation: ", annot_path)
            file_annot = open(annot_path, "r")

            annot_data = file_annot.read().split("\n")
            for line in annot_data[4:-1]:
                x_top_left, y_top_left, width, height, label = line.split("\t")     
                x_top_left, y_top_left, width, height, label =  \
                    (float(x_top_left.strip(" ")), float(y_top_left.strip(" ")), float(width.strip(" ")), float(height.strip(" ")), int(label.strip(" ")))
                # print("Before bbox and label: ", x_top_left, y_top_left, width, height,label ) 
                x_top_left = x_top_left*width_img /100
                y_top_left = y_top_left*height_img /100
                # x_bottom_right =  x_top_left + width*1477 /100
                # y_bottom_right =  y_top_left + height*2048 /100
                width =   width*width_img /100
                height =   height*height_img /100

                # print("After bbox: ",x_top_left, y_top_left, width, height )       
                annotation = {}
                annotation['segmentation'] = []
                annotation['iscrowd'] = 0 
                annotation['area'] = int((width*width_img /100) *  (height*height_img /100))
                annotation['image_id'] = i
                annotation["bbox"] = [
                    int(x_top_left),
                    int(y_top_left),
                    int(width),
                    int(height)
                ]
                annotation['category_id'] = label + 1
                annotation['id'] = j
                new_annotation['annotations'].append(annotation)
                j += 1
    new_data = json.dumps(new_annotation, indent = 4)
    print("new annotation: ", new_data)
    print("output: ", output)
    output_file = open(output, "w")
    output_file.write(new_data)
    print("DONE")

def main(args):
    # Read file txt
    convert_annotation(args.input, args.output)
    # Write new file 
   
def args_parse():
    parser = argparse.ArgumentParser(description="This argument for convert .txt to .json")
    parser.add_argument('-i', '--input',  default="./Test",
                        help="This folder path of image and annotation")
    parser.add_argument('-o', '--output', default = "./annotation.json")

    return parser.parse_args()
if __name__ == "__main__":
    args = args_parse()
    main(args)



'''
python convert_format_annot.py \
    --input "/storageStudents/Datasets/ICDAR2021_MDF/Tr10" \
    --output "/storageStudents/K2018/tiendv/tiennv/mfd_2021/detectron2/Annotation/Tr10/annotation.json"
'''