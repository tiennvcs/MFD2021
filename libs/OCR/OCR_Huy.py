import requests
import json
import matplotlib as plt
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import argparse
import glob
# from visualize import visualize


def output_txt(data_task1, data_task2, output_path):
    bboxs = data_task1['result']
    words = data_task2['result']
    # print(words)
    result_txt_list = ''
    for box, word in zip(bboxs, words):
        result_txt_line = ''
        box = box['bbox']
        word = word['words']
        # print(box)
        # print(word)
        for b in box:
            result_txt_line += str(b) + ' '
            # print(result_txt_line)
        result_txt_line += str(word)
        result_txt_list += result_txt_line + '\n'
    with open(output_path, 'w') as out:
        out.write(result_txt_list)
        print('output_path: ',output_path, '----ok----')


def output_txt_box(data_task1, data_task2, output_path):
    bboxs = data_task1['result']
    # print(words)
    result_txt_list = ''
    for box in bboxs:
        result_txt_line = ''
        box = box['bbox']
        # print(box)
        # print(word)
        for b in box:
            result_txt_line += str(b) + ' '
        result_txt_list += result_txt_line + '\n'
    with open(output_path, 'w') as out:
        out.write(result_txt_list)
        print('output_path: ',output_path, '----ok----')

def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        # point = point.astype(int)
        cv2.line(img_path, tuple(point[0]), tuple(point[1]), color, thickness)
        cv2.line(img_path, tuple(point[1]), tuple(point[2]), color, thickness)
        cv2.line(img_path, tuple(point[2]), tuple(point[3]), color, thickness)
        cv2.line(img_path, tuple(point[3]), tuple(point[0]), color, thickness)
    return img_path

def get_boxs_list(txt_path):
    with open(txt_path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    boxes_list = []
    for ele in content:
        bbox_str = ele.split(" ")[0:8]
        world = ele.split(" ",8)[-1]
        bbox = []
        for i in range(0, 8, 2):
            xs = int(bbox_str[i])
            ys = int(bbox_str[i+1])
            point = [xs, ys]
            bbox.append(point)
            # print('bx:', bbox)
        boxes_list.append(bbox)
    print(boxes_list)
    return boxes_list

def visualize_polygon(img, img_path, txt_path, output):
    boxes_list = get_boxs_list(txt_path)
    img = draw_bbox(cv2.imread(img_path)[:, :, ::-1], boxes_list)
    name_img = img_path.split('/')[-1]
    if not os.path.exists(output):
        os.mkdir(output)
    output_path_file = os.path.join(output, name_img)
    cv2.imwrite(output_path_file, img)


# TASK1_URL = 'http://service.mmlab.uit.edu.vn/receipt/task1/predict'
TASK1_URL = 'http://service.aiclub.cs.uit.edu.vn/gpu150/pannet/predict'
TASK2_URL = 'http://service.mmlab.uit.edu.vn/receipt/task2/predict'
# TASK2_URL = 'http://service.aiclub.cs.uit.edu.vn/gpu150/vietocr/predict'
# OUTPUT = 'output/output_img'

def predict(input_path, output_path):
    fail_file_path = ( output_path + "fail.txt")
    fail_file = open(fail_file_path, "a+")
    fail_file.write("==================================================================================================================\n")

    for img_path in glob.glob(input_path + "/*"):
        img_name = img_path.split('/')[-1]
        print("Img path: ", img_path)
        img = cv2.imread(img_path)
        
        try: 
            detect_task1 = requests.post(TASK1_URL, files={"file": (
                "filename", open(img_path, "rb"), "image/jpeg")}).json()
            print(detect_task1)

            output_txt_file = os.path.join(output_path, img_name.split('.')[0] + '.txt')
            output_txt_box(detect_task1, 'detect_task2', output_txt_file)
            visualize_polygon(img, img_path, output_txt_file, output_path)
        except:
            fail_file.write(img_path + "\n")
        
    fail_file.close()
            


def main(args):
    print("Hello world")
    predict(args.input, args.output)

def args_parse():
    
    parser = argparse.ArgumentParser(description="This argument predict with OCR ")
    parser.add_argument('-i', '--input',  default="../detectron2/DATASET/edited_data/Valid_data",
                        help="This folder path of image that you want predict")
    parser.add_argument('-o', '--output', default = "./output/Valid",
                        help = "This path save predict result")
    return parser.parse_args()

if __name__ == "__main__":
    args = args_parse()
    main(args)

#Cmd

    