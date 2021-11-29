import argparse
import os
import cv2

def get_result(result_path):
    result = {}
    # Read result
    fi = open(result_path, "r")
    data =  fi.readlines()
    # print("Data of faster: ", data)
    for line in data: 
        key = line.split(",")[0]
        value = [ float(x) for x in line.split(",")[1:7]]
        if not key in result.keys():
            result[key] = [value]
        else:
            result[key].append(value) 
    fi.close()
    return result

def visualize(imgs_path, sub_path, output_path):
    # Read bbox from submission 
    result_sub = get_result(sub_path)
    for img_name in result_sub.keys():
        print("Visualizing images: ", img_name)
        # Read image 
        img_path = os.path.join(imgs_path, img_name)
        img = cv2.imread(img_path)
        # Draw bboxs to image
        for bbox in result_sub[img_name]:
            if bbox[5] == 0:
                img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (100, 184, 255), 2)
            else: 
                img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (239, 184, 100), 2)
        draw_img_path = os.path.join(output_path, img_name)
        cv2.imwrite(draw_img_path, img)
        print("\tDrawn image: ",draw_img_path )
        
def main(args):
    visualize(args.imgs_path, args.sub_path, args.output_path) 

def args_parser():
    parser = argparse.ArgumentParser(description="This script help to visualize  ")
    parser.add_argument('-im', '--imgs_path',  default="./imgs",
                        help="The path result of Faster RCNN")
    parser.add_argument('-sp', '--sub_path', default = "./submission.csv",
                        help = "This path result of submission")
    parser.add_argument('-o', '--output_path', default = "./Visualize/",
                        help = "This path result of folder save visualize image")
    return parser.parse_args()

if __name__ == "__main__":
    args = args_parser()
    print("args: ", args)
    main(args)

