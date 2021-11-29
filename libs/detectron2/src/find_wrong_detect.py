import argparse
import os
import cv2

# This script will find all bbox difference between predicted result and ground truth 

def compute_IoU(bbox1, bbox2):
    """
    This function calculate IoU between bbox1 and bbox2
    ===========
    Parameters
        + bbox1: 
            + Type: list
            + Des: Info bbox of ground truth. 
            + Value: [x1, y1, x2, y2, class_name]
        + bbox2:
            + Type: list
            + Des: Info bbox of predicted result.
            + Value: [x1, y1, x2, y2, class_name]

    ===========
    Return: float 
        Percentage overlap between bbox1 and bbox2 in [0,1]

    """

    assert bbox1[0] < bbox1[2]
    assert bbox1[1] < bbox1[3]
    assert bbox2[0] < bbox2[2]
    assert bbox2[1] < bbox2[3]

    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox1[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        # print('x_right {}, x_left: {}, y_bottom: {}, y_top: {}'.format(x_right, x_left ,y_bottom, y_top))
        return 0.0

    intersection = (x_right - x_left)*(y_bottom - y_top)
    # print("intersection: ", intersection)
    union = ((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])) + ((bbox2[2] - bbox2[0])*(bbox2[3] -bbox2[1])) - intersection
    # print("union: ", union)
    IoU = intersection / union 
    return IoU 

def read_predict_bbox(path):

    predict_bboxs = []
    bbox_file = open(path, "r")
    data = bbox_file.read()
    # print('data:\n',data)
    list_bbox = data.split("\n")
    # print("list bbox: ", list_bbox)
    for i in range(0,len(list_bbox)-1):
        # print("bbox: ", list_bbox[i].split(","))
        real_bbox = [int(e) for e in list_bbox[i].split(",")]
        # print("real bbox: ", real_bbox)
        predict_bboxs.append([real_bbox,0])
    return predict_bboxs
    
def read_ground_truth(path, width, height): 
    # print('width {}, height {}'.format(width, height))
    gt_bboxs = []
    gt_file = open(path, "r")
    data = gt_file.read()
    # print("data: ", data)
    list_bbox = data.split("\n")
    # print("list data: ", list_bbox)
    for i in range(4,len(list_bbox)-1):
        # print("bbox: ", list_bbox[i].split("\t"))
        x1, y1, w, h, cls_name = [float(e.strip(" ")) for e in list_bbox[i].split("\t") ]
        # print("bbox: ", x1, y1, w, h, cls_name )
        real_x1 = int(x1*width/100)
        real_y1 = int(y1*height/100)
        real_w = w*width/100
        real_x2 = int(real_x1 + real_w)
        real_h = h*height/100
        real_y2 = int(real_y1 + real_h)
        # print("real bbox: ", real_x1, real_y1, real_x1, real_y2, int(cls_name))
        gt_bboxs.append([[ real_x1, real_y1, real_x2, real_y2, int(cls_name)],0])
    return gt_bboxs


def find_bbox_difference(predict_path,ground_truth_path, output_path, thresold):
    predicts_bbox_path = os.path.join(predict_path,"results")
    dic = {}
    for bbox_file_name in os.listdir(predicts_bbox_path):

        # Get width and height of images 
        img_path = os.path.join(predict_path,"imgs",bbox_file_name.split(".")[0]+".jpg")
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        print(width, height)
        # Read bboxs from predict file into dictionary 
        
        # print("bbox file: ", bbox_file_name)
        bbox_file_path = os.path.join(predicts_bbox_path,bbox_file_name)
        predict_bboxs = read_predict_bbox(bbox_file_path)
        # print("predict bboxs: ",predict_bboxs )
            
        # Read bboxs from ground truth file into dictionary
        tmp1, tmp2 =  bbox_file_name.split("-")
        gt_file_name =tmp1 + "-color_" + tmp2
        gt_file_path = os.path.join(ground_truth_path,gt_file_name)
        gt_bboxs = read_ground_truth(gt_file_path, width, height)
        # print("ground truth bbox: ", gt_bboxs)

        # Find The whole bboxs difference between predict and ground truth 
        for p_bbox in predict_bboxs:
            for gt_bbox in gt_bboxs:
                if gt_bbox != 0:
                    # print("Info: ",p_bbox, gt_bbox )
                    IoU = compute_IoU(p_bbox[0], gt_bbox[0])
                    if IoU < thresold: 
                        continue
                    else:
                        p_bbox[1] = 1
                        gt_bbox[1] = 1
                        # print("Accept: ",p_bbox, gt_bbox)
                else:
                    continue
        # print("predict bboxs: ",predict_bboxs )
        # print("ground truth bbox: ", gt_bboxs)

        diff_predict_bboxs = [k[0]  for k in predict_bboxs if k[1] == 0]
        diff_gt_bboxs = [j[0] for j in gt_bboxs if j[1] == 0]

        # print("diff predict bboxs: ",diff_predict_bboxs )
        # print("diff gt bboxs: ", diff_gt_bboxs)
        diff_predict_bboxs, diff_gt_bboxs

        # Draw difference 
        img_name = bbox_file_name.split('.')[0] + '.jpg'
        img1_name = img_name 
        img2_name = "visualize_" + img_name 
        img1_path = os.path.join(predict_path,'imgs',img1_name)
        img2_path = os.path.join(ground_truth_path, img2_name)
        print("img1{} \nimg2{}: ".format(img1_path, img2_path))
        draw_diff_img(diff_predict_bboxs, diff_gt_bboxs,img1_path, img2_path , output_path)

def draw_diff_img(diff_predict_bboxs, diff_gt_bboxs, img1_path, img2_path, output):
    img_name = img1_path.split("/")[-1]
    print("image name: ", img_name)
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    for bbox in diff_predict_bboxs:
        img1 = cv2.rectangle(img1, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2)
    for bbox in diff_gt_bboxs:
        img2 = cv2.rectangle(img2, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2)
    
    img1_output = os.path.join(output,'Predict','predict' + img_name )
    img2_output = os.path.join(output, "Ground_truth",'gt' + img_name )
    cv2.imwrite(img1_output, img1)
    cv2.imwrite(img2_output, img2)
    print("Saved: ", img1_output)
    print("Saved: ", img2_output)



def main(args):
    find_bbox_difference(args.predict_path, args.ground_truth_path,args.output_path, args.thresold) 

def args_parse():
    parser = argparse.ArgumentParser(description="This argument wrong detect ")
    parser.add_argument('-p', '--predict_path',  default="./Test",
                        help="This folder path of detected image ")
    parser.add_argument('-g', '--ground_truth_path',  default="./Test",
                        help="This folder path of ground truth ")
    parser.add_argument('-o', '--output_path', default = "./DATASET/Test/Test_check_wrong_detect/Diffrence",
                        help = "This path save compare result")
    parser.add_argument('-ts', '--thresold', default = 0.75, type = int,
                        help = "Thresold of IOU ")


    return parser.parse_args()

if __name__ == "__main__":
    args = args_parse()
    main(args)

'''
CMD: 
python ./src/find_wrong_detect.py \
-p "./output/cfg2/Va01" \
-g "./DATASET/visualize_data/Va01" \
-o "./Analysis_Data/Diff/Va01"
'''
    