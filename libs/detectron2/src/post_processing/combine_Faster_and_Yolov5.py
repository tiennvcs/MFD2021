import argparse 

def check_IoU(bbox1, bbox2):

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

def combine(faster_result, yolo_result, output_path, thresold_IoU, thresold_size):

    output_file = open(output_path, "w")

    # read result of faster 
    faster_result = get_result(faster_result)
    # print("Faster result: ", faster_result)

    # read result of yolov5
    yolov5_result = get_result(yolo_result)

    imgs_list = faster_result.keys()
    for img in imgs_list:
        print("Processing img: ", img)  
        # check key 
        if not img in yolov5_result.keys():
            print("\tSkip image: ", img)
            continue
        else:
            # print("yolo: ", yolov5_result[img])
            # print("faster: ",faster_result[img] )

            for faster_bbox in faster_result[img]:

                flag = 1
                for yolo_bbox in yolov5_result[img]:
                    if check_IoU(faster_bbox, yolo_bbox) >= thresold_IoU:
                        flag = 0
                        break 
                if flag == 1: 
                    ratio = (faster_bbox[2] - faster_bbox[0]) / (faster_bbox[3] - faster_bbox[1])
                    if  ratio > thresold_size:
                        accept_bbox = "{},{},{},{},{},{},{}\n".format(img, str(faster_bbox[0]), str(faster_bbox[1]), str(faster_bbox[2]), str(faster_bbox[3]), str(faster_bbox[4]), str(faster_bbox[5]))
                        output_file.write(accept_bbox)
                        print("\tAccepted bbox: {} ".format( accept_bbox) )

    # write result of yolov5 into combine file
    yolov5_file = open(yolo_result,'r')

    yolov5_data = yolov5_file.read()
    output_file.write(yolov5_data)
    yolov5_file.close()
    output_file.close()

def main(args):
    print("Hello Minh Gioi Thieu Nganh")
    combine(args.input_faster, args.input_yolo, args.output, args.thresold_IoU, args.thresold_size)

def args_parse():
    parser = argparse.ArgumentParser(description="This argument combine result of Faster and Yolov5  ")
    parser.add_argument('-if', '--input_faster',  default="/storageStudents/K2018/tiendv/tiennv/mfd_2021/detectron2/TOOL_EVALUATION/Predict_folder/Ts00+01_cfg2submission.csv",
                        help="The path result of Faster RCNN")
    parser.add_argument('-iy', '--input_yolo', default = "/storageStudents/K2018/tiendv/tiennv/mfd_2021/yolov5/runs/submission/yolov5x_test/submission.txt",
                        help = "This path result of Yolov5 ")
    parser.add_argument('-o', '--output', default = "./combine_faster_yolo.csv",
                        help = "This path combine result between Faster and Yolov5")
    parser.add_argument('-t1', '--thresold_IoU', default = 0.3, type = float,
                        help = "Thresold to check IoU")
    parser.add_argument('-t2', '--thresold_size', default = 9, type = float,
                        help = "Thresold to check IoU")
    return parser.parse_args()

if __name__ == "__main__": 
    args = args_parse()
    main(args)


'''
python ./src/post_processing/combine_Faster_and_Yolov5.py \
    --input_faster "/storageStudents/K2018/tiendv/tiennv/mfd_2021/detectron2/TOOL_EVALUATION/Predict_folder/testmodel_cfg2submission.csv" \
    --input_yolo "/storageStudents/K2018/tiendv/tiennv/mfd_2021/yolov5/runs/submission/yolov5x_train-val-test_add_Tr10_eval_test/submission.csv" \
    --output  "./TOOL_EVALUATION/combine_faster(Tr10)_and_yolo(Tr10)_test_24_03.csv" \
    --thresold_IoU 0.3 \
    --thresold_size 10.5  

'''
