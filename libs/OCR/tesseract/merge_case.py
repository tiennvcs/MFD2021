import argparse
from shapely.geometry import Polygon
def check_IoU_Thuyen(bbox1, bbox2):
    
    assert bbox1[0] < bbox1[2]
    assert bbox1[1] < bbox1[3]
    assert bbox2[0] < bbox2[2]
    assert bbox2[1] < bbox2[3]

    min_area = min(((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])), ((bbox2[2] - bbox2[0])*(bbox2[3] -bbox2[1]))) 
    poly_box1 = Polygon([   (bbox1[0],bbox1[1]) , (bbox1[2],bbox1[1]) , (bbox1[2],bbox1[3]) , (bbox1[0],bbox1[3])  ] )
    poly_box2 = Polygon([   (bbox2[0],bbox2[1]) , (bbox2[2],bbox2[1]) ,(bbox2[2],bbox2[3]) , (bbox2[0],bbox2[3])  ] )

    # print("\tminIoU ... ",poly_box1.intersection(poly_box2).area/min_area)
    intersection = poly_box1.intersection(poly_box2).area

    IoU = intersection / min_area 
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

def merge_case1(input_path, input_case, output_path):
    pass

def merge_case2(input_path, input_case, output_path):
    pass

def merge_case3(input_path, input_case, output_path):
    pass

def merge_case4(input_path, input_case, output_path):
    pass

def merge_case5(input_path, input_case, output_path):
    # Open output file
    output_file = open(output_path, 'w') 

    sub_result = get_result(input_path)
    case_result = get_result(input_case)

    #print("sub_result: {}\ncase_result: {}".format(sub_result, case_result))
    imgs_list = sub_result.keys()
    count = 0
    update = 0
    result = []
    for img in imgs_list:
        print("Processing img: ", img)  
        # check key 
        if not img in case_result.keys():
            print("\tMerge all bbox of image: ", img)
            all_bbox = [[img] + bbox for bbox in sub_result[img] ]
            result = result + all_bbox
        else:
            for case_bbox in case_result[img]:
                flag = 1
                for i, sub_bbox in enumerate(sub_result[img]):
                    min_IoU = check_IoU_Thuyen(case_bbox, sub_bbox)
                    if  min_IoU > 0.05  :
                        if min_IoU  > 0.9 and sub_bbox[-1] != 1:
                            print('\tmin IoU: ', min_IoU)
                            x_min = min(sub_bbox[0], case_bbox[0])
                            y_min = min(sub_bbox[1], case_bbox[1])
                            x_max = max(sub_bbox[2], case_bbox[2])
                            y_max = max(sub_bbox[3], case_bbox[3])
                            new_score = max(sub_bbox[4], case_bbox[4])
                            new_bbox = [x_min, y_min, x_max, y_max,new_score,0]
                            #update new bbox 
                            sub_result[img][i] = new_bbox
                            print("\tUpdated bbox: ", new_bbox)
                            update += 1
                        flag = 0
                        break 
                if flag == 1 :
                    print("\tMerged bbox: ", [img]+case_bbox)
                    # output_file.write("{},{},{},{},{},{}\n".format(case_bbox[0], case_bbox[1], case_bbox[2], case_bbox[3], case_bbox[4], case_bbox[5]))
                    result.append([img]+case_bbox)
                    count += 1
                        
            for k in range(len(sub_result[img])):
                result.append([img]+sub_result[img][k])
                    # output_file.write("{},{},{},{},{},{}\n".format(sub_result[img][k][0], sub_result[img][k][1], sub_result[img][k][2], sub_result[img][k][3], sub_result[img][k][4], sub_result[img][k][5]))
    # Write result
    # print("Result: ", result)
    final_result = ""
    for bbox in result:
        final_result = final_result + "{},{},{},{},{},{},{}\n".format(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], bbox[6])
    output_file.write(final_result)
    print("Number update: {}\nNumber add: {}".format(update, count))
    output_file.close()


def main(args):
    print("Hello world")
    dic_case={
        1:merge_case1,
        2:merge_case2,
        3:merge_case3,
        4:merge_case4,
        5:merge_case5
    }
    dic_case[args.case](args.input_path, args.input_case, args.output)

def args_parse():
    parser = argparse.ArgumentParser(description="This argument of paddle ocr")
    parser.add_argument("-i", "--input_path", default="./submission.csv",
                        help = "The path of submission")
    parser.add_argument("-c", "--case", default=1, type=int,
                        help = "Choose case to merge")
    parser.add_argument("-ic", "--input_case", default="./case1.txt", 
                        help = "The path of case")
    parser.add_argument("-o", "--output", default="merger_case_submission.csv")

    return parser.parse_args()

if __name__ == '__main__':
    args = args_parse()
    main(args)

'''
python merge_case.py \
    --input_path "/storageStudents/K2018/tiendv/tiennv/mfd_2021/detectron2/TOOL_EVALUATION/combine_faster_and_yolo_Ts01+00.csv" \
    --case 5 \
    --input_case "/storageStudents/K2018/tiendv/tiennv/mfd_2021/OCR/tesseract/ocr_Ts00+Ts01/case5.txt" \
    --output "/storageStudents/K2018/tiendv/tiennv/mfd_2021/detectron2/TOOL_EVALUATION/merge_case_5_Ts00+01.csv" 
'''