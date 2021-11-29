import argparse

def intersection(bbox1, bbox2):

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
    min_bbox = min(((bbox2[2] - bbox2[0])*(bbox2[3] -bbox2[1])), ((bbox1[2] - bbox1[0])*(bbox1[3] -bbox1[1])))
    return intersection / min_bbox

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

def filter_bboxs(input_path, output_path, threshold_intersection):
    # Read result from input 
    result = get_result(input_path)
    # print("Result: ", result)
    output_file = open(output_path,'w')
    # iter each images 
    for img_name in result.keys():
        print("Processing: ", img_name)
        bboxs = result[img_name]
        # add flag 
        [bbox.append(1) for bbox in bboxs]
        # print("Before sort:\n",bboxs)
        # Sort bboxs of each image with coordinates
        sort_bboxs = sorted(bboxs, key=lambda bboxs: bboxs[1] )
        num_bbox = len(sort_bboxs)
        # print("Sort:\n", sort_bboxs)
        for i in range(0, num_bbox-1):
            if sort_bboxs[i][-1] != 1:
                continue
            else: 
                for j in range(i+1,num_bbox):
                    if sort_bboxs[j][-1] != 1: 
                        continue
                    else:
                        if intersection(sort_bboxs[i][:4], sort_bboxs[j][:4]) > threshold_intersection:
                            #remove bbox 
                            sort_bboxs[j][-1] = -1
                            print("\tRemove bbox: ", sort_bboxs[j])

                sort_bboxs[i][-1] = 0 
        
        # Write bboxs after filter
        for bbox in sort_bboxs:
            if bbox[-1] != -1:
                output_file.write("{},{},{},{},{},{},{}\n".format(img_name, str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3]),str(bbox[4]), str(bbox[5])))
    output_file.close()

def main(args):
    print("Hello world")
    filter_bboxs(args.input_path, args.output_path, args.threshold)

def args_parser():
    parser = argparse.ArgumentParser(description="This argument of bbox filter   ")
    parser.add_argument('-i', '--input_path',  default="/storageStudents/K2018/tiendv/tiennv/mfd_2021/detectron2/TOOL_EVALUATION/Predict_folder/Ts00+01_cfg2submission.csv",
                        help="The path result after combine and add embedded references")
    parser.add_argument('-o', '--output_path', default="./filter_bbox.csv",
                        help="The path result after filter bboxs")
    parser.add_argument('-th', '--threshold', default=0.5, type=float,
                        help="Thresold intersection. ")

    return parser.parse_args()

if __name__ == "__main__":
    args = args_parser()
    main(args)

'''
python ./src/post_processing/filter_bboxs.py \
    --input_path "/storageStudents/K2018/tiendv/tiennv/mfd_2021/detectron2/TOOL_EVALUATION/combine_faster_and_yolo_Ts01+00.csv" \
    --output_path "./TOOL_EVALUATION/filter_bbox_Ts00_01.csv" \
    --threshold 0.7
'''
