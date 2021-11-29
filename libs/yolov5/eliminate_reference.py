"""
usage:
    python eliminate_reference.py --detect_result submission.txt --ref_result ref_detection.txt
"""
import os
import argparse
from utils.post_processing import eliminate_reference




def read_data(path):
    with open(path, 'r') as f:
        data = f.readlines()
    return data


def filter_result(math_data, reference_data, output_path):
    
    ref_page_ids = [x.split(",")[0] for x in reference_data]
    ref_lst = dict()
    for line in reference_data:
        key = line.split(",")[0]
        value = [ float(x) for x in line.split(",")[1:5]]
        if not key in ref_lst.keys():
            ref_lst[key] = [value]
        else:
            ref_lst[key].append(value)
    with open(os.path.join(output_path, 'final_submission.txt'), 'w') as f:
        for line in math_data:
            page_id = line.split(",")[0]
            if page_id in ref_lst.keys():
                xyxy_ = [float(x) for x in line.split(",")[1:5]]
                if eliminate_reference(xyxy_, ref_lst[page_id]):
                    print("---> Eliminate detected fomular in {}".format(page_id))
                    continue
            f.write(line)


def main(args):
    
    # Check available paths
    if not os.path.exists(args['detect_result']):
        print("[error] INVALID detection detection result path")
        exit(0)

    if not os.path.exists(args['ref_result']):
        print("[error] INVALID reference detection result path." )
        # exit(0)

    # Read data from 2 files
    math_data = read_data(path=args['detect_result'])
    reference_data = read_data(path=args['ref_result'])

    # Filter detection result
    filter_result(math_data=math_data, reference_data=reference_data, output_path=args['output_path'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge reference detection results with yolo detection results.')
    parser.add_argument('--detect_result', required=True,
                        help='The path of inferenred detection results')
    parser.add_argument('--ref_result', required=True,
                        help='The path of reference detection results')
    parser.add_argument('--output_path', required=True, 
                        help='The output path after filter')
    args = vars(parser.parse_args())

    main(args)