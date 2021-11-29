"""
Usage:

python get_reference_embedded.py \
    --img_lst ./img_lst.txt \
    --output_dir ./output
    --verbose True

"""

import re
import os
import cv2
import glob2
import argparse
import pytesseract
import numpy as np
from tqdm import tqdm
import re
from constrains import check_embedded


def get_reference_embbeded(path, output_dir, verbose=False):
    img = cv2.imread(path)
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Extract information from the image
    extract_data = pytesseract.image_to_data(bw_img, output_type=pytesseract.Output.DICT)

    # Get the object recognized is page_num, block_num, par_num, line_num, word_num
    # page_num_indices = np.where(np.array(extract_data['level'])==1)[0]
    # block_num_indices = np.where(np.array(extract_data['level'])==2)[0]
    # par_num_indices = np.where(np.array(extract_data['level'])==3)[0]
    # line_num_indices = np.where(np.array(extract_data['level'])==4)[0]
    word_num_indices = np.where(np.array(extract_data['level'])==5)[0]

    # Get the object have specific patterns
    # Case 1
    filter_word_num_indices_case1 = [i for i in word_num_indices
                                    if re.match("[A-Z0-9]\-[A-Za-z]{4,10}", extract_data['text'][i])]
    # # Case 2
    filter_word_num_indices_case2 = [i for i in word_num_indices
                                    if re.match("\([A-Z]\.[0-9]", extract_data['text'][i])]
    bw_img = bw_img[:, :, 0]
    filter_word_num_indices_case2 = [
        i for i in filter_word_num_indices_case2
            if check_embedded(bin_img=bw_img,
                            bbox=(extract_data['left'][i], 
                                  extract_data['top'][i], 
                                  extract_data['width'][i], 
                                  extract_data['height'][i]
                            )
            )
    ]

    # # Case 3
    filter_word_num_indices_case3 = [i for i in word_num_indices
                                    if re.match("[\-+]{0,1}[0-9]{1,2}[+\-\*/><=][0-9]{1,2}", extract_data['text'][i])]
    
    # # Case 4
    filter_word_num_indices_case4 = [i for i in word_num_indices
                                    if re.match("exp\(|dim\(|rank\(|cos\(|sin\(|tan\(|cotan\(|Ker\(|Im\(|LSM\(", extract_data['text'][i])]

    # Case 5
    filter_word_num_indices_case5 = [i for i in word_num_indices
                                    if re.match("[A-Z]+\(.", extract_data['text'][i])]


    if len(filter_word_num_indices_case1) + len(filter_word_num_indices_case2) + len(filter_word_num_indices_case3) + len(filter_word_num_indices_case4) + len(filter_word_num_indices_case5) == 0:
        return
    

    # Write detection result to file
    with open(os.path.join(output_dir, 'case1.txt'), 'a+') as f:
        for i in filter_word_num_indices_case1:
            line = "{}, {}, {}, {}, {}, {}, {}".format(
                os.path.basename(path),
                extract_data['left'][i], 
                extract_data['top'][i], 
                extract_data['left'][i]+extract_data['width'][i], 
                extract_data['top'][i]+extract_data['height'][i],
                extract_data['conf'][i]/100, 0
            )
            f.write(line + '\n')
    with open(os.path.join(output_dir, 'case2.txt'), 'a+') as f:
        for i in filter_word_num_indices_case2:
            line = "{}, {}, {}, {}, {}, {}, {}".format(
                os.path.basename(path),
                extract_data['left'][i], 
                extract_data['top'][i], 
                extract_data['left'][i]+extract_data['width'][i], 
                extract_data['top'][i]+extract_data['height'][i],
                extract_data['conf'][i]/100, 0
            )
            f.write(line + '\n')
    with open(os.path.join(output_dir, 'case3.txt'), 'a+') as f:
        for i in filter_word_num_indices_case3:
            line = "{}, {}, {}, {}, {}, {}, {}".format(
                os.path.basename(path),
                extract_data['left'][i], 
                extract_data['top'][i], 
                extract_data['left'][i]+extract_data['width'][i], 
                extract_data['top'][i]+extract_data['height'][i],
                extract_data['conf'][i]/100, 0
            )
            f.write(line + '\n')
    with open(os.path.join(output_dir, 'case4.txt'), 'a+') as f:
        for i in filter_word_num_indices_case4:
            line = "{}, {}, {}, {}, {}, {}, {}".format(
                os.path.basename(path),
                extract_data['left'][i], 
                extract_data['top'][i], 
                extract_data['left'][i]+extract_data['width'][i], 
                extract_data['top'][i]+extract_data['height'][i],
                extract_data['conf'][i]/100, 0
            )
            f.write(line + '\n')
    with open(os.path.join(output_dir, 'case5.txt'), 'a+') as f:
        for i in filter_word_num_indices_case5:
            line = "{}, {}, {}, {}, {}, {}, {}".format(
                os.path.basename(path),
                extract_data['left'][i], 
                extract_data['top'][i], 
                extract_data['left'][i]+extract_data['width'][i], 
                extract_data['top'][i]+extract_data['height'][i],
                extract_data['conf'][i]/100, 0
            )
            f.write(line + '\n')

    if verbose:
        # Draw case 1
        for i in filter_word_num_indices_case1:
            (x, y, w, h) = (extract_data['left'][i], 
                            extract_data['top'][i], 
                            extract_data['width'][i], 
                            extract_data['height'][i])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, '1', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


        # Draw case 2
        for i in filter_word_num_indices_case2:
            (x, y, w, h) = (extract_data['left'][i], 
                            extract_data['top'][i], 
                            extract_data['width'][i], 
                            extract_data['height'][i])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 127, 255), 2)
            cv2.putText(img, '2', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 127, 255), 2)

        # Draw case 3
        for i in filter_word_num_indices_case3:
            (x, y, w, h) = (extract_data['left'][i], 
                            extract_data['top'][i], 
                            extract_data['width'][i], 
                            extract_data['height'][i])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(img, '3', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Draw case 4
        for i in filter_word_num_indices_case4:
            (x, y, w, h) = (extract_data['left'][i], 
                            extract_data['top'][i], 
                            extract_data['width'][i], 
                            extract_data['height'][i])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (127, 0, 127), 2)
            cv2.putText(img, '4', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (127, 0, 127), 2)

        # Draw case 5
        for i in filter_word_num_indices_case5:
            (x, y, w, h) = (extract_data['left'][i], 
                            extract_data['top'][i], 
                            extract_data['width'][i], 
                            extract_data['height'][i])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, '5', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        save_bin_img_file = os.path.join(output_dir, 'bin_images', os.path.basename(path))
        save_img_file = os.path.join(output_dir, 'images', os.path.basename(path))
        #save_text_file = os.path.join(output_dir, 'text', os.path.basename(path).split(".")[0]+'.txt')
        
        cv2.imwrite(save_bin_img_file, bw_img)
        cv2.imwrite(save_img_file, img)
        # with open(save_text_file, 'w') as f:
        #     for i in filter_word_num_indices_case1:
        #         f.write(extract_data['text'][i]+'\n')
        #     for i in filter_word_num_indices_case2:
        #         f.write(extract_data['text'][i]+'\n')
        #     for i in filter_word_num_indices_case3:
        #         f.write(extract_data['text'][i]+'\n')
        #     for i in filter_word_num_indices_case4:
        #         f.write(extract_data['text'][i]+'\n')
        #     for i in filter_word_num_indices_case5:
        #         f.write(extract_data['text'][i]+'\n')
        #     pass


def main(args):
    
    # Check invalid input or not
    if not os.path.exists(args['img_lst']):
        print("INVALID IMAGE LIST FILE !")
        exit(0)
    if not os.path.exists(args['output_dir']):
        print("Creating output directory: {}".format(args['output_dir']))
        try:
            os.mkdir(args['output_dir'])
        except:
            print("INVALID OUTPUT DIRECTORY !")
            exit(0)
        os.mkdir(os.path.join(args['output_dir'], 'images'))
        os.mkdir(os.path.join(args['output_dir'], 'bin_images'))
        os.mkdir(os.path.join(args['output_dir'], 'text'))

    if not os.path.exists(os.path.join(args['output_dir'], 'bin_images')):
        os.mkdir(os.path.join(args['output_dir'], 'bin_images'))
    if not os.path.exists(os.path.join(args['output_dir'], 'images')):
        os.mkdir(os.path.join(args['output_dir'], 'images'))
    if not os.path.exists(os.path.join(args['output_dir'], 'text')):
        os.mkdir(os.path.join(args['output_dir'], 'text'))

    # Get the existing image path from file
    with open(args['img_lst'], 'r') as f:
        img_paths = [line.rstrip() for line in f.readlines() if os.path.exists(line.rstrip())]

    # Extract information from each image
    for i, img_path in enumerate(img_paths):
        print("{:6}/{:6} Extracting {}".format(
            str(i).zfill(6), str(len(img_paths)).zfill(6), os.path.basename(img_path)))
        get_reference_embbeded(path=img_path, output_dir=args['output_dir'], verbose=args['verbose'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract information from image using Tesseract OCR engine.')
    parser.add_argument('--img_lst', required=True,
        help='an integer for the accumulator')
    parser.add_argument('--output_dir', required=True,
        help='The output directory to store extracted information')
    parser.add_argument('--verbose', action='store_true',
        help='Store image to disk wheter or not.')

    args = vars(parser.parse_args())

    print(args)

    main(args)

