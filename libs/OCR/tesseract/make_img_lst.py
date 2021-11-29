import os
import glob2
import argparse
from tqdm import tqdm

def make_img_lst(dir_path, output_file):
    
    img_paths = sorted(glob2.glob(os.path.join(dir_path, '*.jpg')))
    with open(output_file, 'w') as f:
        for img_path in tqdm(img_paths):
            f.write(img_path+'\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract information from image using Tesseract OCR engine.')
    parser.add_argument('--img_dir', required=True,
        help='The directory contain all images')
    parser.add_argument('--output_file', required=True,
        help='The output file contain list of image paths.')
    args = vars(parser.parse_args())
    print(args)

    print("Creating image list file ...")
    make_img_lst(dir_path=args['img_dir'], output_file=args['output_file'])
    print("DONE ! Check at {}".format(args['output_file']))