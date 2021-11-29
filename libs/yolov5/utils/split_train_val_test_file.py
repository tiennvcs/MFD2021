import glob2
import os
import argparse
import numpy as np
from convert_yolo_format import load_data_from_dir


def split_train_val_test_file(data_dir, ratio):

    if not os.path.exists(data_dir):
        print("INVALID data directory")
        exit(0)

    img_paths, _ = load_data_from_dir(path=data_dir)

    np.random.seed(18521489)
    val_files = np.random.choice(img_paths, size=int(len(img_paths)*ratio))
    train_files = list(set(img_paths) - set(val_files))
    print("The number of training images: ", len(train_files))
    print("The number of validataion imaes: ", len(val_files))
    train_path = os.path.join(data_dir, 'training_data.txt')
    val_path = os.path.join(data_dir, 'validation_data.txt')

    with open(train_path, 'w') as f:
        for img_path in train_files:
            f.write(img_path+'\n')

    with open(val_path, 'w') as f:
        for img_path in val_files:
            f.write(img_path+'\n')

    print("Split DONE ! Check at {} and {}".format(train_path, val_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split data from directory into training and validation files')
    parser.add_argument('--data_dir', default='./converted_dataset/Tr00/',
                        type=str, help='The path container all images and groundtruths.')
    parser.add_argument('--ratio', default=0.2, type=float,
                        help='The ratio between number of validataion files and all data')
    args = vars(parser.parse_args())

    split_train_val_test_file(data_dir=args['data_dir'], ratio=args['ratio'])