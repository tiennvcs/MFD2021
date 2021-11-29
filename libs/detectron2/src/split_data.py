import argparse
import os
import shutil 
from math import ceil
from glob import glob



def split_data(input_path, outpath, no_sub):
    all_imgs_name = glob(input_path + "/*.jpg")
    num_imgs = len(all_imgs_name)
    space = ceil(num_imgs/no_sub)
    print("Space: ", space)
    
    for i in range(0, num_imgs, space):
        folder_path = outpath + "/split_" + str(i)
        os.mkdir(folder_path)
        low = i 
        if i + space <= num_imgs:
            end = i + space
        else:
            end = num_imgs
        print("{} Split {} {}".format("="*40, folder_path, "="*40))

        for j in range(low, end):
            img_name = all_imgs_name[j].split("/")[-1]
            new_imgs_path = os.path.join(folder_path, img_name)
            shutil.copyfile(all_imgs_name[j], new_imgs_path)
            print("\t Copied image {}".format(all_imgs_name[j]))

def main(args):
    split_data(args.input_path, args.output_path, args.num_sub)

def args_parse():
    parse = argparse.ArgumentParser(description="This argument of split data",)
    parse.add_argument("-i", "--input_path", default = "./DATASET/edited_data/Train_data",
                        help = "The path of data that you want to split")
    parse.add_argument("-o", "--output_path", default = "./DATASET/split_data_train",
                        help = "The path save splited data")
    parse.add_argument("-n", "--num_sub", default = 2, type = int,
                        help = "Number subset, you want to split")
    
    return parse.parse_args()

if __name__ == "__main__":
    args = args_parse()
    print("Argument: \n{}\n{}\n{}".format(args.input_path, args.output_path, args.num_sub))
    main(args)

'''
python ./src/split_data.py \
    -i "./DATASET/edited_data/Train_data" \
    -o "./DATASET/split_data_train" \
    -n 8
'''