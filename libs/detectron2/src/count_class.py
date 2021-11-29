import os 
import glob 

path = "/mlcv/WorkingSpace/NCKH/tiennv/mfd_2021/detectron2/DATASET/raw_data/Ts01" 

isolated = 0
embedded = 0 

list_anno = glob.glob(path + "/*.txt")

for file_path in list_anno:
    print(file_path)
    fi = open(file_path, 'r')
    data = fi.readlines()
    # print(data[4:])
    for anno in data[4:]:
        # print(anno)
        label =  anno.split("\t")[-1].strip("\n").strip(" ")
        if label == "1":
            isolated += 1
        elif label == "0":
            embedded += 1

print('path: ', path)
print("isolated: ", isolated)
print("embedded: ", embedded)

