from glob import glob 
input_path = "/storageStudents/Datasets/ICDAR2021_MDF/test"
output = "/storageStudents/K2018/tiendv/tiennv/mfd_2021/OCR/tesseract/imgs_lst_Ts00_01.txt"

output_file = open(output, "w")

for img_path in glob(input_path + "/*"):
    print("img path: ", img_path)

    if img_path.split(".")[-1] == "jpg":
        output_file.write(img_path+"\n")

output_file.close()