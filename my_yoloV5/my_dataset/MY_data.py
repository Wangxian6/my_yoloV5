import os
import shutil

img_path='/home/iceberg/下载/yolov5-master/my_dataset/data/JPEGImages'
xml_path="/home/iceberg/下载/yolov5-master/my_dataset/data/label"
img=os.listdir(img_path)
xml=os.listdir(xml_path)
X=[]
for i in xml:
    name,txt=os.path.splitext(i)
    X.append(name)
a=0
for i in img:
    name,txt=os.path.splitext(i)
    if X.count(name):
        if a <= int(len(img) * 0.8):
            shutil.copy(img_path+"/"+i,'/home/iceberg/下载/V5_data/train/images')
            shutil.copy(xml_path+"/"+name+'.txt','/home/iceberg/下载/V5_data/train/labels')
        if int(len(img) * 0.8) < a <= int(len(img) * 0.9):
            shutil.copy(img_path + "/" + i, '/home/iceberg/下载/V5_data/test/images')
            shutil.copy(xml_path + "/" + name + '.txt', '/home/iceberg/下载/V5_data/test/labels')
        if int(len(img) * 0.9) < a <= int(len(img)):
            shutil.copy(img_path + "/" + i, '/home/iceberg/下载/V5_data/valid/images')
            shutil.copy(xml_path + "/" + name + '.txt', '/home/iceberg/下载/V5_data/valid/labels')
        a+=1