import os
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
# yolo_v5
# ├── my_data
# │   ├── images
# │   │   ├── train
# │   │   └── val
# │   └── labels
# │       ├── train
# │       └── val
#

def creat_data_dir():
    work_path = os.getcwd()
    Annotations = work_path + '/data/Annotations'
    ImageSets = work_path + '/data/ImageSets'
    JPEGImages = work_path + '/data/JPEGImages'
    if not os.path.exists(Annotations):
        os.makedirs(Annotations)
        os.makedirs(ImageSets)
        os.makedirs(JPEGImages)


def data_split(data_path,train,label):
    data_name = os.listdir(data_path)

    train_name=[]
    label_name=[]
    for i in data_name:
        name, text = os.path.splitext(i)
        if text ==".jpg":
            train.append(i)
            train_name.append(name)
        else:
            label.append(i)
            label_name.append(name)
    if len(train)!=len(label):
        for i in range(len(train)):
            if train_name[i] not  in label_name:
                train.remove(train[i])
        for i in range(len(label)):
            if label_name[i] not  in train_name:
                label.remove(label[i])

    for i,j in zip(train,label):
        shutil.move(data_path+'/'+i,os.path.join(data_path,"JPEGImages"))
        shutil.move(data_path+'/'+j,os.path.join(data_path,"Annotations"))
    return train,label

def xml_yolo(path,name):
    classes=['face',"face_mask",'hat','person',"smoke"]
    file_name=name+'.txt'
    with open("/home/iceberg/下载/yolov5-master/my_dataset/data/label/"+file_name,'w') as f:
        #获取到整棵树
        root=ET.parse(path).getroot()
        print(path)
        object=root.findall('object')
        #宽高
        width=int(root.find('size').find('width').text)
        height=int(root.find('size').find('height').text)
        for i in object:
            cla=str(i.findtext('name'))
            Bbox=i.find('bndbox')
            #左上角  右下角
            x1=float(Bbox.find('xmin').text)
            y1=float(Bbox.find('ymin').text)
            x2=float(Bbox.find('xmax').text)
            y2=float(Bbox.find('ymax').text)
            #中心横坐标与图像宽度比值
            x_1=((x1+x2)/2.0)/width
            #中心点纵坐标于图像高度比值
            y_1=((y1+y2)/2.0)/height
            #bbox宽度与图像宽度比值
            w_1=(x2-x1)/width
            #bbox高度与图像高度比值
            h_1=(y2-y1)/height
            #类别
            cla=classes.index(cla)
            yolo=str(cla)+'   '+str(x_1)+'   '+str(y_1)+'   '+str(w_1)+'   '+str(h_1)
            f.write(yolo)
            f.write('\r\n')
def split_train_val_test(Path):
    path=Path+"/JPEGImages/"
    file=os.listdir(path)
    a=0
    with open(Path+'/train.txt','w') as f1:
        with open(Path + '/test.txt', 'w') as f2:
            with open(Path + '/val.txt', 'w') as f3:
                for i in range(len(file)):
                    if a<= int(len(file)*0.8):
                        f1.write(path+file[i])
                        f1.write('\r\n')
                    if int(len(file)*0.8)<a<=int(len(file)*0.9):
                        f2.write(path + file[i])
                        f2.write('\r\n')
                    if int(len(file)*0.9)<a<=int(len(file)):
                        f3.write(path + file[i])
                        f3.write('\r\n')
                    a+=1

def Label(xml_path):#data
    xml_file=os.listdir(os.path.join(xml_path,"Annotations"))
    for i in xml_file:
        name,text=os.path.splitext(i)
        if text==".txt":
            shutil.copy(xml_path+"/Annotations/"+i,xml_path+'/label')
        else:
            xml_yolo(xml_path+"/Annotations/"+i,name)

def del_txt(path):
    Path=path+"/Annotations"
    file=os.listdir(Path)
    for i in file:
        name,txt=os.path.splitext(i)
        if txt==".txt":
            os.remove(Path+'/'+i)
            os.remove("/home/iceberg/下载/yolov5-master/my_dataset/data/JPEGImages/"+name+'.jpg')
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--path", type=str, default="data/")

    img_path='/home/iceberg/下载/data'
    label_path="/home/iceberg/下载/data"
    data_save_path="/home/iceberg/下载/yolov5-master/my_dataset/data"

    train = os.listdir(os.path.join(data_save_path,"JPEGImages"))
    label = os.listdir(os.path.join(data_save_path,"Annotations"))
    creat_data_dir()

    #train,label=data_split(img_path,train,label)
    #Label(data_save_path)
    split_train_val_test(data_save_path)
    #del_txt(data_save_path)


