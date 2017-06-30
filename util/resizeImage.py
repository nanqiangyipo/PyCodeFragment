from PIL import Image
import os
import shutil
import numpy as np
import sys


# data_path=r'/home/xinye/PycharmProjects/PyCodeFragment/data/animal'
data_path=r'/home/xinye/workingdirectory/PyCodeFragment/data/bot_train'
resized_data_path=r'/home/xinye/PycharmProjects/PyCodeFragment/data/resized_animal'

# 各个图片文件夹的名字
animal_path_name=os.listdir(data_path)

# 图片文件夹全路径
animal_paths = [os.path.join(data_path,filename) for filename in animal_path_name]
print(animal_paths[0:4],end='\n\n')

# 重命名后图片文件夹的全路径
resized_paths=[os.path.join(resized_data_path,filename) for filename in animal_path_name]
for p in resized_paths:
    if not os.path.exists(p):
        os.makedirs(p)
    elif len(os.listdir(p))>0:
        print('删目录')
        shutil.rmtree(p)
        os.makedirs(p)



def getImagePaths(par_path):
    """获取图片文件全路径"""
    return [os.path.join(par_path,filename) for filename in os.listdir(par_path)]

new_size=(213,213)
for i in range(len(animal_paths)):
    image_paths=getImagePaths(animal_paths[i])
    resized_image=[Image.open(image).resize(size=new_size) for image in image_paths]

    print(i,animal_paths[i])
    # 存图片咯
    for fp,image in zip(image_paths,resized_image):
        new_filepath = os.path.join(resized_paths[i],os.path.basename(fp))
        # 将非jpg,jpeg的gif，png图片转为jpg
        if(os.path.splitext(new_filepath)[1].lower() not in ('.jpg','.jpeg')):
            if os.path.splitext(new_filepath)[1].lower() =='.png':
                new_filepath=os.path.join(resized_paths[i],os.path.splitext(new_filepath)[0]+'.jpg')
            else:
                image = image.convert('RGB')
                new_filepath = os.path.join(resized_paths[i], os.path.splitext(new_filepath)[0] + '.jpg')
        # print(2, new_filepath)
        image.save(new_filepath)

#
