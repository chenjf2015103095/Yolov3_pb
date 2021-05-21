# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Author :       陈剑锋
   Date：         2021-04-16 下午2:05
   Description :  Dream it possible!

-------------------------------------------------
   Change Activity:

-------------------------------------------------
"""

import os
from PIL import Image

class Convert_Image(object):

    def __init__(self):

        pass


def bmpToJpg(file_path):
    """
    将bmp格式转换为jpg
    :param file_path:
    :return:
    """
    for fileName in os.listdir(file_path):
        print(fileName)
        afileName=fileName
        # newFileName = fileName[0:fileName.find("_")]+".jpg"
        newFileName = fileName+'.jpg'
        print(newFileName)
        try:
            im = Image.open(file_path+"/"+fileName)
            jpg_path=file_path+'/JPG'
            if not os.path.exists(jpg_path):
                os.mkdir(jpg_path)
            im.save(jpg_path+"/"+newFileName)
        except Exception as ex:
            print(ex)


def deleteImages(file_path):
    """
    删除原来的位图
    :param file_path:
    :return:
    """
    for file in os.listdir(file_path):
        afile=file
        if afile.endswith('.jpg'):
            name=afile.split('.jpg')[0]
            file_jpg=name+'.jpg'
            f_path=os.path.join(file_path,file_jpg)
            print(f_path)
            # os.remove(f_path)


def main():
    file_path = "/media/ubuntu/Seagate Expansion Drive/chenjianfeng/20210427_data/01"
    bmpToJpg(file_path)
    deleteImages(file_path)


if __name__ == '__main__':
    main()
