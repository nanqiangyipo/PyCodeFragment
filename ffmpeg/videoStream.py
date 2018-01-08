# coding:utf-8

import cv2
import sys

def play(url):
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("opened")

    while cap.isOpened():
        ret,frame = cap.read()
        cv2.imshow("frame",frame)
        cv2.waitKey(1)


def main():
    # if len(sys.argv) != 2:
    #     print ("Usage : ")
    #     print ("        python tplayer.py")
    #     print ("Author : ") 
    #     print ("        WangYihang <wangyihanger@gmail.com>")
    #     exit(1)
    # url = sys.argv[1]
    url="http://ivi.bupt.edu.cn/hls/cctv3hd.m3u8"
    play(url)


if __name__ == '__main__':
    main()

