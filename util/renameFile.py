import os

# files=os.listdir(r'E:\download\video\厨师烹饪技术大全视频教程 厨师基本功刀功培训教材资料\资料\烹饪大全视频B\烹饪26法B')
# files=os.listdir(r'E:\download\video\厨师烹饪技术大全视频教程 厨师基本功刀功培训教材资料\资料\烹饪大全视频A\烹饪26法A')
#
# print(files)

# for

class Files:
    def getFileList(self,parentPath):
        self.files=os.listdir(parentPath)
        print(self.files)


f=Files()
f.getFileList(r'D:\installed\百度云管家_vip破解_单文件版\App\BaiduYunGuanjia')