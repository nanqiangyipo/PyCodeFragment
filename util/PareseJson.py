import json

import os

# xColum=30
# yColum=30
from math import ceil

xColum=yColum=200
jsonPath=os.path.dirname(os.path.abspath('.'))##获取当前目录的父目录

jsonPath=os.path.join(jsonPath, 'data' + os.sep + 'shiju.geojson')
# print(curPath)

cqjson=open(jsonPath, encoding='utf8')
cqDict=dict(json.loads(cqjson.read()))
cqjson.close()


# print(cqDict)#获取经纬度坐标集
# print(cqDict.keys())
xyList=cqDict.get("features")[0].get("geometry").get("coordinates")[0]
x=[point[0] for point in xyList]
y=[point[1] for point in xyList]

# 获得左下角和右上角坐标
maxX=max(x)
minX=min(x)

maxY=max(y)
minY=min(y)

print("右上角坐标(x,y)=("+repr(maxX),","+repr(maxY)+")",end='\n')
print("左下角坐标(x,y)=("+repr(minX),","+repr(minY)+")",end='\n')
# 计算间隔
xSpan=maxX-minX
ySpan=maxY-minY
print(repr(xColum)+"列经度间隔："+repr((xSpan)/xColum) ,repr(yColum)+"行纬度间隔："+repr((ySpan)/yColum),sep='\n')
# 计算起始坐标
x0=minX+xSpan/xColum/2
y0=minY+ySpan/yColum/2
print("左下角第一个方格中心坐标：\n（\n"+repr(x0)+"\n,\n"+repr(y0)+"\n)")

#差值
subtractionListX=list()
subtractionListY=list()
for i in range(1,len(x)):
    subtractionListX.append(x[i]-x[i-1])
for i in range(1,len(y)):
    subtractionListY.append(y[i]-y[i-1])


maxX=max(subtractionListX)
minX=min(subtractionListX)

maxY=max(subtractionListY)
minY=min(subtractionListY)
# print(maxX,minX,sep='\n')
# print(maxY,minY,sep='\n')


# step=1
# maxAmounts=2000
# tatol=9500
# if(tatol>maxAmounts):
#     step=ceil(tatol/(maxAmounts));
# c=0
#
# for i in range(0,tatol-1,1):
#     i=i*step
#     if i>tatol:
#         break
#     c+=1
#     print(tatol,step,i,c,sep="--")
# print("c=",c)
