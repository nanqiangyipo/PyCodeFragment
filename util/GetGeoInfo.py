import json
import os

import sys

import time


class Geo:

    @property
    def guid(self):
        return self._guid
    @guid.setter
    def guid(self,name):
        self._guid=name

    @property
    def communityId(self):
        return self._ci
    @communityId.setter
    def communityId(self,name):
        self._ci=name

    @property
    def commDesc(self):
        return self._commDesc
    @commDesc.setter
    def commDesc(self,name):
        self._commDesc=name

    @property
    def policeStationId(self):
        return self._ps
    @policeStationId.setter
    def policeStationId(self, policeStaionId):
        self._ps=policeStaionId

    @property
    def stationDesc(self):
        return self._stationDesc
    @stationDesc.setter
    def stationDesc(self, name):
        self._stationDesc=name

    @property
    def areaId(self):
        return self._areaId
    @areaId.setter
    def areaId(self,ai):
        self._areaId=ai

    @property
    def areaDesc(self):
        return self._areaDesc
    @areaDesc.setter
    def areaDesc(self, name):
        self._areaDesc=name

rootPath=os.path.dirname(os.path.abspath('.'))
rootPath=os.path.join(rootPath,'data'+os.sep)
communityFile=os.path.join(rootPath,'shequ.json')
# 获得社区的id，中文信息，派出所id，区域id
communityOpen=open(communityFile,encoding='utf8')
comDict=dict(json.loads(communityOpen.read()))
comData=comDict.get('features')

count=1
geoList=list()
for oneCommunity in comData:
    properties=oneCommunity.get("properties")
    geo=Geo()
    geo.guid=properties.get("YWBSM")#社区唯一id号
    geo.policeStationId=properties.get("ZZJGDM")#派出所id
    geo.commDesc=properties.get("MC")
    geo.areaId=geo.policeStationId[0:6]+"000000"
    geo.communityId=geo.policeStationId[0:8]+repr(count).zfill(4)
    geoList.append(geo)
    count+=1

print(communityFile,len(geoList))
communityOpen.close()
# communityJson=open()
# 获得派出所的中文信息
pcsFile=os.path.join(rootPath,'pcs.geojson')
pcsOpen=open(pcsFile,encoding='utf8')
pcsDict=dict(json.loads(pcsOpen.read()))
pcsData=pcsDict.get('features')

pcsList=list()
print('获得派出所的中文信息')
for pcs in pcsData:
    pcs=pcs.get('properties')
    for geo in geoList:
        if geo.policeStationId==pcs.get('ZZJGDM'):
            geo.stationDesc=pcs.get('MC')

# print(communityFile,len(geoList))
pcsOpen.close()


# 更新重庆市区域名称
print('更新重庆市区域名称')
areaFile=os.path.join(rootPath,'chongqing.json')
areaOpen=open(areaFile,encoding='utf8')
areaDict=dict(json.loads(areaOpen.read()))
areaData=areaDict.get('features')

areaList=list()
for area in areaData:
    area=area.get('properties')
    for geo in geoList:
        if geo.areaId==area.get('ZZJGDM'):
            geo.areaDesc=area.get('MC')
areaOpen.close()

sqlPiece="insert into GEOGRAPHY_DIM_NEW (area_id, area_name, police_station_id, police_station_name, community_id, community_name, community_guid) values (" \
         "'{0}','{1}','{2}','{3}','{4}','{5}','{6}');"
count=0
g=geoList[6642:]
sqlList=list()
for geo in geoList:
    print(count)
    count+=1
    temp=sqlPiece[:]
    print(count,temp.format(geo.areaId
                      ,geo.areaDesc
                      ,geo.policeStationId
                      ,geo.stationDesc
                      ,geo.communityId
                      ,geo.commDesc
                      ,geo.guid))
    sqlList.append(temp.format(geo.areaId
                      ,geo.areaDesc
                      ,geo.policeStationId
                      ,geo.stationDesc
                      ,geo.communityId
                      ,geo.commDesc
                      ,geo.guid)+"\n")
# geo_dim=open(os.path.join(rootPath,'geography_dim.sql'),mode='w',encoding='utf8')
geo_dim=open(os.path.join(rootPath,'geography_dim.sql'),mode='w')
geo_dim.writelines(sqlList)
geo_dim.flush()