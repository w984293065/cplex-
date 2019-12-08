#!/usr/bin/env python2
# -*- coding: utf-8 -*-
print("---")
import numpy as np
import re

#配置文件的全局变量
numOfPoint3=15  #  根据事情的情况而改变，表示三级点的数量
numOfPoint2=3  #  根据事情的情况而改变，表示二级点的数量
length=6  #  卡距离专用变量
cv2=20  #  标准车的单位运输成本
grade_3_distance=np.loadtxt('3-point-distance')#3级（末端）网点的距离表
car_loading=np.loadtxt('car-loading-size')#汽车运输量
#point_2_load_size=[12000,3500,5000,2300,3700,1600,2500,1000,2100,700]#二级点的分拣能力
point_2_load_size=[600,150,400,100,300,80,250,50,210,30]#二级点的分拣能力'
grade_3_and_grade_2_distance=np.loadtxt('3-2-point-distance')#3级和二级（末端）网点的距离表
point_12_distance=np.loadtxt('1-2-point-distance')#3级和二级（末端）网点的距离表
point_12_distance=np.reshape(point_12_distance,[1,-1])
expressWeight=[2,5,10,20,30]
#成本相关
moneyofProcessingExpress=np.loadtxt('money-express')
fs=300#建设成为标准点
fL=1000#建设成为大件快递点
fmix=1200#建设成为混合点

#求三、二级点之间的距离
def Point32Distance(id_a,id_b):
    return grade_3_and_grade_2_distance[id_a][id_b]

#求三级点之间的距离
def Point3Distance(id_a,id_b):
    return grade_3_distance[id_a][id_b]

#三级点的类
class Point3:#末端节点    
    def init(self,vmyid,vP):#全部定义为实例变量
        self.myid=vmyid#实例的id
        self.P=np.zeros([6,2])#特定区间包裹的数量
        self.small=0
        self.big=0
        self.smallcar=0
        self.bigcar=0
        for i in range(0,6):
            self.P[i][0]=int(vP[2*i])+0
            self.P[i][1]=int(vP[2*i+1])+0
    
#按固定文件路径初始化三级网点的数据
def Point3LoadData(filename):#从文本读入末端网点数据
    point3=[]
    f=open(filename) 
    s=f.readlines()
    for i in range(numOfPoint3):
        s[i]=re.split('\t|\n',s[i])
        s[i]=s[i][0:13]
        
    for i in range(numOfPoint3):
        temp_point3=Point3()
        temp_point3.init(s[i][0],s[i][1:13])
        point3.append(temp_point3)

    return point3


print("--")
