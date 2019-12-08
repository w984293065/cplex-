#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import copy
import config as cf
import time
from docplex.mp.model import Model


# In[2]:


def superCustomer(i, point3):  #  生成两类超级顾客，为列表，每一个元素代表该超级顾客的顾客点的组成
    # 用来存放超级顾客信息
    smallSup = []
    bigSup = []

    # 根据该划分值求三级网点普通和大件的量
    for j in range(cf.numOfPoint3):
        # 求普通件的量
        sumnum = 0
        for t in range(i + 1):
            sumnum = sumnum + point3[j].P[t][0]
        point3[j].small = sumnum
        # 求大件的量
        sumnum = 0
        for t in range(5 - i):
            sumnum = sumnum + point3[j].P[5 - t][0]
        point3[j].big = sumnum
        # print("i=%d small=%d big=%d"%(j,point3[j].small,point3[j].big))
    # 生成普通件超级顾客，用V2进行运输
    pointleft = np.zeros(cf.numOfPoint3)  # 点被使用与否的标志，0表示未划分，1表示划分
    car = cf.car_loading[1][2 * i]
    ##开始模拟生成
    while (isAllUsed(pointleft) == 0):
        oneSuperCustomer = []  # 一个超级顾客成员
        wholeExpressNum = 0  # 一个超级顾客总运输量
        wholeTime = 0  # 一个超级顾客花费总时间
        lastPoint = -1  # 上一个待加入的点
        while (car > wholeExpressNum and wholeTime < 100):
            # 第一个点，顺序选择未加入的点加入
            if wholeTime == 0:
                nowPoint = -1  # 待加入的点
                for j in range(cf.numOfPoint3):
                    if pointleft[j] == 0:
                        nowPoint = j
                        stayTime = np.random.randint(15, 20)  # 每个点花费的时间
                        oneSuperCustomer.append(nowPoint)
                        pointleft[nowPoint] = 1
                        wholeExpressNum = wholeExpressNum + point3[nowPoint].small
                        wholeTime = wholeTime + stayTime
                        lastPoint = nowPoint
                        break
            # 除第一个点按照距离最近的点加入
            else:
                distance = 1000  # 定义一个超长距离
                nowPoint = -1  # 定义待加入的点
                for j in range(cf.numOfPoint3):
                    if cf.Point3Distance(lastPoint, j) < distance and lastPoint != j and pointleft[j] == 0:
                        distance = cf.Point3Distance(lastPoint, j)
                        nowPoint = j
                # 判断能不能加入这个点
                stayTime = np.random.randint(15, 20)  # 点花费的时间
                if wholeTime + stayTime > 100 or wholeExpressNum + point3[nowPoint].small > car or nowPoint == -1:
                    break
                else:
                    oneSuperCustomer.append(nowPoint)
                    pointleft[nowPoint] = 1
                    wholeExpressNum = wholeExpressNum + point3[nowPoint].small
                    wholeTime = wholeTime + stayTime
                    lastPoint = nowPoint
        smallSup.append(oneSuperCustomer)

    # 生成大件超级顾客，用V3进行运输
    pointleft = np.zeros(cf.numOfPoint3)  # 点被使用与否的标志，0表示未划分，1表示划分
    car = cf.car_loading[2][2 * i + 1]
    ##开始模拟生成
    while (isAllUsed(pointleft) == 0):
        oneSuperCustomer = []  # 一个超级顾客成员
        wholeExpressNum = 0  # 一个超级顾客总运输量
        wholeTime = 0  # 一个超级顾客花费总时间
        lastPoint = -1  # 上一个待加入的点
        while (car > wholeExpressNum and wholeTime < 180):
            # 第一个点，顺序选择未加入的点加入
            if wholeTime == 0:
                nowPoint = -1  # 待加入的点
                for j in range(cf.numOfPoint3):
                    if pointleft[j] == 0:
                        nowPoint = j
                        stayTime = np.random.randint(15, 20)  # 每个点花费的时间
                        oneSuperCustomer.append(nowPoint)
                        pointleft[nowPoint] = 1
                        wholeExpressNum = wholeExpressNum + point3[nowPoint].big
                        wholeTime = wholeTime + stayTime
                        lastPoint = nowPoint
                        break
            # 除第一个点按照距离最近的点加入
            else:
                distance = 1000  # 定义一个超长距离
                nowPoint = -1  # 定义待加入的点
                for j in range(cf.numOfPoint3):
                    if cf.Point3Distance(lastPoint, j) < distance and lastPoint != j and pointleft[j] == 0:
                        distance = cf.Point3Distance(lastPoint, j)
                        nowPoint = j
                # 判断能不能加入这个点
                stayTime = np.random.randint(15, 16)  # 点花费的时间
                if wholeTime + stayTime > 180 or wholeExpressNum + point3[nowPoint].big > car or nowPoint == -1:
                    break
                else:
                    oneSuperCustomer.append(nowPoint)
                    pointleft[nowPoint] = 1
                    wholeExpressNum = wholeExpressNum + point3[nowPoint].big
                    wholeTime = wholeTime + stayTime
                    lastPoint = nowPoint
        bigSup.append(oneSuperCustomer)

    # 返回
    return smallSup, bigSup
def smallSup_size(smallSup,point3):  #  计算每个标准超级顾客包含的标准包裹数量
    smallSup_sizes=[]
    for l in range(len(smallSup)):
        w = 0
        for m in range(len(smallSup[l])):
            w += point3[smallSup[l][m]].small
        smallSup_sizes.append(w)
    return smallSup_sizes

def bigSup_size(bigSup,point3):  #  计算每个大件超级顾客包含的大件包裹数量
    bigSup_sizes=[]
    for l in range(len(bigSup)):
        w = 0
        for m in range(len(bigSup[l])):
            w += point3[bigSup[l][m]].big
        bigSup_sizes.append(w)
    return bigSup_sizes

def smallSup_cycle_distance(smallSup):
    '估算超级顾客内部循环的距离'
    list5 = []  # 运来存放每个超级顾客内部的最长距离
    for j in range(len(smallSup)):
        list1 = []
        for l in range(len(smallSup[j]) - 1):
            for m in range(l, len(smallSup[j])):
                list1.append(cf.Point3Distance(smallSup[j][l], smallSup[j][m]))
        list1.sort()
        list1.reverse()
        if len(list1) == 0:  # 由于超级顾客在形成中产生了空超级顾客，所以有此判断；若不会形成空超级顾客，则可删除
            list5.append(0)
        else:
            list5.append(list1[0])
    cycle_distance = []
    for j in range(len(list5)):
        cycle_distance.append(0.71*((list5[j] ** 2 / 2 * len(smallSup[j])) ** (1 / 2)))
    return cycle_distance

def bigSup_cycle_distance(bigSup):
    list5 = []  # 运来存放每个超级顾客内部的最长距离
    for j in range(len(bigSup)):
        list1 = []
        for l in range(len(bigSup[j]) - 1):
            for m in range(l, len(bigSup[j])):
                list1.append(cf.Point3Distance(bigSup[j][l], bigSup[j][m]))
        list1.sort()
        list1.reverse()
        if len(list1) == 0:  # 由于超级顾客在形成中产生了空超级顾客，所以有此判断；若不会形成空超级顾客，则可删除
            list5.append(0)
        else:
            list5.append(list1[0])
    cycle_distance = []
    for j in range(len(list5)):
        cycle_distance.append(0.71*((list5[j] ** 2 / 2 * len(bigSup[j])) ** (1 / 2)))
    return cycle_distance

#判断point3是否全部被划分
def isAllUsed(Array):
    for i in range(len(Array)):
        if Array[i]==0:
            return 0
    return 1

#  cyk:生成各超级顾客到各中转中心的距离（平均距离），每行代表该超级顾客到各中心的距离，每列代表该中心到各超级顾客的距离
def get_distance(smallSup,bigSup,point3):
    smallSupDistance=np.zeros((len(smallSup),int(cf.numOfPoint2)))
    for i in range(len(smallSup)):
        for j in range(cf.numOfPoint2):
            for k in range(len(smallSup[i])):
                smallSupDistance[i][j]+=cf.Point32Distance(smallSup[i][k],j)*(1/len(smallSup[i]))
    bigSupDistance=np.zeros((len(bigSup),int(cf.numOfPoint2)))
    for i in range(len(bigSup)):
        for j in range(cf.numOfPoint2):
            for k in range(len(bigSup[i])):
                bigSupDistance[i][j]+=cf.Point32Distance(bigSup[i][k],j)*(1/len(bigSup[i]))
    return smallSupDistance,bigSupDistance

def stock_table(Sup):
    table=[]
    [table.append([]) for i in range(cf.numOfPoint2)]
    for j in range(len(table)):
        [table[j].append(0) for k in range(len(Sup))]
    return table

def get_small(i,table,smallSup,smallSupDistance,point3,smallSup_sizes):
    totalmoney=0
    totalopen=0
    m1=0
    m2=0
    m3=0
    m4=0
    m5=0
    allsizesize=0
    x_location=[]
    for j in range(cf.numOfPoint2):  #  可以尝试简化
        if any(table[j][k]!=0 for k in range(len(smallSup))):
            x_location.append(j)
    totalopen=len(x_location)
    totalmoney=totalopen*cf.fs
    m1+=totalopen*cf.fs  #  第一部分建设成本

    for j in x_location:
        totalmoney += 2 * cf.point_12_distance[0][j]*10
        m2 += 2 * cf.point_12_distance[0][j]*10  # 一级到二级的运输成本
    final_size=0
    for j in range(cf.numOfPoint3):
        for k in range(i+1):
            for l in range(2):
                final_size+=point3[j].P[k][l]
    totalmoney+=final_size*cf.moneyofProcessingExpress[i][1]
    m3=final_size*cf.moneyofProcessingExpress[i][1]  #  二级点的处理成本
    list5 = []  # 运来存放每个超级顾客内部的最长距离
    for j in range(len(smallSup)):
        list1 = []
        for l in range(len(smallSup[j]) - 1):
            for m in range(l, len(smallSup[j])):
                list1.append([cf.Point3Distance(l, m), l, m])
        list1.sort()
        list1.reverse()
        if len(list1) == 0:  # 由于超级顾客在形成中产生了空超级顾客，所以有此判断；若不会形成空超级顾客，则可删除
            list5.append(0)
        else:
            list5.append(list1[0][0])
    cycle_distance = []
    for j in range(len(list5)):
        cycle_distance.append((list5[j] ** 2 / 2 * len(smallSup[j])) ** (1 / 2))
    tt = 0  # 记录空超级顾客的数量
    distance = 0
    for j in range(len(cycle_distance)):
        if cycle_distance[j] == 0:
            tt += 1
    for j in x_location:
        for k in range(len(smallSup)):
            if table[j][k] == 1 and cycle_distance[k] != 0:
                distance += 2 * smallSupDistance[k][j]
    totaldistance = distance + sum(cycle_distance)
    totalmoney += cf.cv2 * totaldistance
    m4 += cf.cv2 * totaldistance

    '''for j in x_location:
        for k in range(len(smallSup)):
            list1 = []  # 用来存放已经开放的中转中心所服务的三级点之间的距离和编号
            if table[j][k] == 1:
                for l in range(len(smallSup[k])):
                    for m in range(l+1,len(smallSup[k])):
                        list1.append([cf.Point32Distance(l,m),l,m])
                list1.sort()
                list1.reverse()
                if len(list1)==0:
                    totaldistance=((((0/(2**(1/2)))**2)*len(smallSup[k]))**(1/2))*0.71+2*smallSupDistance[k][j]
                else:
                    totaldistance=((((list1[0][0]/(2**(1/2)))**2)*len(smallSup[k]))**(1/2))*0.71+2*smallSupDistance[k][j]
                totalmoney+=1*totaldistance
                m4+=1*totaldistance  #  二级点到超级顾客和超级顾客内部的运输成本'''
    final_size=0
    for j in range(cf.numOfPoint3):
        for k in range(i+1):
            for l in range(2):
                final_size+=point3[j].P[k][l]
    
    totalmoney+=final_size*cf.moneyofProcessingExpress[i][0]
    m5+=final_size*cf.moneyofProcessingExpress[i][0]
    return totalmoney, m1,m2+m4,m3+m5

def get_big(i,table,bigSup,bigSupDistance,point3,bigSup_sizes):
    totalmoney=0
    totalopen=0
    m1=0
    m2=0
    m3=0
    m4=0
    m5=0
    allsizesize=0
    x_location=[]
    for j in range(cf.numOfPoint2):
        if any(table[j][k]==1 for k in range(len(bigSup))):
            x_location.append(j)
    totalopen=len(x_location)
    totalmoney=totalopen*cf.fL
    m1+=totalopen*cf.fL
    for j in x_location:
        totalmoney += 2 * cf.point_12_distance[0][j]*10
        m2 += 2 * cf.point_12_distance[0][j]*10  # 一级到二级的运输成本

    final_size=0
    for j in range(cf.numOfPoint3):
        for k in range(5-i):
            for l in range(2):
                final_size+=point3[j].P[5-k][l]
    totalmoney += final_size * cf.moneyofProcessingExpress[i][3]
    m3 += final_size * cf.moneyofProcessingExpress[i][3]  # 二级点的处理成本
    list5 = []  # 运来存放每个超级顾客内部的最长距离
    for j in range(len(bigSup)):
        list1 = []
        for l in range(len(bigSup[j]) - 1):
            for m in range(l, len(bigSup[j])):
                list1.append([cf.Point3Distance(l, m), l, m])
        list1.sort()
        list1.reverse()
        if len(list1) == 0:  # 由于超级顾客在形成中产生了空超级顾客，所以有此判断；若不会形成空超级顾客，则可删除
            list5.append(0)
        else:
            list5.append(list1[0][0])
    cycle_distance = []
    for j in range(len(list5)):
        cycle_distance.append((list5[j] ** 2 / 2 * len(bigSup[j])) ** (1 / 2))
    tt = 0  # 记录空超级顾客的数量
    distance = 0
    for j in range(len(cycle_distance)):
        if cycle_distance[j] == 0:
            tt += 1
    for j in x_location:
        for k in range(len(bigSup)):
            if table[j][k] == 1 and cycle_distance[k] != 0:
                distance += 2 * bigSupDistance[k][j]
    totaldistance = distance + sum(cycle_distance)
    totalmoney += 5 * totaldistance
    m4 += 5 * totaldistance

    '''for j in x_location:
            for k in range(len(bigSup)):
                if table[j][k] == 1:
                    if list5[k]==0:      #    由于超级顾客在形成中产生了空超级顾客，所以有此判断；若不会形成空超级顾客，则可删除
                        totaldistance=0
                    else:
                        totaldistance = ((list5[k] ** 2) / 2) * len(bigSup[k]) ** (1 / 2) + 2 * 0.71 * bigSupDistance[k][j]
                    print('totaldistance=',totaldistance)
                    totalmoney += 5 * totaldistance
                    m4 += 5 * totaldistance  # 二级点到超级顾客和超级顾客内部的运输成本'''
    '''for j in x_location:
        for k in range(len(bigSup)):
            list1 = []  # 用来存放已经开放的中转中心所服务的三级点之间的距离和编号
            if table[j][k] == 1:
                for l in range(len(bigSup[k])):
                    for m in range(l+1,len(bigSup[k])):
                        list1.append([cf.Point32Distance(l,m),l,m])
                list1.sort()
                list1.reverse()
                if len(list1)==0:
                    totaldistance=((((0/(2**(1/2)))**2)*len(bigSup[k]))**(1/2))*0.71+2*bigSupDistance[k][j]
                else:
                    totaldistance=((((list1[0][0]/(2**(1/2)))**2)*len(bigSup[k]))**(1/2))*0.71+2*bigSupDistance[k][j]
                totalmoney+=5*totaldistance
                m4+=5*totaldistance  #  二级点到超级顾客和超级顾客内部的运输成本'''
    final_size=0
    for j in range(cf.numOfPoint3):
        for k in range(5-i):
            for l in range(2):
                final_size+=point3[j].P[5-k][l]
    totalmoney+=final_size*cf.moneyofProcessingExpress[i][2]
    m5+=final_size*cf.moneyofProcessingExpress[i][2]  #  三级点的处理成本
    return totalmoney, m1,m2+m4,m3+m5


# In[3]:


if __name__=='__main__':
    start=time.time()
    cv_1=10  #  一层运输的单位距离运输成本
    cv_2=2  #  二层标准运输的单位距离运输成本
    cv_3=5  #  二层大件运输的单位距离运输成本

    point3 = cf.Point3LoadData('3-point-expressNum-new.txt')
    
    for we in range(5):
        print("正在生成%dkg的相关信息---" % (cf.expressWeight[we]))

        smallSup, bigSup = superCustomer(we, point3)
        smallSup_sizes=smallSup_size(smallSup, point3)
        bigSup_sizes=bigSup_size(bigSup, point3)
        small_cycle_distance=smallSup_cycle_distance(smallSup)
        big_cycle_distance=bigSup_cycle_distance(bigSup)
        smallSupDistance,bigSupDistance=get_distance(smallSup,bigSup,point3)

        model=Model()

        #  创建必要的集合

        M={j for j in range(cf.numOfPoint2)}  #  中转中心备选点集合
        K={j for j in range(len(smallSup))}  #  标准超级顾客集合
        G={j for j in range(len(bigSup))}  #  大件超级顾客集合



        #  创建d_mk：中转中心m到标准超级顾客k的往返距离（包括内部循环距离）
        d_mk=np.zeros((len(K),len(M)))
        for i in range(len(K)):
            for j in range(len(M)):
                d_mk[i,j]=smallSupDistance[i,j]*2+small_cycle_distance[i]
        d_mk=d_mk.T

        #  创建d_mg：中转中心m到大件超级顾客k的往返距离（包括内部循环距离）
        d_mg=np.zeros((len(G),len(M)))
        for i in range(len(G)):
            for j in range(len(M)):
                d_mg[i,j]=bigSupDistance[i,j]*2+big_cycle_distance[i]
        d_mg=d_mg.T

        x_var_list=[(i,j) for i in range(3) for j in range(len(M))]
        y_var_list=[(i,j) for i in range(len(M)) for j in range(len(K))]
        z_var_list=[(i,j) for i in range(len(M)) for j in range(len(G))]

        #  生成变量
        x=model.binary_var_dict(x_var_list,name='x')
        y=model.binary_var_dict(y_var_list,name='y')
        z=model.binary_var_dict(z_var_list,name='z')

        part1=sum( x[0,m]*cf.fs + x[1,m]*cf.fL + x[2,m]*cf.fmix for m in M)

        part2=2*sum(cv_1*cf.point_12_distance[0,m]*(x[0,m]+x[1,m]+x[2,m]) for m in M)

        part3=sum(cv_2*d_mk[m,k]*y[m,k] for m in M for k in K)

        part4=sum(cv_3*d_mg[m,g]*z[m,g] for m in M for g in G)

        model.minimize(part1+part2+part3+part4)

        for m in M:
            model.add_constraint(sum(x[t,m] for t in range(3)) <= 1)

        for k in K:
            model.add_constraint(sum(y[m,k] for m in M) == 1)

        for g in G:
            model.add_constraint(sum(z[m,g] for m in M) == 1)

        for k in K:
            for m in M:
                model.add_constraint(y[m,k]-x[0,m]-x[2,m] <= 0)

        for g in G:
            for m in M:
                model.add_constraint(z[m,g]-x[1,m]-x[2,m] <= 0)

        for m in M:
            model.add_constraint(sum(y[m,k]*smallSup_sizes[k] for k in K) + sum(z[m,g]*bigSup_sizes[g] for g in G) <= cf.point_2_load_size[we]*x[0,m]+cf.point_2_load_size[we+1]*x[1,m]+(cf.point_2_load_size[we]+cf.point_2_load_size[we+1])*x[2,m])

        sol=model.solve()

        table_small=stock_table(smallSup)

        table_big=stock_table(bigSup)

        for m in M:
            for k in K:
                table_small[m][k]=y[m,k].solution_value
        for m in M:
            for g in G:
                table_big[m][g]=z[m,g].solution_value

        small_cost,small_cost1,small_cost2,small_cost3=get_small(we,table_small,smallSup,smallSupDistance,point3,smallSup_sizes)

        big_cost,big_cost1,big_cost2,big_cost3=get_big(we,table_big,bigSup,bigSupDistance,point3,bigSup_sizes)
        
        small_x_location=[]
        big_x_location=[]
        mix_x_location=[]
        
        for m in M: 
            if x[0,m].solution_value==1:
                small_x_location.append(m)
            if x[1,m].solution_value==1:
                big_x_location.append(m)
            if x[2,m].solution_value==1:
                mix_x_location.append(m)
        reduced_cost=0
        for j in small_x_location:
            for k in big_x_location:
                if j==k:
                    reduced_cost+=cf.point_12_distance[0][j]*10*2
        total_cost=small_cost+big_cost-reduced_cost
        total_cost1=small_cost1+big_cost1
        total_cost2=small_cost2+big_cost2-reduced_cost
        total_cost3=small_cost3+big_cost3
        
        
        print('标准中转中心开放编号=',small_x_location,',个数为：',len(small_x_location))
        print('大件中转中心开放编号=',big_x_location,',个数为:',len(big_x_location))
        print('混合中转中心开放编号=',mix_x_location,',个数为:',len(mix_x_location))
        print('总成本=%f,建设成本=%f,运输成本=%f,处理成本=%f'%(total_cost,total_cost1,total_cost2,total_cost3))
        end=time.time()
        print('time=',end-start)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




