#并行更新+双层模型1

import random
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import itertools as it
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import animation
import moviepy.editor as mp
import argparse
import pandas as pd
from matplotlib import rcParams
import pylab 
import copy
import itertools as it 

matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

#-------------------------------------------------------------------------
global ks,kd,kw,kL,r1,r2,ke,kv,kg,g1,g2,MAX_STEPS,dx,exit,enter,sim1,lrwidth,ewidth,decay,slight,image_time,ACC

# 系统中的人数
npeds = 60
#静态场域灵敏度参数
ks = 1.5
#动态场域灵敏度参数
kd = 0
#行人路径选择路径长度灵敏度参数
kL = 1.5
#行人路径选择路径拥堵灵敏度参数
kj = 8
#行人路径选择路径跟随特性灵敏度参数
kg = 1
#模拟时间步长
MAX_STEPS = 120
#生成图片的时间间隔
image_time = 10
#路径内是否可以加速，1表示可以，0表示不可以
ACC=0

#----------------------------------------------
steps = range(MAX_STEPS)
dx = 150  #圆点大小
lrwidth = 6 #左右预留宽度
ewidth =1  #入口宽度
L = 2*lrwidth+24+2*ewidth #总长度
w = 22 #总宽度
w1= 10 #疏散区宽度 
w2 = 12  #路径选择区域宽度
slight = 8  #视野区域长度
decay = 0.005
r1 = 2
r2 =1 #路径选择两条路的长度
acc={1:0.25,2:0.45,3:0.3}

exit=(lrwidth+1,w1)
enter = (L-lrwidth,w1)
route = [(enter[0]-1,enter[1]),(enter[0],enter[1]+1)]

sim1=[]  #疏散区域左右预留行人位置坐标集
for i in range(1,lrwidth+1):
    sim1.append((i,w1))
    sim1.append((L-i+1,w1))

sim0=[]  #疏散区矩形区域
for i,j in it.chain(it.product(range(1,L+1), range(1,w1))):
     sim0.append((i,j))

route1=[]
for i in range(1,25):
    route1.append((enter[0]-i,enter[1]))

route21=[]
route22=[]
route23=[]
for i in range(1,w2):
    route21.append((enter[0],enter[1]+i))
for i in range(0,25):
    route22.append((enter[0]-i,w))
for i in range(0,w2):
    route23.append((exit[0],w-i))  #不包含出口

route2 =[]
route2 = route21+route22+route23  #一共48个

peds_filename = 'peds_images'
video_filename = 'peds_video'

#-----------------------------------函数部分--------------------------------------
#绘制整个实验区域
def plot_space():
    plt.cla()
    fig = plt.figure(figsize = (19,11))
    ax = plt.axes()
    plt.xlim(0.4,L + 0.6)
    plt.ylim(0.4,w + 0.6)
    plt.axis('off')

    #绘制边界
    plt.plot([0.5,L+0.5],[0.5,0.5],color='k',linewidth=3,linestyle='-')
    plt.plot([0.5,lrwidth+0.5],[w1+0.5,w1+0.5],color='k',linewidth=3,linestyle='-')
    plt.plot([L-lrwidth+0.5,L+0.5],[w1+0.5,w1+0.5],color='k',linewidth=3,linestyle='-')
    plt.plot([lrwidth+0.5,L-lrwidth+0.5],[w+0.5,w+0.5],color='dodgerblue',linewidth=3,linestyle='-')
    plt.plot([lrwidth+1.5,L-lrwidth-0.5],[w1-0.5,w1-0.5],color='dodgerblue',linewidth=3,linestyle='-')
    plt.plot([lrwidth+1.5,L-lrwidth-0.5],[w1+0.5,w1+0.5],color='dodgerblue',linewidth=3,linestyle='-')
    plt.plot([lrwidth+1.5,L-lrwidth-0.5],[w-0.5,w-0.5],color='dodgerblue',linewidth=3,linestyle='-')
   
    plt.plot([0.5,0.5],[0.5,w1+0.5],color='k',linewidth=3,linestyle='-')
    plt.plot([L+0.5,L+0.5],[0.5,w1+0.5],color='k',linewidth=3,linestyle='-')
    plt.plot([lrwidth+0.5,lrwidth+0.5],[w1+0.5,w+0.5],color='dodgerblue',linewidth=3,linestyle='-')
    plt.plot([lrwidth+1.5,lrwidth+1.5],[w1+0.5,w-0.5],color='dodgerblue',linewidth=3,linestyle='-')
    plt.plot([L-lrwidth-0.5,L-lrwidth-0.5],[w1+0.5,w-0.5],color='dodgerblue',linewidth=3,linestyle='-')
    plt.plot([L-lrwidth+0.5,L-lrwidth+0.5],[w1+0.5,w+0.5],color='dodgerblue',linewidth=3,linestyle='-')

    #绘制行人疏散区的网格
    for i in range(1,L):
        x = [i+0.5,i+0.5]
        y = [0.5,w1 + 0.5]#竖线
        plt.plot(x,y,color='dimgrey',linewidth=1,linestyle='-')
    for j in range(1,10):
        if j==9:
            x1 = [0.5,lrwidth+1.5]
            x2 = [L-lrwidth-0.5,L+0.5]
            y= [9.5,9.5]
            plt.plot(x1,y,color='dimgrey',linewidth=1,linestyle='-')
            plt.plot(x2,y,color='dimgrey',linewidth=1,linestyle='-')
        else:
            x = [0.5,L + 0.5]
            y = [j+0.5,j+0.5] #横线
            plt.plot(x,y,color='dimgrey',linewidth=1,linestyle='-')

    #绘制行人行走区的网格
    for i in range(10,22):
        x1 = [lrwidth+0.5,lrwidth+1.5]
        x2 = [L-lrwidth-0.5,L-lrwidth+0.5]
        y = [i+0.5,i+0.5]#横线
        plt.plot(x1,y,color='dimgrey',linewidth=1,linestyle='-')
        plt.plot(x2,y,color='dimgrey',linewidth=1,linestyle='-')
    for i in range(lrwidth,L-lrwidth+1):
        y = [w-0.5,w+0.5]
        x = [i+0.5,i+0.5]#竖线
        plt.plot(x,y,color='dimgrey',linewidth=1,linestyle='-')
    
    plt.text(L/2,16,'行人路径选择区域',ha='center',va='center',fontdict=dict(fontsize=28, color='steelblue'))
    plt.text(exit[0],exit[1],'出',ha='center',va='center',fontdict=dict(fontsize=17, color='seagreen'),alpha=0.8)
    plt.text(enter[0],enter[1]+1,'2',ha='center',va='center',fontdict=dict(fontsize=17, color='seagreen'),alpha=0.8)
    plt.text(enter[0]-1,enter[1],'1',ha='center',va='center',fontdict=dict(fontsize=17, color='seagreen'),alpha=0.8)
    plt.text(enter[0],enter[1],'入',ha='center',va='center',fontdict=dict(fontsize=16, color='seagreen'),alpha=0.8)
    plt.savefig('space.png')
    return 

#绘制行人的散点图  
def plot_peds(peds,t,n1,n2,n0):
    peds1 = np.array(peds)
    x = peds1[:,0]
    y = peds1[:,1]
    plot_space()
    plt.scatter(x,y,color='tomato',s = dx)
    S = 't: %3.3d | 路径1选择:%2.2d 路径2选择:%2.2d 已选择人数:%2.2d' % (t,n1,n2,n0)
    plt.title("%15s" % S,fontsize = 26)
    figure_name = os.path.join(peds_filename, 'peds%.6d.png' % t)
    plt.savefig(figure_name)
    plt.close()

#初始化行人数组
def init_peds():
  
    #随机分布行人
    w11 = 9
    L11 = L
    num = 0
    lisz = []
    while num != npeds:
        sjs = np.random.randint(1,w11*L11+1)
        if ((sjs in lisz) == 0):
            lisz.append(sjs)
            num += 1   
    x= []
    y = []
    peds=[]
    
    for i in range(0,len(lisz)):
        x=int((lisz[i] - 1) / w11) + 1
        y=lisz[i] - (x - 1) * w11
        x += L-L11
        peds.append((x,y))

    return peds  #列表的顺序默认为行人顺序！按顺序更新位置即可

##计算静态场域
def SFF():
    sff = {}
    
    for i,j in it.chain(it.product(range(1,L+1), range(1,w1))):
            distance = abs(i-enter[0])+abs(j-enter[1])
            sff[(i,j)]= 40-distance
    for (i,j) in sim1:  
        if i >= lrwidth+1:
            distance = abs(i-enter[0])+abs(j-enter[1])
            sff[(i,j)]= 40-distance
        elif i <= lrwidth:
            distance = abs(i-enter[0])+abs(j-enter[1])
            sff[(i,j)]= 40-(distance+2)
    sff[enter]=40
    return sff
  

def plot_SFF(sff):
    
    plot_space()
    for k, v in sff.items():
        i,j = k
        plt.text(i,j,v,ha='center',va='center',fontdict=dict(fontsize=12, color='r'))

    plt.savefig('SFF.png')

#疏散区行人邻居，四个
def get_neigh(cell):
    
    neighbors = []
    i, j = cell
    
    if cell==(enter[0]+1,enter[1]) or cell==(enter[0],enter[1]-1):
        neighbors.append(enter)
    else:
        neighbors1=[(i-1,j),(i,j+1),(i,j-1),(i+1,j)]
        neighbors=[(i-1,j),(i,j+1),(i,j-1),(i+1,j)]
        for m,n in neighbors1:
            if (m,n)  not in sim1+sim0:
                neighbors.remove((m,n))
                
    random.shuffle(neighbors)
   
    return neighbors


#初始化动态场域
def init_DFF():
    return np.zeros((L+1,w+1))

#动态场域的衰减
def update_DFF(dff):
    '''
    上下边界墙体动态场域==0
    '''
    for i,j in it.chain(it.product(range(1, L+1), range(1, w1+1))):
        dff[i,j] = (1-decay)*dff[i,j]
    return dff

#疏散区域行人移动概率
def sim1_peds(cell,peds,dff):

    probs ={}
    sump =0
    neis = get_neigh(cell)
    nei= []

    p0 = np.exp(ks*sff[cell]+kd*dff[cell]) #不存在位置交换
    probs[(0,0)] = p0 
    nei.append((0,0))   #计算原地静止不动概率
    sump += p0
    
    for neighbor in neis:
        p = np.exp(ks*sff[neighbor]+kd*dff[neighbor])*(1-int(neighbor in peds)) #不存在位置交换
        if p != 0:
            probs[neighbor] = p  
            nei.append(neighbor)
            sump += p 
            
    r = np.random.rand() * sump

    if sump == 0:  # pedestrian in cell can not move
        pos_peds = (0,0)
    else:
        for neighbor in nei: 
            r -= probs[neighbor]
            if r <= 0:  
                pos_peds =  neighbor
                break
    
            #检验入口
    if cell==(enter[0]+1,enter[1]) or cell==(enter[0],enter[1]-1):
        if enter not in peds:  #入口处没人才可以进去
            pos_peds = enter

    return pos_peds

#行人路径选择区拥堵参数
def jam(peds):
    jam1 = 0  #路径1
    jam2 = 0  #路径2
    for i in range(1,slight+1):
        if (enter[0]-i,enter[1])  in peds:
            jam1 +=1
        if (enter[0],enter[1]+i)  in peds:
            jam2 +=1

    return jam1/slight,jam2/slight

#行人移动速度参数
def avg_speed(speed1,speed2):
    v1 = 0
    v2 = 0
    n1 = 0
    n2 = 0

    for i in range(slight):
        if speed1[i] !=0 :
            v1 +=speed1[i]
            n1+=1
        if speed2[i] !=0:
            v2 +=speed2[i]
            n2+=1

    if n1==0:
        avg1=0
    else:
        avg1 = v1/n1

    if n2==0:
        avg2=0
    else:
        avg2 = v2/n2
    return avg1,avg2

#判断路径1是否加速
def jamis1(cell,peds):
    if ACC==1:
        if cell==enter:
            index=-1
        else:
            index=route1.index(cell)

        if index>16:
            accis = 0  #不可以加速
        else:
            accis = 1
            for i in range(1,slight-3):
                if route1[index+i] in peds:
                    accis = 0  #不可以加速
                    break
    else:
        accis=0
    return accis  #前方位置空闲才可以加速

#判断路径2是否加速
def jamis(cell,peds):
    if ACC==1:
        if cell==enter:
            index=-1
        else:
            index=route2.index(cell)

        if index>36:
            accis = 0  #不可以加速
        else:
            accis = 1
            for i in range(1,slight-3):
                if route2[index+i] in peds:
                    accis = 0  #不可以加速
                    break
    else:
        accis=0
    return accis  #前方位置空闲才可以加速

#行人路径选择区
def route_peds(peds,follow,velo1,velo2):
    
    sump =0
    jam1,jam2= jam(peds)
   

    p1 = np.exp(kL*r1-kj*jam1+kg*follow[0])*(1-int(route[0] in peds))
    p2 = np.exp(kL*r2-kj*jam2+kg*follow[1])*(1-int(route[1] in peds))

    sump=p1+p2
    r = np.random.rand() * sump
    if sump == 0:  # pedestrian in cell can not move
        route_ = 0  #路径选择为0
        pos_peds=(0,0)
    else:
        r = np.random.rand() * sump
        if r-p1 <= 0:  #选择路径1
            route_ =  1
            follow[0] = 1
            follow[1] = 0
            pos_peds=route1[0]
            velo1+=1
           
        else:
            route_ = 2  
            follow[1] = 1  
            follow[0] =0
            ncell =1 
            if jamis(enter,peds):
                ncell = accer()
            pos_peds = route2[ncell-1]
            velo2+=ncell
            
            '''
            for j in range(1,ncell+1):
                if (enter[0],enter[1]+j) in peds:  
                    tmp_speed2[j-2]=j-1
                    break
                else:
                   pos_peds = (enter[0],enter[1]+j) #不存在行人位置冲突，可以直接更新follow
                   velo2+=1
            '''
    return route_,follow,pos_peds,velo1,velo2

#路径2加速行为
def accer():
    
    r = np.random.rand() 
    for i in range(1,4): 
        r -= acc[i]
        if r <= 0:   
            acc_cell=i
            break
    return acc_cell

#路径1内部行人行走
def route1_pos(cell,peds,velo1):
    i,j = cell
    if cell==route1[-1]:
        if exit not in peds:
            pos = exit          #出口位置
        else:
            pos = (0,0)
    else:    #路径1
        ncell =1
        accis = jamis1(cell,peds)
        index = route1.index(cell)
        if accis== 1 :  #可以加速
            ncell = accer()   #加速参数
            pos = route1[index+ncell]
            velo1+=ncell
            
        else:  #不可以加速
            if route1[index+ncell] in peds:
                pos = (0,0)
            else:
                pos = route1[index+ncell]
                velo1+=ncell
            
    return pos,velo1

def route2_pos(cell,peds,velo2):
    i,j = cell

    if cell==route2[-1]:
        if exit not in peds:
            pos = exit          #出口位置
        else:
            pos = (0,0)
    else:
        ncell =1
        accis = jamis(cell,peds)
        index = route2.index(cell)
        if accis== 1 :
            ncell = accer()   #加速参数
            pos = route2[index+ncell]
            velo2+=ncell
   
        else:
            if route2[index+ncell] in peds:
                pos = (0,0)
            else:
                pos = route2[index+ncell]
                velo2+=ncell
               
    return pos,velo2 


#计算所有行人的目标位置
def position(peds,dff,follow,route_choice):

    pos_peds = [] #用于存储行人的目标位置坐标
  
    velo1=0
    velo2=0
    npeds1=0
    npeds2=0
    pos_peds=[]
    route_=0

    for n in range(npeds):
        cell = peds[n]
        if cell in sim1+sim0:          #疏散区行人
            pos_peds.append(sim1_peds(cell,peds,dff))

        elif cell == enter:                       #入口处行人,不存在行人位置冲突
            route_,follow,pos,velo1,velo2 =route_peds(peds,follow,velo1,velo2)
            
            if route_==1:
                npeds1+=1
            elif route_==2:
                npeds2+=1
            pos_peds.append(pos)
            for m in range(MAX_STEPS):
                if route_choice[n,m] ==0:
                    route_choice[n,m] = route_
                    break  #插入行人路径选择矩阵
        elif cell == exit:                         #出口行人
            if (exit[0],exit[1]-1) not in peds:
                pos_peds.append((exit[0],exit[1]-1))
            elif (exit[0]-1,exit[1]) not in peds:
                pos_peds.append((exit[0]-1,exit[1]))
            else:
                pos_peds.append((0,0))
                
        else:     #路径内部行人
            if cell in route1:
                npeds1 += 1
                pos,velo1=route1_pos(cell,peds,velo1)
                pos_peds.append(pos)
            elif cell in route2:
                npeds2 += 1 
                pos,velo2=route2_pos(cell,peds,velo2)
                pos_peds.append(pos)
    if npeds1!=0:
        v1=round(velo1/npeds1,2)
    else:
        v1 = 0.00
    if npeds2 !=0:
        v2=round(velo2/npeds2,2)
    else:
        v2 = 0.00

    print('\t行人路径选择：',route_)
    #print('\t行人路径选择：',route_,'\t路径1行人平均速度：',v1,'\t路径2行人平均速度：',v2)
    
    return pos_peds,follow,v1,v2,route_,route_choice

def rc(pos_peds,value,peds):
    index = []
    number = np.shape(pos_peds)[0]
    for n in range(number):  
        if pos_peds[n]== value:
            index.append(n)   
    return index

#处理行人位置冲突（行人移动）
def peds_move(pos_peds,peds,dff):
    tmp_peds = copy.deepcopy(peds)

    pos_peds3 = np.array(pos_peds)   #pos_peds3是数组，pos_peds是二维列表
    x=pos_peds3[:,0]+pos_peds3[:,1]*1j   #换成虚数
    y,num=np.unique(x,return_counts = True) 
    s = np.shape(y)
    z=[]
    con = 0  #位置冲突人数
    
    '''
    s1 = rc(pos_peds,(0,0),peds)       #静止行人位置坐标索引
    static += len(s1)
    '''
    #如果多个行人选择同一个目标元胞，处理位置冲突,只能选择空位。
    for i in range(s[0]):
        if num[i] >= 2 and y[i] !=0:
            con +=1
            z.append((int(y[i].real),int(y[i].imag)))    #注意换成整数

    for (i,j) in z:                     #存储冲突目标位置的坐标（i,j）的 z列表
        r_c =rc(pos_peds,(i,j),peds)       #目标位置是（i,j）的行人位置索引r_c
        '''
        static += len(r_c)-1
        '''
        xy = random.choice(r_c)   #目标位置空闲，等概率随机选取一个行人占据，其他行人不动
        tmp_peds[xy] = (i,j)
        (p,q) = peds[xy]
        if (p,q) in sim1 or (p,q) in sim0:  
            dff[p,q] +=1 # 更新动态场域
        for m in r_c: 
            pos_peds[m] = (0,0)  #已经判断移动的行人目标位置置为0

    #处理没有多目标选同一个元胞
    for m in range(npeds):  
        if pos_peds[m] == (0,0):
            continue
        tmp_peds[m] =  pos_peds[m]        #移动到空位
        (p,q) = peds[m]
        if (p,q) in sim1+sim0:  
            dff[p,q] +=1 # 更新动态场域

    return tmp_peds,dff

def people_init(init_people):
    people =np.zeros((MAX_STEPS+1,npeds,2))
    people[0]= init_people
    return people

def simulation():
    print(" %d peds walk and choice in a box= %d  × %d "%(npeds,L,w))
    print('\t---time: %3d ' % 0)
    print('\t 初始化...')
    peds = init_peds() #记得更新人！！空间不变
    peo = people_init(peds)
    dff = init_DFF()
    route_choice= np.zeros((npeds,MAX_STEPS))
    route =[]
    follow = [0,0]
    velo1=[]
    velo2=[]
    
    npeds1=[]
    npeds2 =[]
    s1=0
    s2=0
    plot_peds(peds,0,0,0,0)

    for t in steps:  
        print('\t---time: %3d ' % (t+1))
        pos_peds,follow,v1,v2,route_,route_choice = position(peds,dff,follow,route_choice)
        peds,dff = peds_move(pos_peds,peds,dff)

        route.append(route_)
        dff= update_DFF(dff)
        velo1.append(v1)
        velo2.append(v2)
        peo[t+1] = peds

        if route_==1:
            s1+=1
            npeds1.append(s1)
            npeds2.append(s2)
        elif route_==2:
            s2+=1
            npeds2.append(s2)
            npeds1.append(s1)
        else:
            npeds1.append(s1)
            npeds2.append(s2)

        #plot_peds(peds,t+1,s1,s2,s1+s2)   #画
        
        if round((t+1)/image_time)==(t+1)/image_time:
            plot_peds(peds,t+1,s1,s2,s1+s2)   #画图
        
    print('--------Walking ＆ Choice are over!----------')
    print('路径1选择：',s1,'路径2选择：',s2)  
    print('路径1：路径2 =',round(s1/s2,2))
    print('路径1流量：',round(s1/MAX_STEPS,2),'路径2流量：',round(s2/MAX_STEPS,2))  

    return route,peo,velo1,velo2,route_choice,npeds1,npeds2

sff = SFF()
#plot_SFF(sff)
route,peo,velo1,velo2,route_choice,npeds1,npeds2 = simulation()

###----------------------------------------------制作行人流动画----------------------------------------------

print('----- Movie is making,please wait for a moment......')
movie_flag =  peo

fig = plt.figure(figsize = (19,11))
ax = plt.axes()


#构造开始帧的函数
def movie_init():
    plt.xlim(0.4,L + 0.6)
    plt.ylim(0.4,w + 0.6)
    plt.axis('off')

    #绘制边界
    plt.plot([0.5,L+0.5],[0.5,0.5],color='k',linewidth=3,linestyle='-')
    plt.plot([0.5,lrwidth+0.5],[w1+0.5,w1+0.5],color='k',linewidth=3,linestyle='-')
    plt.plot([L-lrwidth+0.5,L+0.5],[w1+0.5,w1+0.5],color='k',linewidth=3,linestyle='-')
    plt.plot([lrwidth+0.5,L-lrwidth+0.5],[w+0.5,w+0.5],color='dodgerblue',linewidth=3,linestyle='-')
    plt.plot([lrwidth+1.5,L-lrwidth-0.5],[w1-0.5,w1-0.5],color='dodgerblue',linewidth=3,linestyle='-')
    plt.plot([lrwidth+1.5,L-lrwidth-0.5],[w1+0.5,w1+0.5],color='dodgerblue',linewidth=3,linestyle='-')
    plt.plot([lrwidth+1.5,L-lrwidth-0.5],[w-0.5,w-0.5],color='dodgerblue',linewidth=3,linestyle='-')
   
    plt.plot([0.5,0.5],[0.5,w1+0.5],color='k',linewidth=3,linestyle='-')
    plt.plot([L+0.5,L+0.5],[0.5,w1+0.5],color='k',linewidth=3,linestyle='-')
    plt.plot([lrwidth+0.5,lrwidth+0.5],[w1+0.5,w+0.5],color='dodgerblue',linewidth=3,linestyle='-')
    plt.plot([lrwidth+1.5,lrwidth+1.5],[w1+0.5,w-0.5],color='dodgerblue',linewidth=3,linestyle='-')
    plt.plot([L-lrwidth-0.5,L-lrwidth-0.5],[w1+0.5,w-0.5],color='dodgerblue',linewidth=3,linestyle='-')
    plt.plot([L-lrwidth+0.5,L-lrwidth+0.5],[w1+0.5,w+0.5],color='dodgerblue',linewidth=3,linestyle='-')

    #绘制行人疏散区的网格
    for i in range(1,L):
        x = [i+0.5,i+0.5]
        y = [0.5,w1 + 0.5]#竖线
        plt.plot(x,y,color='dimgrey',linewidth=1,linestyle='-')
    for j in range(1,10):
        if j==9:
            x1 = [0.5,lrwidth+1.5]
            x2 = [L-lrwidth-0.5,L+0.5]
            y= [9.5,9.5]
            plt.plot(x1,y,color='dimgrey',linewidth=1,linestyle='-')
            plt.plot(x2,y,color='dimgrey',linewidth=1,linestyle='-')
        else:
            x = [0.5,L + 0.5]
            y = [j+0.5,j+0.5] #横线
            plt.plot(x,y,color='dimgrey',linewidth=1,linestyle='-')

    #绘制行人行走区的网格
    for i in range(10,22):
        x1 = [lrwidth+0.5,lrwidth+1.5]
        x2 = [L-lrwidth-0.5,L-lrwidth+0.5]
        y = [i+0.5,i+0.5]#横线
        plt.plot(x1,y,color='dimgrey',linewidth=1,linestyle='-')
        plt.plot(x2,y,color='dimgrey',linewidth=1,linestyle='-')
    for i in range(lrwidth,L-lrwidth+1):
        y = [w-0.5,w+0.5]
        x = [i+0.5,i+0.5]#竖线
        plt.plot(x,y,color='dimgrey',linewidth=1,linestyle='-')
    
    plt.text(L/2,16,'行人路径选择区域',ha='center',va='center',fontdict=dict(fontsize=28, color='steelblue'))
    plt.text(exit[0],exit[1],'出',ha='center',va='center',fontdict=dict(fontsize=17, color='seagreen'),alpha=0.8)
    plt.text(enter[0],enter[1]+1,'2',ha='center',va='center',fontdict=dict(fontsize=17, color='seagreen'),alpha=0.8)
    plt.text(enter[0]-1,enter[1],'1',ha='center',va='center',fontdict=dict(fontsize=17, color='seagreen'),alpha=0.8)
    plt.text(enter[0],enter[1],'入',ha='center',va='center',fontdict=dict(fontsize=16, color='seagreen'),alpha=0.8)

    peds = movie_flag[0]  
    x = peds[:,0]
    y = peds[:,1]
   
    plt.scatter(x,y,color='tomato',s = dx)
    S = 't: %3.3d | 路径1选择:%2.2d 路径2选择:%2.2d 已选择人数:%2.2d' % (0,0,0,0)
    plt.show()
    return ax  

def animate(n):
    plt.cla()
    plt.xlim(0.4,L + 0.6)
    plt.ylim(0.4,w + 0.6)
    plt.axis('off')

    #绘制边界
    plt.plot([0.5,L+0.5],[0.5,0.5],color='k',linewidth=3,linestyle='-')
    plt.plot([0.5,lrwidth+0.5],[w1+0.5,w1+0.5],color='k',linewidth=3,linestyle='-')
    plt.plot([L-lrwidth+0.5,L+0.5],[w1+0.5,w1+0.5],color='k',linewidth=3,linestyle='-')
    plt.plot([lrwidth+0.5,L-lrwidth+0.5],[w+0.5,w+0.5],color='dodgerblue',linewidth=3,linestyle='-')
    plt.plot([lrwidth+1.5,L-lrwidth-0.5],[w1-0.5,w1-0.5],color='dodgerblue',linewidth=3,linestyle='-')
    plt.plot([lrwidth+1.5,L-lrwidth-0.5],[w1+0.5,w1+0.5],color='dodgerblue',linewidth=3,linestyle='-')
    plt.plot([lrwidth+1.5,L-lrwidth-0.5],[w-0.5,w-0.5],color='dodgerblue',linewidth=3,linestyle='-')
   
    plt.plot([0.5,0.5],[0.5,w1+0.5],color='k',linewidth=3,linestyle='-')
    plt.plot([L+0.5,L+0.5],[0.5,w1+0.5],color='k',linewidth=3,linestyle='-')
    plt.plot([lrwidth+0.5,lrwidth+0.5],[w1+0.5,w+0.5],color='dodgerblue',linewidth=3,linestyle='-')
    plt.plot([lrwidth+1.5,lrwidth+1.5],[w1+0.5,w-0.5],color='dodgerblue',linewidth=3,linestyle='-')
    plt.plot([L-lrwidth-0.5,L-lrwidth-0.5],[w1+0.5,w-0.5],color='dodgerblue',linewidth=3,linestyle='-')
    plt.plot([L-lrwidth+0.5,L-lrwidth+0.5],[w1+0.5,w+0.5],color='dodgerblue',linewidth=3,linestyle='-')

    #绘制行人疏散区的网格
    for i in range(1,L):
        x = [i+0.5,i+0.5]
        y = [0.5,w1 + 0.5]#竖线
        plt.plot(x,y,color='dimgrey',linewidth=1,linestyle='-')
    for j in range(1,10):
        if j==9:
            x1 = [0.5,lrwidth+1.5]
            x2 = [L-lrwidth-0.5,L+0.5]
            y= [9.5,9.5]
            plt.plot(x1,y,color='dimgrey',linewidth=1,linestyle='-')
            plt.plot(x2,y,color='dimgrey',linewidth=1,linestyle='-')
        else:
            x = [0.5,L + 0.5]
            y = [j+0.5,j+0.5] #横线
            plt.plot(x,y,color='dimgrey',linewidth=1,linestyle='-')

    #绘制行人行走区的网格
    for i in range(10,22):
        x1 = [lrwidth+0.5,lrwidth+1.5]
        x2 = [L-lrwidth-0.5,L-lrwidth+0.5]
        y = [i+0.5,i+0.5]#横线
        plt.plot(x1,y,color='dimgrey',linewidth=1,linestyle='-')
        plt.plot(x2,y,color='dimgrey',linewidth=1,linestyle='-')
    for i in range(lrwidth,L-lrwidth+1):
        y = [w-0.5,w+0.5]
        x = [i+0.5,i+0.5]#竖线
        plt.plot(x,y,color='dimgrey',linewidth=1,linestyle='-')
    
    plt.text(L/2,16,'行人路径选择区域',ha='center',va='center',fontdict=dict(fontsize=28, color='steelblue'))
    plt.text(exit[0],exit[1],'出',ha='center',va='center',fontdict=dict(fontsize=17, color='seagreen'),alpha=0.8)
    plt.text(enter[0],enter[1]+1,'2',ha='center',va='center',fontdict=dict(fontsize=17, color='seagreen'),alpha=0.8)
    plt.text(enter[0]-1,enter[1],'1',ha='center',va='center',fontdict=dict(fontsize=17, color='seagreen'),alpha=0.8)
    plt.text(enter[0],enter[1],'入',ha='center',va='center',fontdict=dict(fontsize=16, color='seagreen'),alpha=0.8)


    peds = movie_flag[n+1]
    x = peds[:,0]
    y = peds[:,1]

    plt.scatter(x,y,color='tomato',s = dx)
    S = 't: %3.3d | 路径1选择:%2.2d 路径2选择:%2.2d 已选择人数:%2.2d' % (n+1,npeds1[n],npeds2[n],npeds1[n]+npeds2[n])
    plt.title("%15s" % S,fontsize = 26)
    return ax

ani = animation.FuncAnimation(fig=fig,
                              func=animate,
                              frames=MAX_STEPS,
                              init_func=movie_init,
                              interval=500,
                              blit=False)
gif_name = os.path.join(video_filename, '双层模型1+并行更新CA模拟.gif')
ani.save(gif_name, writer='pillow')
vfc = mp.VideoFileClip(gif_name)
vfc.write_videofile(os.path.join(video_filename, '双层模型1+并行更新CA模拟.mp4'))
os.remove(gif_name)
plt.close()
















'''
#路径内部行人目标位置  不包括入口处行人位置选择
def sim2_peds(cell,peds,tmp_speed1,tmp_speed2,velo1,velo2):
    i,j = cell
    if cell==route1[-1] or cell==route2[-1]:
        if exit not in peds:
            pos = exit          #出口位置
        else:
            pos=(0,0)
    elif cell in route1:    #路径1
        if (i-1,j) in peds:
            pos = (0,0)  #不可移动
        else:
            pos = (i-1,j)
            velo1+=1
            index = route1.index(cell)
            if index<slight:
                tmp_speed1[index]=1
    elif cell in route2:    #路径2
        ncell =1
        accis = jamis(cell,peds)
        index = route2.index(cell)
        if accis== 1 :
            ncell = accer()   #加速参数
        
        if route2[index+ncell] in peds:
            pos = (0,0)  #不可移动
        else:
            pos = route2[index+ncell]
            velo2+=ncell
            if ncell+index<slight:
                tmp_speed2[ncell+index]= ncell

    
    return pos,tmp_speed1,tmp_speed2,velo1,velo2
'''





'''
        if cell in route21:   #路径2第一段
            if (i,j+1) in peds:  #前面有人，不能行走
                pos_peds=(0,0)
            else:
                turn = cell[1]+ncell-w
                if turn <= 0:#未拐弯至第二段
                    for n in range(1,ncell+1):   
                        number = cell[1]-exit[1]+n
                        if (i,j+n) in peds:
                            if number-1 <= slight:           
                                tmp_speed2[number-1]=1
                            break
                        else:
                            pos_peds = (i,j+n)
                            if n==ncell:
                                if number <= slight:           
                                    tmp_speed2[number]=1
                else:  #拐弯到第二段,必不在视野范围之内
                    turn1 = w-cell[1]
                    for n in range(1,turn1+1):
                        if (i,j+n) in peds:
                            break
                        else:
                            pos_peds = (i,j+n)
                     if pos_peds == (i,w):
                        for n in range(1,turn+1):
                            if (i-n,w) in peds:
                                break
                            else:
                                pos_peds = (i-n,w)
        elif cell in route22:  #路径2第二段
            if (i-1,j) in peds:  #前面有人，不能行走
                pos_peds=(0,0)
            else:
                turn = cell[0]-ncell-exit[0]
                if turn <=0:  #未拐弯至第三段
'''
