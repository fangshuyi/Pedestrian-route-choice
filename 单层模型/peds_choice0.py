#串行更新原则,单层模型

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

#------------------参数调整区域--------------------------------------------------
global ks,kd,g1,g2,MAX_STEPS,dx,exit,enter,lrwidth,ewidth,decay,slight,rc1,rc2

#静态场域灵敏度参数
ks = 1.5
#动态场域灵敏度参数
kd = 0
# 系统中的人数
npeds = 60
#模拟时间步长
MAX_STEPS = 120 
steps = range(MAX_STEPS)
#路径1选择概率参数
rc1=41
#路径2选择概率参数
rc2=40.68
#生成图片的时间间隔
image_time = 10

#----------------------模拟空间设置-----------------------------------------------
dx = 150  
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
sim11=[]
sim12=[]
for i in range(1,lrwidth+1):
    sim1.append((i,w1))
    sim11.append((i,w1))
    sim1.append((L-i+1,w1))
    sim12.append((L-i+1,w1))

sim0=[]  #疏散区矩形区域
for i,j in it.chain(it.product(range(1,L+1), range(1,w1))):
     sim0.append((i,j))

ch0 = []
for i,j in it.chain(it.product(range(lrwidth,0,-1), range(1,w1+1))):
     ch0.append((i,j))

ch1 = []
for i,j in it.chain(it.product(range(L-lrwidth-2,lrwidth,-1), range(1,w1))):
     ch1.append((i,j))

ch2 = []
for i,j in it.chain(it.product(range(L-lrwidth-1,L+1), range(1,w1))):
     ch1.append((i,j))

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
    #plt.savefig('space.png')
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

#计算静态场域，整个模拟空间，以出口为目标！
def SFF():
    sff = {}
    sff[exit]=100
    sff[enter]=40

    sff[route2[0]]=rc2
    for m in range(1,48):
        (i,j)=route2[m]
        sff[(i,j)]=sff[route2[m-1]]+1

    sff[route1[0]]= rc1
    for m in range(1,24):
        (i,j)=route1[m]
        sff[(i,j)]=sff[route1[m-1]]+1


    for i,j in it.chain(it.product(range(1,L+1), range(1,w1))):
        distance = abs(i-enter[0])+abs(j-enter[1])
        sff[(i,j)]= sff[enter]-distance
    for (i,j) in sim1:  
        if i >= lrwidth+1:
            distance = abs(i-enter[0])+abs(j-enter[1])
            sff[(i,j)]= sff[enter]-distance
        elif i <= lrwidth:
            distance = abs(i-enter[0])+abs(j-enter[1])
            sff[(i,j)]= sff[enter]-distance-2
    
    return sff

def plot_SFF(sff):
    plot_space()
    for k, v in sff.items():
        i,j = k
        plt.text(i,j,v,ha='center',va='center',fontdict=dict(fontsize=12, color='r'))

    plt.savefig('SFF.png')


#整个区域行人邻居，包含返程区以及路径内部
def get_neigh(cell):
    
    neighbors = []
    i, j = cell
    
    if cell==(enter[0]+1,enter[1]) or cell==(enter[0],enter[1]-1):
        neighbors.append(enter)
    elif cell in route1:
        ind= route1.index(cell)
        if ind==23:
            neighbors.append(exit)
        else:
            neighbors.append(route1[ind+1])
    elif cell in route2:
        ind= route2.index(cell)
        if ind==47:
            neighbors.append(exit)
        else:
            neighbors.append(route2[ind+1])  #路径内部不允许后退
    elif cell==enter:
        neighbors.append((enter[0]-1,enter[1]))
        neighbors.append((enter[0],enter[1]+1))
    else:
        neighbors1=[(i-1,j),(i,j+1),(i,j-1),(i+1,j)]
        neighbors=[(i-1,j),(i,j+1),(i,j-1),(i+1,j)]
        for (m,n) in neighbors1:
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


#返程区域行人移动概率
def sim_peds(cell,peds,dff):
    probs ={}
    sump =0
    neis = get_neigh(cell)
    nei= []

    if cell !=exit:
        p0 = np.exp(ks*sff[cell]+kd*dff[cell]) #不存在位置交换
        probs[(0,0)] = p0 
        nei.append((0,0))   #计算原地静止不动概率
        sump += p0

    for neighbor in neis:
        p = np.exp(ks*sff[neighbor]+kd*dff[neighbor])*(1-int(neighbor in peds)) #不存在位置交换
        if p!=0:
            probs[neighbor] = p  
            nei.append(neighbor)
            sump += p 
            
    r = np.random.rand() * sump

    if sump == 0:  # pedestrian in cell can not move
        pos_peds = (0,0)
    else:
        for neighbor in nei: 
                r -= probs[neighbor]
                if r <= 0 :
                    pos_peds =  neighbor      
                    break

    if cell==(enter[0]+1,enter[1]) or cell==(enter[0],enter[1]-1):
        if enter not in peds:  #入口处没人才可以进去
            pos_peds = enter  #替换

    return pos_peds


#串行更新行人移动
def peds_move(peds,dff,npeds1,npeds2):

    tmp_peds = copy.deepcopy(peds)
    tmp_dff = copy.deepcopy(dff)
    cho=0

    #0左侧返程区的行人
    for (i,j) in ch0:
        if (i,j) not in peds:
            continue
        index = peds.index((i,j))
        pos = sim_peds((i,j),tmp_peds,dff)
        if pos !=(0,0):   
            tmp_peds[index]=pos
            tmp_dff[i,j] +=1

    #⑤更新中间返程区行人
    for (i,j) in ch1:         
        if (i,j) not in peds:
            continue
        index = peds.index((i,j))
        pos=sim_peds((i,j),tmp_peds,dff)
        if pos !=(0,0):
            tmp_peds[index]=pos
            tmp_dff[i,j] +=1

    #①更新出口处的行人,更新下一轮模拟的出口选择
    if exit in peds:
        index = peds.index(exit)
        pos = sim_peds(exit,tmp_peds,dff)
        if pos !=(0,0):   
            tmp_peds[index]=pos
            tmp_dff[exit] +=1

    #随机选取更新路径1还是路径2
    r = np.random.rand()
    if r<=0.5:
        #②更新路径1处的行人
        for (i,j) in route1[::-1]:
            if (i,j) not in peds:
                continue
            index = peds.index((i,j))
            pos = sim_peds((i,j),tmp_peds,dff)
            if pos !=(0,0):
                tmp_dff[i,j] +=1
                tmp_peds[index]=pos
                
        #③更新路径2处的行人
        for (i,j) in route2[::-1]:
            if (i,j) not in peds:
                continue
            index = peds.index((i,j))
            pos = sim_peds((i,j),tmp_peds,dff)
            if pos !=(0,0):
                tmp_dff[i,j] +=1
                tmp_peds[index] = pos
                
    else:
         #②更新路径2处的行人
        for (i,j) in route2[::-1]:
            if (i,j) not in peds:
                continue
            index = peds.index((i,j))
            pos = sim_peds((i,j),tmp_peds,dff)
            if pos !=(0,0):
                tmp_peds[index] = pos
                tmp_dff[i,j] +=1
        #③更新路径1处的行人
        for (i,j) in route1[::-1]:
            if (i,j) not in peds:
                continue
            index = peds.index((i,j))
            pos = sim_peds((i,j),tmp_peds,dff)
            if pos !=(0,0):
                tmp_peds[index]=pos
                tmp_dff[i,j] +=1

    #④更新入口处行人的行人
    if enter in peds:    
        index = peds.index(enter)
        pos=sim_peds(enter,tmp_peds,dff)
        if pos !=(0,0):
            tmp_peds[index]=pos
            tmp_dff[enter] +=1
        if pos==route1[0]:
                npeds1+=1
                cho=1
        elif pos==route2[0]:
                npeds2+=1
                cho=2
        else:
                cho=0

    #⑥更新右侧返程区行人
    for (i,j) in ch2+sim12:         
        if (i,j) not in peds:
            continue
        index = peds.index((i,j))
        pos=sim_peds((i,j),tmp_peds,dff)
        if pos !=(0,0):
            tmp_peds[index]=pos
            tmp_dff[i,j] +=1

    print('\t行人路径选择：',cho)
    
    return tmp_peds,tmp_dff,npeds1,npeds2


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
  
    s1=0
    s2=0
    npeds1=[]
    npeds2 =[]
    plot_peds(peds,0,0,0,0)
    for t in steps:  
        print('\t---time: %3d ' % (t+1))
        peds,dff,s1,s2 = peds_move(peds,dff,s1,s2)
        dff= update_DFF(dff)
       
        peo[t+1] = peds
        npeds1.append(s1)
        npeds2.append(s2)
        #plot_peds(peds,t+1,npeds1,npeds2,npeds1+npeds2)   #画图
        
        if round((t+1)/image_time)==(t+1)/image_time:
            plot_peds(peds,t+1,s1,s2,s1+s2)  #画图
        
    print('--------Walking ＆ Choice are over!----------')
    print('路径1选择：',s1,'路径2选择：',s2)  
    print('路径1：路径2 =',round(s1/s2,2))
    print('路径1流量：',round(s1/MAX_STEPS,2),'路径2流量：',round(s2/MAX_STEPS,2))    
    
    return peo,npeds1,npeds2

sff = SFF()
plot_SFF(sff)
peo,npeds1,npeds2 = simulation()

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
gif_name = os.path.join(video_filename, '串行更新+单层模型CA模拟.gif')
ani.save(gif_name, writer='pillow')
vfc = mp.VideoFileClip(gif_name)
vfc.write_videofile(os.path.join(video_filename, '串行更新+单层模型CA模拟.mp4'))
os.remove(gif_name)
plt.close()


