#串行更新原则,单层模型！！！

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
            if (m,n) not in sim1+sim0:
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
                    if neighbor not in peds:
                        pos_peds =  neighbor
                    else:
                        pos_peds = (0,0)
                    break

    if cell==(enter[0]+1,enter[1]) or cell==(enter[0],enter[1]-1):
        if enter not in peds:  #入口处没人才可以进去
            pos_peds = enter  #替换

    return pos_peds

#计算所有行人的目标位置
def position(peds,npeds1,npeds2,dff):

    pos_peds = [] #用于存储行人的目标位置坐标
    route_=0

    for n in range(npeds):
        cell = peds[n]
        pos = sim_peds(cell,peds,dff)
        pos_peds.append(pos)
        if cell ==enter:
            if pos==route1[0]:
                npeds1+=1
                route_=1
            elif pos==route2[0]:
                npeds2+=1
                route_=2
            else:
                route_=0

    print('\t行人路径选择：',route_)
    return pos_peds,npeds1,npeds2

def rc(pos_peds,value,peds):
    index = []
    number = np.shape(pos_peds)[0]
    for n in range(number):  
        if pos_peds[n]== value:
            index.append(n)   
    return index

#并行更新行人移动,处理行人位置冲突
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
  
    s1=0
    s2=0
    npeds1=[]
    npeds2 =[]

    plot_peds(peds,0,0,0,0)


    for t in steps:  
        print('\t---time: %3d ' % (t+1))
        pos_peds,s1,s2= position(peds,s1,s2,dff)
        peds,dff =peds_move(pos_peds,peds,dff)
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
gif_name = os.path.join(video_filename, '并行更新+单层模型CA模拟.gif')
ani.save(gif_name, writer='pillow')
vfc = mp.VideoFileClip(gif_name)
vfc.write_videofile(os.path.join(video_filename, '并行更新+单层模型CA模拟.mp4'))
os.remove(gif_name)
plt.close()




