# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 01:38:45 2018

@author: liuzx17
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import time
import pandas as pd
from math import *

def cal_card(agents):
    sumcard = 0;
    N =len(agents)
    for i in range(N):
        sumcard = sumcard + len (agents[i])
    mean_card = sumcard/N
    return mean_card

def random_initialise(agents_number, grid):
    '''
    initialise the agents randomly (with a random number of beliefs)
    '''
    agents=[]
    # initialise the agents
    for num in range(agents_number):
        agents.append(set())
        num_blf = random.randint(1,len(grid)-1)
        k = 0
        while k < num_blf :
            i=random.randint(0,len(grid)-1)
            #print(i)
            agents[-1] = agents[-1]|grid[i]
            k=k+1
            
    return agents
#tossing coin
def random_initialise_toss(agents_number, grid):
    '''
    initialise the agents randomly (with a random number of beliefs)
    '''
    agents=[set()]
    i=0
    # initialise the agents
    while i<agents_number:
        if agents[-1] != set():
             i=i+1
             agents.append(set())
        num_blf = len(grid)
        k = 0
        while k < num_blf :
            x=random.randint(0,1)
            if x ==0:
                agents[-1] = agents[-1]|grid[k]
                #print(agents[-1])
                #print('sad')           
            k = k+1
            
    return agents  

def get_proportion(grid,agents):
    proportion =len(grid)*[0]
    N = len(agents)
    #print(N)
    for i in range(len(grid)):  
        for j in range(len(agents)):
            if agents[j]&grid[i] != set():
                proportion[i]=proportion[i]+1/(len(agents[i])*N)
    #proportion = [x/N for x in proportion]
    return proportion
                
def get_reward(agents,Q):    
    reward =[]
    for i in range(len(agents)):
        #print(agents[i])
        #print(list(agents[i]))
        goal = random.choice(list(agents[i]))
        #print(goal)
        point_reward= Q[goal]
        #for item in agents[i]:
        #    point_reward.append(Q[item])
        
        reward.append(point_reward)
    return reward

def norm_reward(reward):
    maxreward = max(reward)
    normed = [x/maxreward for x in reward]
    return normed

def get_sq_rank(reward):
    obj = pd.Series(reward)
    r = obj.rank(method = 'min')
    r = [x**2 for x in r]
    return r

def Gaussian(mu, sigma, x):
        
        # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
    return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))

def pick_agent(agents,prob):
    index = int(random.random()*len(agents))
    beta = 0
    maxweight = max(prob)
    T = sum(prob)
    for i in range(len(agents)):
        beta +=random.random()*2*maxweight#
        while beta > prob[index]:
            beta -= prob[index]
            index = (index+1)%len(agents)
    return index

def plot(proportion,path, figurename,grid,mapx,mapy):
    font = {'family' : 'serif',#'sans-serif':['Computer Modern Sans serif'],#Times New Roman',
        'weight' : 'light',
         #'size'   : list(figsize)[1]**1.6,
        }
    figsize = 6,4
    figure, ax = plt.subplots(figsize=figsize)
    plt.ylabel('Percentage of each point',font)
    plt.xlabel('Points - Quality',font)
    #plt.title('Similarity-Iteration',font)
    labels =ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('serif') for label in labels]
    group_labels = ['H0-1','H1-12', 'H2-7', 'H3-8', 'H4-6','H5-3','H6-2','H7-4','H8-5','H9-1']
    x= np.array(range(1,len(grid)+1))
    #print(x)
    plt.xticks(x, group_labels, rotation=0)
    plt.yticks([])       
            
    for i in range(len(grid)):
        for item in grid[i]:
            if item =='H1'or item == 'H3':
                plt.scatter(mapx[item],mapy[item],s=proportion[i]*600,label = round(proportion[i],2))
                plt.legend(loc=0, prop={'size': 12})
            else:
            #print(proportion)
                plt.scatter(mapx[item],mapy[item],s=proportion[i]*600)#,label = round(proportion[i],2))#, sta)
        #plt.xlabel('Time (ms)')
        #plt.ylabel('Stimulus')
        #plt.title('Spike-Triggered Average')
        plt.savefig(path+'Average_percentage'+figurename+'_2'+'.pdf')
    #plt.show()
    #print(Q['H0'])