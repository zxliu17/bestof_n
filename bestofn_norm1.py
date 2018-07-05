# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 02:39:33 2018

@author: liuzx17
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import copy
import time
import bnfunctions as bnfc
Q = {'H0':1,'H1': 12, 'H2': 7, 'H3': 8, 'H4': 6,'H5': 3,'H6': 2,'H7': 4,'H8':5,'H9':1}
grid = [{'H0'},{"H1"},{'H2'},{'H3'},{'H4'},{'H5'},{'H6'},{'H7'},{'H8'},{'H9'}]
map = {'H0':(1,1),'H1':(1,2),'H2': (1,3), 'H3': (2,1), 'H4': (2,2),'H5': (2,3),'H6': (3,1),'H7': (3,2),'H8':(3,3),'H9':(3,4)}
mapx = {'H0':1,'H1':2,'H2': 3, 'H3': 4, 'H4': 5,'H5': 6,'H6': 7,'H7': 8,'H8':9,'H9':10}
mapy = {'H0':1,'H1':1,'H2': 1, 'H3': 1, 'H4': 1,'H5': 1,'H6': 1,'H7': 1,'H8':1,'H9':1}
iter = 3000
agent_number =200
N=100
figurename = 'norm1_toss'+str(agent_number)+'_'+str(iter)+'_'+str(N)

path = 'figsNorm1/'

def iteration(agents,agent_number, iteration_times):
    cardinality = []
    P_choosing_best =[] 
    #an = agent_number
    #sn = proposition_number
    N = iteration_times
    iteration=0 # iteration time count
    while iteration < N:
        reward = bnfc.get_reward(agents,Q)
        iteration =iteration +1 
        square_rank = bnfc.get_sq_rank(reward)
        index1 = bnfc.pick_agent(agents,square_rank)
        index2 = bnfc.pick_agent(agents,square_rank)
        #t = agents [index1]
        #s = agents [index2]
        Intersection = agents[index1]&agents[index2]
        Union = agents[index1]|agents[index2]
    #distance = hammingdis(s,t) # check if overlap exists
        if (Intersection == set()) : 
                   agents [index1] =Union 
                   agents [index2] =Union
    
        else:
                   agents [index1]=Intersection#intersect if not
                   agents [index2]=Intersection
        cardinality.append(bnfc.cal_card(agents))
        proportion=bnfc.get_proportion(grid,agents)       
        P_choosing_best.append(proportion[1])
    return agents, P_choosing_best,cardinality

start1 = time.time()
times = 0     

bestPstore = []
Pstore =[]
cardstore=[]
#correct = 0
while times < N:
    print(times)
    agents = bnfc.random_initialise_toss(agent_number, grid)
    #print(agents)
    rd = bnfc.get_reward(agents,Q)
    #bnfc.plot(rd,path,figurename,grid,mapx,mapy)
    #plot(agents,get_proportion(grid,agents))
    ##print (agents)
    ##print(get_reward(agents,Q))
    ##reward = get_reward(agents,Q)
    ##print(pick_agent(agents,reward))
    #agents,P1 = iteration(agents,len(agents),500)
    #plot(agents,get_proportion(grid,agents))
    #agents,P2 = iteration(agents,len(agents),500)
    #plot(agents,get_proportion(grid,agents))
    agents,P1,cardinality = iteration(agents,len(agents),iter)
    proportion = bnfc.get_proportion(grid,agents)
    Pstore.append(proportion)
    bestPstore.append(P1)
    cardstore.append(cardinality)
    times =times+1
#print(Ave_P)
sumcard = np.sum(cardstore,axis = 0)
stdcard = np.std(cardstore,axis=0)
ave_card = sumcard/len(cardstore)

sumPbest = np.sum(bestPstore,axis = 0)
stdPbest = np.std(bestPstore,axis=0)
#index = list(range(0,3000))
ave_pbest = sumPbest/len(bestPstore)

sumP = np.sum(Pstore,axis = 0)
ave_p = sumP/len(Pstore)
Pstore2 = copy.deepcopy(Pstore)
percent = [0]*len(Pstore2)
for i in range(len(Pstore2)):
    for j in range(len(Pstore2[i])):
        if  Pstore2[i][j] >=0.8:
            Pstore2[i][j]=1
        else:
            Pstore2[i][j]=0
sumPall = np.sum(Pstore2,axis = 0)


correctBest =[]
for item in bestPstore:
    if item[-1] >=0.8:
        correctBest.append(item)
sumCbest = np.sum(correctBest,axis = 0)
stdCbest = np.std(correctBest,axis=0)      
aveCbest = sumCbest/len(correctBest)     
   
j=0
ave_pbest_f = []
stdPbest_f= []
aveCbest_f=[]
stdCbest_f =[]
ave_card_f=[]
stdcard_f = []
index = []
while j < len(stdPbest):

    index.append(j)
    ave_pbest_f.append(ave_pbest[j])
    stdPbest_f.append(stdPbest[j])
    aveCbest_f.append(aveCbest[j])
    stdCbest_f.append(stdCbest[j])
    ave_card_f.append(ave_card[j])
    stdcard_f.append(stdcard[j])
    j = j+100

#average probability to choose best
print(ave_pbest)

font = {'family' : 'serif',#'sans-serif':['Computer Modern Sans serif'],#Times New Roman',
        'weight' : 'light',
         #'size'   : list(figsize)[1]**1.6,
        }
    
figsize = 6,4

figure1, ax1 = plt.subplots(figsize=figsize)
plt.ylabel('Average probability of choosing the best',font)
plt.xlabel('Iterations',font)
#plt.title('Similarity-Iteration',font)
labels =ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('serif') for label in labels]
plt.plot(ave_pbest)
plt.savefig(path+'Average_pbest'+figurename+'.pdf')
plt.show()

figure2, ax2 = plt.subplots(figsize=figsize)
plt.ylabel('Average probability of choosing the best with error bar',font)
plt.xlabel('Iterations',font)
#plt.title('Similarity-Iteration',font)
labels =ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('serif') for label in labels]
plt.plot(ave_pbest,color = 'brown')
plt.errorbar(index,ave_pbest_f, yerr = stdPbest_f,fmt='.',color = 'brown')
plt.savefig(path+'Average_pbest_errbar'+figurename+'.pdf')
plt.show()

xaxis = list(range(len(ave_p)))
figure3, ax3 = plt.subplots(figsize=figsize)
plt.ylabel('Percentage of each point',font)
plt.xlabel('Points in the grid',font)
#plt.title('Similarity-Iteration',font)
labels =ax3.get_xticklabels() + ax3.get_yticklabels()
[label.set_fontname('serif') for label in labels]
plt.ylim(0,1)
for a,b in zip(xaxis,ave_p):
    plt.text(a, b+0.01, '%.2f' % b, ha='center', va= 'bottom',fontsize=7)
plt.xticks(np.arange(-1,10,1))
plt.bar(xaxis,ave_p,color = 'black',width = 0.4) 
plt.savefig(path+'average_percentage'+figurename+'.pdf')
plt.show()

bnfc.plot(ave_p,path,figurename,grid,mapx,mapy)


plotsumPall=[]
plotsumPall = [sumPall[1]]#,sumPall[4]]
rest = N - sum(plotsumPall)
plotsumPall.append(rest)
figure4, ax4 = plt.subplots(figsize=figsize)
#plt.title('Similarity-Iteration',font)
labels =ax4.get_xticklabels() + ax4.get_yticklabels()
[label.set_fontname('serif') for label in labels]
labels = ['H1(the best)','Rest']
colors = ['orange', 'green']
explode = (0.05,0)
plt.pie(plotsumPall,explode,labels,colors = colors,autopct = '%3.2f%%',shadow = False,startangle = 90,pctdistance = 0.6)
plt.legend(loc ='upper left', bbox_to_anchor=(-0.2, 1))
plt.savefig(path+'average_percentage'+figurename+'_3'+'.pdf')
plt.show()


#plot the correct situation
figure5, ax5 = plt.subplots(figsize=figsize)
#plt.title('Similarity-Iteration',font)
labels =ax5.get_xticklabels() + ax5.get_yticklabels()
[label.set_fontname('serif') for label in labels]
plt.ylabel('Average probability of choosing the best',font)# only the correct convergence
plt.xlabel('Iterations',font)
plt.plot(aveCbest)
plt.savefig(path+'correct_pbest'+figurename+'.pdf')
plt.show()

figure6, ax6 = plt.subplots(figsize=figsize)
#plt.title('Similarity-Iteration',font)
labels =ax6.get_xticklabels() + ax6.get_yticklabels()
[label.set_fontname('serif') for label in labels]
plt.ylabel('Average probability of choosing the best with error bar',font)# only the correct convergence
plt.xlabel('Iterations',font)
plt.plot(aveCbest,color = 'brown')
plt.errorbar(index,aveCbest_f, yerr = stdCbest_f,fmt='.',color = 'brown')
plt.savefig(path+'correct_pbest_errbar'+figurename+'.pdf')
plt.show()
#print(agents)

figsize = 6,4
figure7, ax7 = plt.subplots(figsize=figsize)
plt.ylabel('Cardinality',font)
plt.xlabel('Iterations',font)
#plt.title('Similarity-Iteration',font)
labels =ax7.get_xticklabels() + ax7.get_yticklabels()
[label.set_fontname('serif') for label in labels]
plt.plot(ave_card)
plt.savefig(path+'card'+figurename+'.pdf')
plt.show()

figure8, ax8 = plt.subplots(figsize=figsize)
plt.ylabel('Cardinality with error bar',font)
plt.xlabel('Iterations',font)
#plt.title('Similarity-Iteration',font)
labels =ax8.get_xticklabels() + ax8.get_yticklabels()
[label.set_fontname('serif') for label in labels]
plt.plot(ave_card,color = 'brown')
plt.errorbar(index,ave_card_f, yerr = stdcard_f,fmt='.',color = 'brown')
plt.savefig(path+'card_errbar'+figurename+'.pdf')
plt.show()

end1 = time.time()
print("Time1 used:",end1-start1) 