import pandas as pd
import numpy as np
import urllib.request
import zipfile
import random
import itertools
import math
import scipy.special as spys
import networkx as nx
from networkx.algorithms import bipartite
import itertools
import time
import pyproj
grs80 = pyproj.Geod(ellps='GRS80')
import shapefile
import shapely
from descartes.patch import PolygonPatch
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import datetime
import collections
import csv
import pickle
import os
import sys

args=sys.argv

#Setting
#input parameters
worker_rate=float(args[1])
task_rate=float(args[2])

#given constant parameters
#number of iterations
num_iteration=1000
#parameter of acceptance probability function
S_0_rate=1.5
#allowable error
epsilon=10**-10

with open('../work/Reward_matrix', 'rb') as web:
    Reward_matrix = pickle.load(web)

df = pd.read_csv("../work/trec-rf10-data.csv")
df_v=df.values
topicID_set=list(set(df_v[:,0]))
workerID_set=list(set(df_v[:,1]))
num_worker=len(workerID_set)
num_task=len(df_v)

def find_value_from_list(lst, value):
    return [i for i, x in enumerate(lst) if x == value]

def get_target_min_index(min_index, distance, unsearched_nodes):
    start = 0
    while True:
        index = distance.index(min_index, start)
        found = index in unsearched_nodes
        if found:
            return index
        else:
            start = index + 1

def value_eval(P,a_sample):
    group1 = range(n)
    group2 = range(n,n+m)
    g_post = nx.Graph()
    g_post.add_nodes_from(group1, bipartite=1)
    g_post.add_nodes_from(group2, bipartite=0)
    for i in range(n):
        if a_sample[i]==1:
            for j in range(m):
                val = -P[i]+W[i,j]
                g_post.add_edge(i, j+n, weight=val)
    matched_edges = nx.max_weight_matching(g_post)
    objective_value=0.0
    reward=np.zeros(n)
    for (i, j) in matched_edges:
        if i>j:
            jtmp=j
            j=i-n
            i=jtmp
        else:
            j=j-n
        objective_value+=-P[i]+W[i,j];
        reward[i]=-P[i]+W[i,j]

    return [objective_value,matched_edges,reward]

def value_eval2(P,a_sample):
    group1 = range(n)
    group2 = range(n,n+m)
    g_post = nx.Graph()
    g_post.add_nodes_from(group1, bipartite=1)
    g_post.add_nodes_from(group2, bipartite=0)
    for i in range(n):
        if a_sample[i]==1:
            for j in range(m):
                val = -P+W[i,j]
                g_post.add_edge(i, j+n, weight=val)
    matched_edges = nx.max_weight_matching(g_post)
    objective_value=0.0
    reward=np.zeros(n)
    for (i, j) in matched_edges:
        if i>j:
            jtmp=j
            j=i-n
            i=jtmp
        else:
            j=j-n
        objective_value+=-P+W[i,j];
        reward[i]=-P+W[i,j]

    return [objective_value,matched_edges,reward]


# Lists to store results
objective_value_proposed_list=[]
objective_value_MRP_list=[]
objective_value_CappedUCB_list=[]

time_proposed_list=[]
time_MRP_list=[]
time_cappedUCB_list=[]

for iteration in range(num_iteration):
    #generate problem
    Rw=np.random.rand(num_worker)
    Rt=np.random.rand(num_task)
    n=sum(Rw>1-worker_rate)
    m=sum(Rt>1-task_rate)

    reference_price=np.random.rand(n)*0.3+0.1

    W=np.zeros([n,m])
    tmp_x=0
    for i in range(num_worker):
        if Rw[i]>1-worker_rate:
            tmp_y=0
            for j in range(num_task):
                if Rt[j]>1-task_rate:
                    topic_k=find_value_from_list(topicID_set,df_v[j,0])
                    W[tmp_x,tmp_y]=Reward_matrix[topic_k[0],i]
                    tmp_y+=1
            tmp_x+=1

    #end generating problem

    #start proposed method
    start_time=time.time()

    #generate the graph of min-cost flow problem
    G = nx.DiGraph()
    #add nodes
    #i
    G.add_nodes_from(range(n))
    #j
    G.add_nodes_from(range(n,n+m))
    #s
    G.add_node(n+m)
    #t
    G.add_node(n+m+1)

    #Set quantities that do not match flow constraints (, that is, equations (3)--(5) in our paper)
    excess=np.zeros(n+m+2)
    excess[n+m]=n
    excess[n+m+1]=-n

    #set the amount of delta (amount to adjust current flow)
    delta=n

    #Matrix representing the cost of increasing the flow by delta
    #The parts related to nodes s and t are calculated later
    Cost_matrix=np.ones([n+m+2,n+m+2])*np.inf
    for i in range(n):
        for j in range(m):
            Cost_matrix[i,n+j]=-W[i,j]
            Cost_matrix[n+j,i]=W[i,j]

    #Matrix representing the remaining capacity of each edge
    Cap_matrix=np.zeros([n+m+2,n+m+2])

    for i in range(n):
        for j in range(m):
            G.add_edge(i, n+j)
            Cap_matrix[i, n+j]=np.inf
            G.add_edge(n+j,i)

    for i in range(n):
        G.add_edge(n+m,i)
        Cap_matrix[n+m,i]=1
        val = ((S_0_rate-1)*reference_price[i]*(delta**2)+reference_price[i]*delta-0)/delta
        Cost_matrix[n+m,i]=val
        G.add_edge(i,n+m)

    for j in range(m):
        G.add_edge(n+j,n+m+1)
        Cap_matrix[n+j,n+m+1]=1
        Cost_matrix[n+j,n+m+1]=0
        G.add_edge(n+m+1,n+j)
        Cost_matrix[n+m+1,n+j]=0

    G.add_edge(n+m,n+m+1)
    Cap_matrix[n+m,n+m+1]=n
    Cost_matrix[n+m,n+m+1]=0

    G.add_edge(n+m+1,n+m)
    Cap_matrix[n+m+1,n+m]=0
    Cost_matrix[n+m+1,n+m]=0

    Flow=np.zeros([n+m+2,n+m+2])
    potential=np.zeros(n+m+2)

    #first iteration
    #delta-scaling phase
    for i in range(n):
        for j in range (m):
            if Cost_matrix[i,n+j] <0:
                Flow[i,n+j]+=delta
                excess[i] -= delta
                excess[n+j] += delta
                Cap_matrix[i,n+j]-=delta
                Cap_matrix[n+j,i]+=delta

    #shortest path phase
    while len(list(*np.where(excess >= delta)))>0 and len(list(*np.where(excess <= -delta)))>0:
        start_node=list(*np.where(excess >= delta))[0]
        node_num = n+m+2
        unsearched_nodes = list(range(node_num))
        distance = [math.inf] * node_num
        previous_nodes = [-1] * node_num
        distance[start_node] = 0
        searched_nodes=[] #nodes with permanent distance

        while(len(unsearched_nodes) != 0): #Repeat until there are no more unsearched nodes
            posible_min_distance = math.inf #dummy number
            #Select the unexplored node with the smallest distance
            for node_index in unsearched_nodes:
                if posible_min_distance > distance[node_index]:
                    posible_min_distance = distance[node_index]
            target_min_index = get_target_min_index(posible_min_distance, distance, unsearched_nodes)
            unsearched_nodes.remove(target_min_index)
            searched_nodes.append(target_min_index)

            if excess[target_min_index] <= -delta:
                end_node=target_min_index
                break

            neighbor_node_list = list(G.succ[target_min_index])
            for neighbor_node in neighbor_node_list:
                if distance[neighbor_node] > distance[target_min_index] + Cost_matrix[target_min_index,neighbor_node] -potential[target_min_index]+potential[neighbor_node] and Cap_matrix[target_min_index,neighbor_node] >= delta:
                    distance[neighbor_node] = distance[target_min_index] + Cost_matrix[target_min_index,neighbor_node]-potential[target_min_index]+potential[neighbor_node]
                    previous_nodes[neighbor_node] =  target_min_index

        #Update potential
        for i in range(n+m+2):
            if i in searched_nodes:
                potential[i] -= distance[i]
            else:
                potential[i] -= distance[end_node]

        #Update Flow
        tmp_node=end_node
        x=0
        while tmp_node!=start_node:
            Flow[previous_nodes[tmp_node],tmp_node]+=delta
            Flow[tmp_node,previous_nodes[tmp_node]]-=delta
            Cap_matrix[previous_nodes[tmp_node],tmp_node] -= delta
            Cap_matrix[tmp_node,previous_nodes[tmp_node]] += delta
            tmp_node=previous_nodes[tmp_node]

        #Update Cost_matrix
        for i in range(n):
            val = ((S_0_rate-1)*reference_price[i]*((Flow[n+m,i]+delta)**2)+reference_price[i]*(Flow[n+m,i]+delta)-((S_0_rate-1)*reference_price[i]*(Flow[n+m,i]**2)+reference_price[i]*Flow[n+m,i]))/delta
            Cost_matrix[n+m,i]=val
            val = ((S_0_rate-1)*reference_price[i]*((Flow[n+m,i]-delta)**2)+reference_price[i]*(Flow[n+m,i]-delta)-((S_0_rate-1)*reference_price[i]*(Flow[n+m,i]**2)+reference_price[i]*Flow[n+m,i]))/delta
            Cost_matrix[i,n+m]=val

        #Update excess
        excess[start_node]-=delta
        excess[end_node]+=delta

    #Update delta
    delta=0.5*delta

    #Update Cost_matrix
    for i in range(n):
            val = ((S_0_rate-1)*reference_price[i]*((Flow[n+m,i]+delta)**2)+reference_price[i]*(Flow[n+m,i]+delta)-((S_0_rate-1)*reference_price[i]*(Flow[n+m,i]**2)+reference_price[i]*Flow[n+m,i]))/delta
            Cost_matrix[n+m,i]=val
            val = ((S_0_rate-1)*reference_price[i]*((Flow[n+m,i]-delta)**2)+reference_price[i]*(Flow[n+m,i]-delta)-((S_0_rate-1)*reference_price[i]*(Flow[n+m,i]**2)+reference_price[i]*Flow[n+m,i]))/delta
            Cost_matrix[i,n+m]=val

    #2nd and subsequent iterations
    while delta>0.001:
        #delta-scaling phase
        for i in range(n):
            for j in range (m):
                if Cost_matrix[i,n+j]-potential[i]+potential[n+j] < -epsilon and Cap_matrix[i,n+j]>=delta:
                    Flow[i,n+j]+=delta
                    excess[i] -= delta
                    excess[n+j] += delta
                    Cap_matrix[i,n+j]-=delta
                    Cap_matrix[n+j,i]+=delta

                if Cost_matrix[n+j,i]-potential[n+j]+potential[i] < -epsilon and Cap_matrix[n+j,i]>=delta:
                    Flow[i,n+j]-=delta
                    excess[i] += delta
                    excess[n+j] -= delta
                    Cap_matrix[i,n+j]+=delta
                    Cap_matrix[n+j,i]-=delta

        for i in range(n):
            if Cost_matrix[n+m,i]-potential[n+m]+potential[i] < -epsilon and Cap_matrix[n+m,i]>=delta:
                Flow[n+m,i]+=delta
                excess[n+m] -= delta
                excess[i] += delta
                Cap_matrix[n+m,i]-=delta
                Cap_matrix[i,n+m]+=delta
                val = ((S_0_rate-1)*reference_price[i]*((Flow[n+m,i]+delta)**2)+reference_price[i]*(Flow[n+m,i]+delta)-((S_0_rate-1)*reference_price[i]*(Flow[n+m,i]**2)+reference_price[i]*Flow[n+m,i]))/delta
                Cost_matrix[n+m,i]=val
                val = ((S_0_rate-1)*reference_price[i]*((Flow[n+m,i]-delta)**2)+reference_price[i]*(Flow[n+m,i]-delta)-((S_0_rate-1)*reference_price[i]*(Flow[n+m,i]**2)+reference_price[i]*Flow[n+m,i]))/delta
                Cost_matrix[i,n+m]=val

            if Cost_matrix[i,n+m]-potential[i]+potential[n+m] < -epsilon and Cap_matrix[i,n+m]>=delta:
                Flow[n+m,i]-=delta
                excess[n+m] += delta
                excess[i] -= delta
                Cap_matrix[n+m,i]+=delta
                Cap_matrix[i,n+m]-=delta
                val = ((S_0_rate-1)*reference_price[i]*((Flow[n+m,i]+delta)**2)+reference_price[i]*(Flow[n+m,i]+delta)-((S_0_rate-1)*reference_price[i]*(Flow[n+m,i]**2)+reference_price[i]*Flow[n+m,i]))/delta
                Cost_matrix[n+m,i]=val
                val = ((S_0_rate-1)*reference_price[i]*((Flow[n+m,i]-delta)**2)+reference_price[i]*(Flow[n+m,i]-delta)-((S_0_rate-1)*reference_price[i]*(Flow[n+m,i]**2)+reference_price[i]*Flow[n+m,i]))/delta
                Cost_matrix[i,n+m]=val

        for j in range (m):
            if -potential[n+j]+potential[n+m+1] < -epsilon and Cap_matrix[n+j,n+m+1]>=delta:
                Flow[n+j,n+m+1]+=delta
                excess[n+j] -= delta
                excess[n+m+1] += delta
                Cap_matrix[n+j,n+m+1]-=delta
                Cap_matrix[n+m+1,n+j]+=delta

            if -potential[n+m+1]+potential[n+j] < -epsilon and Cap_matrix[n+m+1,n+j]>=delta:
                Flow[n+j,n+m+1]-=delta
                excess[n+m+1] -= delta
                excess[n+j] += delta
                Cap_matrix[n+j,n+m+1]+=delta
                Cap_matrix[n+m+1,n+j]-=delta

        if -potential[n+m]+potential[n+m+1] < -epsilon and Cap_matrix[n+m,n+m+1]>=delta:
            Flow[n+m,n+m+1]+=delta
            excess[n+m] -= delta
            excess[n+m+1] += delta
            Cap_matrix[n+m,n+m+1]-=delta
            Cap_matrix[n+m+1,n+m]+=delta

        if -potential[n+m+1]+potential[n+m] < -epsilon and Cap_matrix[n+m+1,n+m]>=delta:
            Flow[n+m,n+m+1]-=delta
            excess[n+m+1] -= delta
            excess[n+m] += delta
            Cap_matrix[n+m,n+m+1]+=delta
            Cap_matrix[n+m+1,n+m]-=delta

        #shortest path phase
        while len(list(*np.where(excess >= delta)))>0 and len(list(*np.where(excess <= -delta)))>0:
            start_node=list(*np.where(excess >= delta))[0]
            node_num = n+m+2
            unsearched_nodes = list(range(node_num))
            distance = [math.inf] * node_num
            previous_nodes = [-1] * node_num
            distance[start_node] = 0
            searched_nodes=[]
            while(len(unsearched_nodes) != 0):
                posible_min_distance = math.inf #dummy number
                for node_index in unsearched_nodes:
                    if posible_min_distance > distance[node_index]:
                        posible_min_distance = distance[node_index]
                target_min_index = get_target_min_index(posible_min_distance, distance, unsearched_nodes)
                unsearched_nodes.remove(target_min_index)
                searched_nodes.append(target_min_index)

                if excess[target_min_index] <= -delta:
                    end_node=target_min_index
                    break

                neighbor_node_list = list(G.succ[target_min_index])
                for neighbor_node in neighbor_node_list:
                    if distance[neighbor_node] - epsilon > distance[target_min_index] + Cost_matrix[target_min_index,neighbor_node] -potential[target_min_index]+potential[neighbor_node] and Cap_matrix[target_min_index,neighbor_node] >= delta:
                        distance[neighbor_node] = distance[target_min_index] + Cost_matrix[target_min_index,neighbor_node]-potential[target_min_index]+potential[neighbor_node]
                        previous_nodes[neighbor_node] =  target_min_index

            for i in range(n+m+2):
                if i in searched_nodes:
                    potential[i] -= distance[i]
                else:
                    potential[i] -= distance[end_node]

            tmp_node=end_node
            x=0
            while tmp_node!=start_node:
                Flow[previous_nodes[tmp_node],tmp_node]+=delta
                Flow[tmp_node,previous_nodes[tmp_node]]-=delta
                Cap_matrix[previous_nodes[tmp_node],tmp_node] -= delta
                Cap_matrix[tmp_node,previous_nodes[tmp_node]] += delta
                tmp_node=previous_nodes[tmp_node]

            for i in range(n):
                val = ((S_0_rate-1)*reference_price[i]*((Flow[n+m,i]+delta)**2)+reference_price[i]*(Flow[n+m,i]+delta)-((S_0_rate-1)*reference_price[i]*(Flow[n+m,i]**2)+reference_price[i]*Flow[n+m,i]))/delta
                Cost_matrix[n+m,i]=val
                val = ((S_0_rate-1)*reference_price[i]*((Flow[n+m,i]-delta)**2)+reference_price[i]*(Flow[n+m,i]-delta)-((S_0_rate-1)*reference_price[i]*(Flow[n+m,i]**2)+reference_price[i]*Flow[n+m,i]))/delta
                Cost_matrix[i,n+m]=val

            excess[start_node]-=delta
            excess[end_node]+=delta

        delta=0.5*delta

        for i in range(n):
            val = ((S_0_rate-1)*reference_price[i]*((Flow[n+m,i]+delta)**2)+reference_price[i]*(Flow[n+m,i]+delta)-((S_0_rate-1)*reference_price[i]*(Flow[n+m,i]**2)+reference_price[i]*Flow[n+m,i]))/delta
            Cost_matrix[n+m,i]=val
            val = ((S_0_rate-1)*reference_price[i]*((Flow[n+m,i]-delta)**2)+reference_price[i]*(Flow[n+m,i]-delta)-((S_0_rate-1)*reference_price[i]*(Flow[n+m,i]**2)+reference_price[i]*Flow[n+m,i]))/delta
            Cost_matrix[i,n+m]=val

    #Calculate the price corresponding to the flow
    price_proposed=np.zeros(n)
    for i in range(n):
        price_proposed[i]=(S_0_rate-1)*reference_price[i]*(Flow[n+m,i])+reference_price[i]
    proposed_time=time.time()-start_time

    #start Capped UCB
    start_time=time.time()

    #the upperbound and lower bound of the price
    p_max=0.6
    p_min=0.1
    #Value of the average of Reward_matrix
    w_ave=0.5895
    #difference between each price
    d_rate=0.05
    #number of price values
    d_number=int(np.trunc(np.log(p_max/p_min)/np.log(1+d_rate)))+1
    #parameters to calculate the acceptance rate
    S_a=1/(S_0_rate-1)
    S_b=1/(1-S_0_rate)

    #average number of acceptances for each price
    S=np.ones(d_number)*np.inf
    p_tmp=p_max
    for k in range(d_number):
        accept_sum=0
        kk=0
        for i in range(num_worker):
            if Rw[i]>1-worker_rate:
                prob=S_a/reference_price[kk]*p_tmp+S_b
                if prob>0:
                    if prob<1:
                        accept_sum+=prob
                    else:
                        accept_sum+=1
                kk+=1
        S[k]=accept_sum/n
        p_tmp=p_tmp/(1+d_rate)

    price_cappedUCB_profit=np.zeros(d_number)
    p_tmp=p_max
    for k in range(d_number):
        price_cappedUCB_profit[k]=(-p_tmp+w_ave)*np.amin([m,n*S[k]])
        p_tmp=p_tmp/(1+d_rate)
    price_cappedUCB_index=np.argmax(price_cappedUCB_profit)
    price_cappedUCB=p_max/((1+d_rate)**price_cappedUCB_index)
    cappedUCB_time=time.time()-start_time

    #Myerson reserved price
    #the upperbound and lower bound of the price
    p_max=0.6
    p_min=0.1
    start_time=time.time()
    #Value of the average of Reward_matrix
    w_ave=0.5895
    #difference between each price
    d_rate=0.05
    #number of price values
    d_number=int(np.trunc(np.log(p_max/p_min)/np.log(1+d_rate)))+1

    #parameters to calculate the acceptance rate
    S_a=1/(S_0_rate-1)
    S_b=1/(1-S_0_rate)

    S=np.ones(d_number)*np.inf
    p_tmp=p_max
    for k in range(d_number):
        accept_sum=0
        kk=0
        for i in range(num_worker):
            if Rw[i]>1-worker_rate:
                prob=S_a/reference_price[kk]*p_tmp+S_b
                if prob>0:
                    if prob<1:
                        accept_sum+=prob
                    else:
                        accept_sum+=1
                kk+=1
        S[k]=accept_sum/n
        p_tmp=p_tmp/(1+d_rate)

    P_MRP_profit=np.zeros(d_number)
    p_tmp=p_max
    for k in range(d_number):
        P_MRP_profit[k]=(-p_tmp+w_ave)*S[k]
        p_tmp=p_tmp/(1+d_rate)

    P_MRP_index=np.argmax(P_MRP_profit)
    P_MRP=p_max/((1+d_rate)**P_MRP_index)

    MRP_time=time.time()-start_time

    #evaluate the price of each method
    S_a=1/(S_0_rate-1)
    S_b=1/(1-S_0_rate)

    acceptance_prpbability_proposed=S_a/reference_price*price_proposed+S_b
    acceptance_prpbability_MRP=S_a/reference_price*P_MRP+S_b
    acceptance_prpbability_cappedUCB=S_a/reference_price*price_cappedUCB+S_b

    a_list_proposed=np.zeros(n)
    a_list_MRP=np.zeros(n)
    a_list_cap=np.zeros(n)

    for i in range(n):
        tmp=np.random.rand()
        if tmp < acceptance_prpbability_proposed[i]:
            a_list_proposed[i]=1
        if tmp < acceptance_prpbability_MRP[i]:
            a_list_MRP[i]=1
        if tmp < acceptance_prpbability_cappedUCB[i]:
            a_list_cap[i]=1
    [objective_value,matched_edges,reward]=value_eval(price_proposed,a_list_proposed)
    [objective_value_MRP,matched_edges_MRP,reward]=value_eval2(P_MRP,a_list_MRP)
    [objective_value_cappedUCB,matched_edges_cap,reward_cap]=value_eval2(price_cappedUCB,a_list_cap)

    objective_value_proposed_list.append(objective_value)
    objective_value_MRP_list.append(objective_value_MRP)
    objective_value_CappedUCB_list.append(objective_value_cappedUCB)

    time_proposed_list.append(proposed_time)
    time_MRP_list.append(MRP_time)
    time_cappedUCB_list.append(cappedUCB_time)

with open('../results/Average_result_PL_phi=%f_psi=%f.csv' %(worker_rate,task_rate), mode='w') as f:
    writer = csv.writer(f)
    writer.writerow(['method','objective value','computational time'])
    writer.writerow(['proposed method',np.average(objective_value_proposed_list),np.average(time_proposed_list)])
    writer.writerow(['MRP',np.average(objective_value_MRP_list),np.average(time_MRP_list)])
    writer.writerow(['Capped UCB',np.average(objective_value_CappedUCB_list),np.average(time_cappedUCB_list)])
