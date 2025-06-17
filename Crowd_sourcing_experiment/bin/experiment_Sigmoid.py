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

# Setting
#input parameters
worker_rate=float(args[1])
task_rate=float(args[2])

#given constant parameters
#number of iterations
num_iteration=100
#parameter of acceptance probability function
beta=1.25
gamma=0.25/math.pi
#allowable error
epsilon=10**-4

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

def val_calc(gamma,beta,to_am,Flow_i,delta,p_flag):
    if p_flag==1:
        if Flow_i!=0:
            if 1-(Flow_i+delta)>0:
                val = (gamma*to_am*np.log((Flow_i+delta)/(1-(Flow_i+delta)))*(Flow_i+delta)+beta*to_am*(Flow_i+delta)-(gamma*to_am*np.log(Flow_i/(1-Flow_i))*Flow_i+beta*to_am*Flow_i))/delta #delta流した時の1単位あたりのコスト
            else:
                val= np.infty
        else:
            if 1-(Flow_i+delta)>0:
                val = (gamma*to_am*np.log((Flow_i+delta)/(1-(Flow_i+delta)))*(Flow_i+delta)+beta*to_am*(Flow_i+delta))/delta
            else:
                val= np.infty
    else:
        if Flow_i-delta>0:
            if Flow_i!=0:
                if 1>Flow_i-delta:
                    val = (gamma*to_am*np.log((Flow_i-delta)/(1-(Flow_i-delta)))*(Flow_i-delta)+beta*to_am*(Flow_i-delta)-(gamma*to_am*np.log(Flow_i/(1-Flow_i))*Flow_i+beta*to_am*Flow_i))/delta #delta流した時の1単位あたりのコスト
                else:
                    val= np.infty
            else:
                if 1-(Flow_i-delta)>0:
                    val = (gamma*to_am*np.log((Flow_i-delta)/(1-(Flow_i-delta)))*(Flow_i-delta)+beta*to_am*(Flow_i-delta))/delta
                else:
                    val= np.infty
        elif Flow_i-delta==0:
            val = (0-(gamma*to_am*np.log(Flow_i/(1-Flow_i))*Flow_i+beta*to_am*Flow_i))/delta
        else:
            val=np.infty
    return val

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
                val = P[i]+W_or[i,j]
                g_post.add_edge(i, j+n, weight=val)
    matched_edges = nx.max_weight_matching(g_post)
    #Allo_rate=len(matched_edges)/m
    objective_value=0.0
    reward=np.zeros(n)
    for (i, j) in matched_edges:
        if i>j:
            jtmp=j
            j=i-n
            i=jtmp
        else:
            j=j-n
        objective_value+=P[i]+W_or[i,j];
        reward[i]=P[i]+W_or[i,j]

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
                val = P+W_or[i,j]
                g_post.add_edge(i, j+n, weight=val)
    matched_edges = nx.max_weight_matching(g_post)
    #Allo_rate=len(matched_edges)/m
    objective_value=0.0
    reward=np.zeros(n)
    for (i, j) in matched_edges:
        if i>j:
            jtmp=j
            j=i-n
            i=jtmp
        else:
            j=j-n
        objective_value+=P+W_or[i,j];
        reward[i]=P+W_or[i,j]

    return [objective_value,matched_edges,reward]


# Lists to store results
objective_value_proposed_list=[]
objective_value_MRP_list=[]
objective_value_CappedUCB_list=[]

time_proposed_list=[]
time_MRP_list=[]
time_cappedUCB_list=[]

for hhh in range(num_iteration):
    Rw=np.random.rand(num_worker)
    Rt=np.random.rand(num_task)
    n=sum(Rw>1-worker_rate)
    m=sum(Rt>1-task_rate)

    refference_price=np.random.rand(n)*0.3+0.1

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
    W_or=W

    #end generating problem

    #start proposed method
    start_time=time.time()

    #generate the graph of min-cost flow problem
    G = nx.DiGraph()

    #add nodes
    # i
    G.add_nodes_from(range(n))
    #j
    G.add_nodes_from(range(n,n+m))
    #s
    G.add_node(n+m)
    #t
    G.add_node(n+m+1)

    #Set quantities that do not match flow constraints (, that is, equations (3)--(5) in our paper)
    excess=np.zeros(n+m+2)
    excess[n+m]=0
    excess[n+m+1]=0

    #set the amout of delta (amount to adjust current flow)
    delta=n+0.5

    #set the amout of delta (amount to adjust current flow)
    Flow=np.zeros([n+m+2,n+m+2])
    potential=np.zeros(n+m+2)

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
            Cap_matrix[i, n+j]=1
            Cap_matrix[n+j,i]=0
            G.add_edge(i, n+j)
            G.add_edge(n+j,i)

    for i in range(n):
        val=np.infty
        Cap_matrix[n+m,i]=1-epsilon
        Cap_matrix[i,n+m]=epsilon
        Cost_matrix[n+m,i]=val
        Cost_matrix[i,n+m]=-val
        excess[i]+=epsilon
        excess[n+m]-=epsilon
        Flow[n+m,i]=epsilon
        Flow[i,n+m]=-epsilon
        G.add_edge(n+m,i)
        G.add_edge(i,n+m)

    for j in range(m):
        Cap_matrix[n+j,n+m+1]=1
        Cap_matrix[n+m+1,n+j]=0
        Cost_matrix[n+j,n+m+1]=0
        Cost_matrix[n+m+1,n+j]=0
        G.add_edge(n+j,n+m+1)
        G.add_edge(n+m+1,n+j)

    G.add_edge(n+m,n+m+1)
    Cap_matrix[n+m,n+m+1]=0
    Cost_matrix[n+m,n+m+1]=0

    G.add_edge(n+m+1,n+m)
    Cap_matrix[n+m+1,n+m]=n
    Cost_matrix[n+m+1,n+m]=0

    #first iteration
    #delta-scaling phase
    for i in range(n):
        for j in range (m):
            if Cost_matrix[i,n+j] <0:
                if Flow[i,n+j]+delta < Cap_matrix[i, n+j]:
                    Flow[i,n+j]+=delta
                    Flow[n+j,i]-=delta
                    excess[i] -= delta
                    excess[n+j] += delta
                    Cap_matrix[i,n+j]-=delta
                    Cap_matrix[n+j,i]+=delta

            if Cost_matrix[n+j,i] <0:
                if Flow[n+j,i]+delta < Cap_matrix[n+j,i]:
                    Flow[i,n+j]-=delta
                    Flow[n+j,i]+=delta
                    excess[i] += delta
                    excess[n+j] -= delta
                    Cap_matrix[i,n+j]+=delta
                    Cap_matrix[n+j,i]-=delta

    # shortest path phase
    while len(list(*np.where(excess >= delta)))>0 and len(list(*np.where(excess <= -delta)))>0:
        start_node=list(*np.where(excess >= delta))[0]
        node_num = n+m+2
        unsearched_nodes = list(range(node_num))
        distance = [math.inf] * node_num
        previous_nodes = [-1] * node_num
        distance[start_node] = 0
        searched_nodes=[]

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
                    distance[neighbor_node] = distance[target_min_index] + Cost_matrix[target_min_index,neighbor_node]-potential[target_min_index]+potential[neighbor_node] # 過去に設定されたdistanceよりも小さい場合はdistanceを更新
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
            Cost_matrix[n+m,i]=val_calc(gamma,beta,refference_price[i],Flow[n+m,i],delta,1)
            Cost_matrix[i,n+m]=val_calc(gamma,beta,refference_price[i],Flow[n+m,i],delta,-1)

        #Update excess
        excess[start_node]-=delta
        excess[end_node]+=delta

    #Update delta
    delta=0.5*delta

    #Update Cost_matrix
    for i in range(n):
        Cost_matrix[n+m,i]=val_calc(gamma,beta,refference_price[i],Flow[n+m,i],delta,1)
        Cost_matrix[i,n+m]=val_calc(gamma,beta,refference_price[i],Flow[n+m,i],delta,-1)

    #2nd and subsequent iterations
    while delta>0.01:
        #delta-scaling phase
        for i in range(n):
            for j in range (m):
                if Cost_matrix[i,n+j]-potential[i]+potential[n+j] < -epsilon and Cap_matrix[i,n+j]>=delta:
                    Flow[i,n+j]+=delta
                    Flow[n+j,i]-=delta
                    excess[i] -= delta
                    excess[n+j] += delta
                    Cap_matrix[i,n+j]-=delta
                    Cap_matrix[n+j,i]+=delta

                if Cost_matrix[n+j,i]-potential[n+j]+potential[i] < -epsilon and Cap_matrix[n+j,i]>=delta:
                    Flow[i,n+j]-=delta
                    Flow[n+j,i]+=delta
                    excess[i] += delta
                    excess[n+j] -= delta
                    Cap_matrix[i,n+j]+=delta
                    Cap_matrix[n+j,i]-=delta

        for i in range(n):
            if Cost_matrix[n+m,i]-potential[n+m]+potential[i] < -epsilon and Cap_matrix[n+m,i]>=delta:
                Flow[n+m,i]+=delta
                Flow[i,n+m]-=delta
                excess[n+m] -= delta
                excess[i] += delta
                Cap_matrix[n+m,i]-=delta
                Cap_matrix[i,n+m]+=delta
                Cost_matrix[n+m,i]=val_calc(gamma,beta,refference_price[i],Flow[n+m,i],delta,1)
                Cost_matrix[i,n+m]=val_calc(gamma,beta,refference_price[i],Flow[n+m,i],delta,-1)

            if Cost_matrix[i,n+m]-potential[i]+potential[n+m] < -epsilon and Cap_matrix[i,n+m]>=delta:
                Flow[n+m,i]-=delta
                Flow[i,n+m]+=delta
                excess[n+m] += delta
                excess[i] -= delta
                Cap_matrix[n+m,i]+=delta
                Cap_matrix[i,n+m]-=delta
                Cost_matrix[n+m,i]=val_calc(gamma,beta,refference_price[i],Flow[n+m,i],delta,1)
                Cost_matrix[i,n+m]=val_calc(gamma,beta,refference_price[i],Flow[n+m,i],delta,-1)

        for j in range (m):
            if -potential[n+j]+potential[n+m+1] < -epsilon and Cap_matrix[n+j,n+m+1]>=delta:
                Flow[n+j,n+m+1]+=delta
                Flow[n+m+1,n+j]-=delta
                excess[n+j] -= delta
                excess[n+m+1] += delta
                Cap_matrix[n+j,n+m+1]-=delta
                Cap_matrix[n+m+1,n+j]+=delta

            if -potential[n+m+1]+potential[n+j] < -epsilon and Cap_matrix[n+m+1,n+j]>=delta:
                Flow[n+j,n+m+1]-=delta
                Flow[n+m+1,n+j]+=delta
                excess[n+m+1] -= delta
                excess[n+j] += delta
                Cap_matrix[n+j,n+m+1]+=delta
                Cap_matrix[n+m+1,n+j]-=delta

        if -potential[n+m]+potential[n+m+1] < -epsilon and Cap_matrix[n+m,n+m+1]>=delta:
            Flow[n+m,n+m+1]+=delta
            Flow[n+m+1,n+m]-=delta
            excess[n+m] -= delta
            excess[n+m+1] += delta
            Cap_matrix[n+m,n+m+1]-=delta
            Cap_matrix[n+m+1,n+m]+=delta

        if -potential[n+m+1]+potential[n+m] < -epsilon and Cap_matrix[n+m+1,n+m]>=delta:
            Flow[n+m,n+m+1]-=delta
            Flow[n+m+1,n+m]+=delta
            excess[n+m+1] -= delta
            excess[n+m] += delta
            Cap_matrix[n+m,n+m+1]+=delta
            Cap_matrix[n+m+1,n+m]-=delta

        while len(list(*np.where(excess >= delta)))>0 and len(list(*np.where(excess <= -delta)))>0:
            start_node=list(*np.where(excess >= delta))[0]
            node_num = n+m+2
            unsearched_nodes = list(range(node_num))
            distance = [math.inf] * node_num
            previous_nodes = [-1] * node_num
            distance[start_node] = 0
            searched_nodes=[]

            while(len(unsearched_nodes) != 0):
                posible_min_distance = math.inf
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
                    if neighbor_node in unsearched_nodes:
                        if distance[neighbor_node] - epsilon > distance[target_min_index] + Cost_matrix[target_min_index,neighbor_node] -potential[target_min_index]+potential[neighbor_node] and Cap_matrix[target_min_index,neighbor_node] >= delta:
                            distance[neighbor_node] = distance[target_min_index] + Cost_matrix[target_min_index,neighbor_node]-potential[target_min_index]+potential[neighbor_node] # 過去に設定されたdistanceよりも小さい場合はdistanceを更新
                            previous_nodes[neighbor_node] =  target_min_index

            for i in range(n+m+2):
                if i in searched_nodes:
                    potential[i] -= distance[i]
                else:
                    potential[i] -= distance[end_node]

            tmp_node=end_node
            x=0
            tmp_kk=0
            while tmp_node!=start_node:
                Flow[previous_nodes[tmp_node],tmp_node]+=delta
                Flow[tmp_node,previous_nodes[tmp_node]]-=delta
                Cap_matrix[previous_nodes[tmp_node],tmp_node] -= delta
                Cap_matrix[tmp_node,previous_nodes[tmp_node]] += delta
                tmp_node=previous_nodes[tmp_node]

                tmp_kk+=1
                if tmp_kk>1000:
                    print(previous_nodes,tmp_node,previous_nodes[tmp_node],previous_nodes[previous_nodes[tmp_node]])
                    error

            for i in range(n):
                Cost_matrix[n+m,i]=val_calc(gamma,beta,refference_price[i],Flow[n+m,i],delta,1)
                Cost_matrix[i,n+m]=val_calc(gamma,beta,refference_price[i],Flow[n+m,i],delta,-1)

            excess[start_node]-=delta
            excess[end_node]+=delta

        delta=0.5*delta

        for i in range(n):
            Cost_matrix[n+m,i]=val_calc(gamma,beta,refference_price[i],Flow[n+m,i],delta,1)
            Cost_matrix[i,n+m]=val_calc(gamma,beta,refference_price[i],Flow[n+m,i],delta,-1)

    #Calculate the price corresponding to the flow
    price_proposed=np.zeros(n)
    for i in range(n):
        price_proposed[i]=-gamma*refference_price[i]*np.log(Flow[n+m,i]/(1-Flow[n+m,i]))-beta*refference_price[i]

    proposed_time=time.time()-start_time
    start_time=time.time()

    #start Myerson reserved price
    #the upperbound and lower bound of the price
    p_max=-0.6
    p_min=-0.1
    #Value of the average of Reward_matrix
    w_ave=0.5895
    #difference between each price
    d_rate=0.05
    #number of price values
    d_number=int(np.trunc(np.log(p_max/p_min)/np.log(1+d_rate)))+1

    #average number of acceptances for each price
    S=np.ones(d_number)*np.inf
    p_tmp=p_max
    for k in range(d_number):
        accept_sum=0
        kk=0
        for i in range(num_worker):
            if Rw[i]>1-worker_rate:
                prob=1-(1/(1+np.exp(-(p_tmp+beta*refference_price[kk])/(gamma*refference_price[kk]))))
                if prob>0:
                    if prob<1:
                        accept_sum+=prob
                    else:
                        accept_sum+=1
                kk+=1
        S[k]=accept_sum/n
        p_tmp=p_tmp/(1+d_rate)

    price_MRP_profit=np.zeros(d_number)
    p_tmp=p_max
    for k in range(d_number):
        price_MRP_profit[k]=(p_tmp+w_ave)*S[k]
        p_tmp=p_tmp/(1+d_rate)

    price_MRP_index=np.argmax(price_MRP_profit)
    price_MRP=p_max/((1+d_rate)**price_MRP_index)

    MRP_time=time.time()-start_time

    #start Capped UCB
    start_time=time.time()

    #the upperbound and lower bound of the price
    p_max=-0.6
    p_min=-0.1
    #Value of the average of Reward_matrix
    w_ave=0.5895
    #difference between each price
    d_rate=0.05
    #number of price values
    d_number=int(np.trunc(np.log(p_max/p_min)/np.log(1+d_rate)))+1

    S=np.ones(d_number)*np.inf
    p_tmp=p_max
    for k in range(d_number):
        accept_sum=0
        kk=0
        for i in range(num_worker):
            if Rw[i]>1-worker_rate:
                prob=1-(1/(1+np.exp(-(p_tmp+beta*refference_price[kk])/(gamma*refference_price[kk]))))
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
        price_cappedUCB_profit[k]=(p_tmp+w_ave)*np.amin([m,n*S[k]])
        p_tmp=p_tmp/(1+d_rate)
    price_cappedUCB_index=np.argmax(price_cappedUCB_profit)
    price_cappedUCB=p_max/((1+d_rate)**price_cappedUCB_index)

    cappedUCB_time=time.time()-start_time

    #evaluate the price of each method
    acceptance_prpbability_proposed=1-(1/(1+np.exp(-(price_proposed+beta*refference_price)/(gamma*refference_price))))
    acceptance_prpbability_MRP=1-(1/(1+np.exp(-(price_cappedUCB+beta*refference_price)/(gamma*refference_price))))
    acceptance_prpbability_cappedUCB=1-(1/(1+np.exp(-(price_MRP+beta*refference_price)/(gamma*refference_price))))

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
    [objective_value_MRP,matched_edges_MRP,reward]=value_eval2(price_MRP,a_list_MRP)
    [objective_value_cappedUCB,matched_edges_cap,reward_cap]=value_eval2(price_cappedUCB,a_list_cap)

    objective_value_proposed_list.append(objective_value)
    objective_value_MRP_list.append(objective_value_MRP)
    objective_value_CappedUCB_list.append(objective_value_cappedUCB)

    time_proposed_list.append(proposed_time)
    time_MRP_list.append(MRP_time)
    time_cappedUCB_list.append(cappedUCB_time)

with open('../results/Average_result_Sigmoid_phi=%f_psi=%f.csv' %(worker_rate,task_rate), mode='w') as f:
    writer = csv.writer(f)
    writer.writerow(['method','objective value','computational time'])
    writer.writerow(['proposed method',np.average(objective_value_proposed_list),np.average(time_proposed_list)])
    writer.writerow(['MRP',np.average(objective_value_MRP_list),np.average(time_MRP_list)])
    writer.writerow(['Capped UCB',np.average(objective_value_CappedUCB_list),np.average(time_cappedUCB_list)])
