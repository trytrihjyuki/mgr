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

#Setting
#input parameters
args=sys.argv
place=args[1]
day=int(args[2])
time_interval=int(args[3])
time_unit=args[4]
simulataion_range=int(args[5])

#given constant parameters
year=2019
month=10
#allowable error
epsilon=10**-10
#number of simulations to calculate an approximated objective value
num_eval=100
#parameters to set w_ij
alpha=18
s_taxi=25
#parameters for the existing method, LinUCB
UCB_alpha=0.5
base_price=5.875
a=np.array([0.6,0.8,1,1.2,1.4])
arm_price=base_price*a

#make dist_matrix, which includes the distance between each area
df_loc=pd.read_csv("../data/area_information.csv")
tu_data=df_loc.values[:,6:8]
dist_matrix = np.zeros([df_loc.values[:,6:8].shape[0],df_loc.values[:,6:8].shape[0]])
for i in range(df_loc.values[:,6:8].shape[0]):
    for j in range(df_loc.values[:,6:8].shape[0]):
        azimuth, bkw_azimuth, distance = grs80.inv(tu_data[i,0], tu_data[i,1], tu_data[j,0], tu_data[j,1])
        dist_matrix[i,j]=distance
dist_matrix=dist_matrix*0.001

df_ID=df_loc[df_loc["borough"] == place]
PUID_set=list(set(df_ID.values[:,4]))
DOID_set =list(set(df_ID.values[:,4]))

#download trained matrices and vectors for the existing method, LinUCB
with open('../work/learned_matrix_PL/201908_%s/A_0_08' % place, 'rb') as web:
    A_0 = pickle.load(web)
with open('../work/learned_matrix_PL/201909_%s/A_0_09' % place, 'rb') as web:
    A_0 += pickle.load(web)
with open('../work/learned_matrix_PL/201907_%s/A_0_07' % place, 'rb') as web:
    A_0 += pickle.load(web)
A_0+=np.eye(A_0.shape[0])
with open('../work/learned_matrix_PL/201908_%s/A_1_08' % place, 'rb') as web:
    A_1 = pickle.load(web)
with open('../work/learned_matrix_PL/201909_%s/A_1_09' % place, 'rb') as web:
    A_1 += pickle.load(web)
with open('../work/learned_matrix_PL/201907_%s/A_1_07' % place, 'rb') as web:
    A_1 += pickle.load(web)
A_1+=np.eye(A_0.shape[0])
with open('../work/learned_matrix_PL/201908_%s/A_2_08' % place, 'rb') as web:
    A_2 = pickle.load(web)
with open('../work/learned_matrix_PL/201909_%s/A_2_09' % place, 'rb') as web:
    A_2 += pickle.load(web)
with open('../work/learned_matrix_PL/201907_%s/A_2_07' % place, 'rb') as web:
    A_2 += pickle.load(web)
A_2+=np.eye(A_0.shape[0])
with open('../work/learned_matrix_PL/201908_%s/A_3_08' % place, 'rb') as web:
    A_3 = pickle.load(web)
with open('../work/learned_matrix_PL/201909_%s/A_3_09' % place, 'rb') as web:
    A_3 += pickle.load(web)
with open('../work/learned_matrix_PL/201907_%s/A_3_07' % place, 'rb') as web:
    A_3 += pickle.load(web)
A_3+=np.eye(A_0.shape[0])
with open('../work/learned_matrix_PL/201908_%s/A_4_08' % place, 'rb') as web:
    A_4 = pickle.load(web)
with open('../work/learned_matrix_PL/201909_%s/A_4_09' % place, 'rb') as web:
    A_4 += pickle.load(web)
with open('../work/learned_matrix_PL/201907_%s/A_4_07' % place, 'rb') as web:
    A_4 += pickle.load(web)
A_4+=np.eye(A_0.shape[0])
with open('../work/learned_matrix_PL/201908_%s/b_0_08' % place, 'rb') as web:
    b_0 = pickle.load(web)
with open('../work/learned_matrix_PL/201909_%s/b_0_09' % place, 'rb') as web:
    b_0 += pickle.load(web)
with open('../work/learned_matrix_PL/201907_%s/b_0_07' % place, 'rb') as web:
    b_0 += pickle.load(web)
with open('../work/learned_matrix_PL/201908_%s/b_1_08' % place, 'rb') as web:
    b_1 = pickle.load(web)
with open('../work/learned_matrix_PL/201909_%s/b_1_09' % place, 'rb') as web:
    b_1 += pickle.load(web)
with open('../work/learned_matrix_PL/201907_%s/b_1_07' % place, 'rb') as web:
    b_1 += pickle.load(web)
with open('../work/learned_matrix_PL/201908_%s/b_2_08' % place, 'rb') as web:
    b_2 = pickle.load(web)
with open('../work/learned_matrix_PL/201909_%s/b_2_09' % place, 'rb') as web:
    b_2 += pickle.load(web)
with open('../work/learned_matrix_PL/201907_%s/b_2_07' % place, 'rb') as web:
    b_2 += pickle.load(web)
with open('../work/learned_matrix_PL/201908_%s/b_3_08' % place, 'rb') as web:
    b_3 = pickle.load(web)
with open('../work/learned_matrix_PL/201909_%s/b_3_09' % place, 'rb') as web:
    b_3 += pickle.load(web)
with open('../work/learned_matrix_PL/201907_%s/b_3_07' % place, 'rb') as web:
    b_3 += pickle.load(web)
with open('../work/learned_matrix_PL/201908_%s/b_4_08' % place, 'rb') as web:
    b_4 = pickle.load(web)
with open('../work/learned_matrix_PL/201909_%s/b_4_09' % place, 'rb') as web:
    b_4 += pickle.load(web)
with open('../work/learned_matrix_PL/201907_%s/b_4_07' % place, 'rb') as web:
    b_4 += pickle.load(web)

#Preprocessing of the data
hour_start=10
hour_end=20
minute_0=0
second_0=0
day_start_time=datetime.datetime(year,month,day,hour_start-1,minute_0+55,second_0)
day_end_time=datetime.datetime(year,month,day,hour_end,minute_0+5,second_0)
df = pd.read_csv("../data/green_tripdata_2019-10.csv",parse_dates=['lpep_pickup_datetime','lpep_dropoff_datetime'])
df=df[['lpep_pickup_datetime','lpep_dropoff_datetime','PULocationID', 'DOLocationID','trip_distance','total_amount']]
df=pd.merge(df, df_loc, how="inner" ,left_on="PULocationID",right_on="LocationID")
df=df[(df["trip_distance"] >10**(-3))&(df["total_amount"] >10**(-3))&(df["borough"] == place)&(df["PULocationID"] < 264)&(df["DOLocationID"] < 264)&(df["lpep_pickup_datetime"] > day_start_time)& (df["lpep_pickup_datetime"] < day_end_time)]
green_tripdata=df[['borough','PULocationID', 'DOLocationID','trip_distance','total_amount','lpep_pickup_datetime','lpep_dropoff_datetime']]
df = pd.read_csv("../data/yellow_tripdata_2019-10.csv",parse_dates=['tpep_pickup_datetime','tpep_dropoff_datetime'])
df=pd.merge(df, df_loc, how="inner" ,left_on="PULocationID",right_on="LocationID")
df=df[(df["trip_distance"] >10**(-3))&(df["total_amount"] >10**(-3))&(df["borough"] == place)&(df["PULocationID"] < 264)&(df["DOLocationID"] < 264)&(df["tpep_pickup_datetime"] > day_start_time)& (df["tpep_pickup_datetime"] < day_end_time)]
yellow_tripdata=df[['borough','PULocationID', 'DOLocationID','trip_distance','total_amount','tpep_pickup_datetime','tpep_dropoff_datetime']]

#define functions used in the proposed method
def get_target_min_index(min_index, distance, unsearched_nodes):
    start = 0
    while True:
        index = distance.index(min_index, start)
        found = index in unsearched_nodes
        if found:
            return index
        else:
            start = index + 1
def dfs(v, visited):
    for u in edges[v]:
        if u in visited:
            continue
        visited.add(u)
        if matched[u] == -1 or dfs(matched[u], visited):
            matched[u] = v
            return True
    return False

#the function calculateing the objectives value for a given price and acceptance results
def value_eval(P, accept_results):
    group1 = range(n)
    group2 = range(n,n+m)
    g_post = nx.Graph()
    g_post.add_nodes_from(group1, bipartite=1)
    g_post.add_nodes_from(group2, bipartite=0)
    for i in range(n):
        if  accept_results[i]==1:
            for j in range(m):
                val = P[i]+W[i,j]
                g_post.add_edge(i, j+n, weight=val)
    matched_edges = nx.max_weight_matching(g_post)
    opt_value=0.0
    reward=np.zeros(n)
    for (i, j) in matched_edges:
        if i>j:
            jtmp=j
            j=i-n
            i=jtmp
        else:
            j=j-n
        opt_value+=P[i]+W[i,j];
        reward[i]=P[i]+W[i,j]
    return [opt_value,matched_edges,reward]

# Lists to store results
objective_value_proposed_list=[]
objective_value_MAPS_list=[]
objective_value_LinUCB_list=[]

time_proposed_list=[]
time_MAPS_list=[]
time_LinUCB_list=[]

for tt_tmp in range(simulataion_range):
    #make simulation data
    tt=tt_tmp*5
    h,m=divmod(tt,60)
    hour=10+h
    minute=m
    second=0
    set_time=datetime.datetime(year,month,day,hour,minute,second)
    if time_unit=='m':
        if minute+time_interval<60:
            after_time=datetime.datetime(year,month,day,hour,minute+time_interval,second)
        else:
            after_time=datetime.datetime(year,month,day,hour+1,minute+time_interval-60,second)
    if time_unit=='s':
        after_time=datetime.datetime(year,month,day,hour,minute,second+time_interval)
    after_time=datetime.datetime(year,month,day,hour,minute,second+30)
    green_tripdata_target_requesters=green_tripdata[(green_tripdata["lpep_pickup_datetime"] > set_time)& (green_tripdata["lpep_pickup_datetime"] < after_time)]
    yellow_tripdata_target_requesters=yellow_tripdata[(yellow_tripdata["tpep_pickup_datetime"] > set_time)& (yellow_tripdata["tpep_pickup_datetime"] < after_time)]
    green_tripdata_target_taxis=green_tripdata[(green_tripdata["lpep_dropoff_datetime"] > set_time)& (green_tripdata["lpep_dropoff_datetime"] < after_time)]
    yellow_tripdata_target_taxis=yellow_tripdata[(yellow_tripdata["tpep_dropoff_datetime"] > set_time)& (yellow_tripdata["tpep_dropoff_datetime"] < after_time)]
    df_requesters=np.concatenate([green_tripdata_target_requesters.values, yellow_tripdata_target_requesters.values])
    df_taxis=np.concatenate([green_tripdata_target_taxis.values, yellow_tripdata_target_taxis.values])[:,2]
    time_consume=np.zeros([df_requesters.shape[0],1])
    for i in range(df_requesters.shape[0]):
        time_consume[i]=(df_requesters[i,6]-df_requesters[i,5]).seconds
    df_requesters=np.hstack([df_requesters[:,0:5],time_consume])

    #delete data with too little distance traveled
    df_requesters=df_requesters[df_requesters[:,3]>10**-3]
    df_requesters=df_requesters[df_requesters[:,4]>10**-3]
    #Sort by distance in descending order, as required by the existing MAPS method
    df_requesters=df_requesters[np.argsort(df_requesters[:,3])]
    #Convert units of distance to km
    df_requesters[:,3]=df_requesters[:,3]*1.60934

    requester_pickup_location_x=tu_data[df_requesters[:,1].astype('int64')-1,0].reshape((df_requesters.shape[0],1))+np.random.normal(0, 0.00306, (df_requesters.shape[0], 1))
    requester_pickup_location_y=tu_data[df_requesters[:,1].astype('int64')-1,1].reshape((df_requesters.shape[0],1))+np.random.normal(0, 0.000896, (df_requesters.shape[0], 1))
    taxi_location_x=tu_data[df_taxis.astype('int64')-1,0].reshape((df_taxis.shape[0],1))+np.random.normal(0, 0.00306, (df_taxis.shape[0], 1))
    taxi_location_y=tu_data[df_taxis.astype('int64')-1,1].reshape((df_taxis.shape[0],1))+np.random.normal(0, 0.000896, (df_taxis.shape[0], 1))

    n=df_requesters.shape[0]
    m=df_taxis.size

    if n==0 or m==0:
        np.append(objective_value_proposed_list,0)
        np.append(objective_value_MAPS_list,0)
        np.append(objective_value_LinUCB_list,0)
        np.append(time_proposed_list,0)
        np.append(time_MAPS_list,0)
        np.append(time_LinUCB_list,0)
    else:
        c=2/df_requesters[:,4]
        d=3
        #calculate distances between requesters and taxis
        distance_ij=np.zeros((df_requesters.shape[0],df_taxis.size))
        for i in range(df_requesters.shape[0]):
            for j in range(df_taxis.size):
                azimuth, bkw_azimuth, distance = grs80.inv(requester_pickup_location_x[i], requester_pickup_location_y[i], taxi_location_x[j],taxi_location_y[j])
                distance_ij[i,j]=distance*0.001
        #calculate edge weights from distances between requesters and taxis
        W=np.zeros((df_requesters.shape[0],df_taxis.size))
        Cost_matrix=np.ones([n+m+2,n+m+2])*np.inf
        for i in range(df_requesters.shape[0]):
            for j in range(df_taxis.size):
                W[i,j]=-(distance_ij[i,j]+df_requesters[i,3])/s_taxi*alpha
                Cost_matrix[i,n+j]=-W[i,j]
                Cost_matrix[n+j,i]=W[i,j]

        #start proposed method
        start_time=time.time()
        #generate the graph of min-cost flow problem
        G = nx.DiGraph()
        #add nodes
        #u
        G.add_nodes_from(range(n))
        #v
        G.add_nodes_from(range(n,n+m))
        #s
        G.add_node(n+m)
        #t
        G.add_node(n+m+1)

        #set the amount of delta (amount to adjust current flow)
        delta=n

        #Cap_matrix representing the remaining capacity of each edge
        Cap_matrix=np.zeros([n+m+2,n+m+2])
        for i in range(n):
            for j in range(m):
                G.add_edge(i, n+j)
                Cap_matrix[i, n+j]=np.inf
                G.add_edge(n+j,i)

        for i in range(n):
            val = (1/c[i]*(delta**2)-(d/c[i])*delta-0)/delta
            G.add_edge(n+m,i)
            Cap_matrix[n+m,i]=1
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

        #Set quantities that do not match flow constraints (, that is, equations (3)--(5) in our paper)
        excess=np.zeros(n+m+2)
        excess[n+m]=n
        excess[n+m+1]=-n

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
            searched_nodes=[]
            while(len(unsearched_nodes) != 0): #Repeat until there are no more unsearched nodes
                posible_min_distance = math.inf  #dummy number
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
                Cap_matrix[previous_nodes[tmp_node],tmp_node] -= delta
                Cap_matrix[tmp_node,previous_nodes[tmp_node]] += delta
                tmp_node=previous_nodes[tmp_node]

            #Update Cost_matrix
            for i in range(n):
                val = (1/c[i]*((Flow[n+m,i]+delta)**2)-(d/c[i])*(Flow[n+m,i]+delta)-(1/c[i]*(Flow[n+m,i]**2)-(d/c[i])*Flow[n+m,i]))/delta #delta流した時の1単位あたりのコスト
                Cost_matrix[n+m,i]=val
                val = (1/c[i]*((Flow[n+m,i]-delta)**2)-(d/c[i])*(Flow[n+m,i]-delta)-(1/c[i]*(Flow[n+m,i]**2)-(d/c[i])*Flow[n+m,i]))/delta #delta流した時の1単位あたりのコスト
                Cost_matrix[i,n+m]=val

            #Update excess
            excess[start_node]-=delta
            excess[end_node]+=delta

        #Update delta
        delta=0.5*delta
        #Update Cost_matrix
        for i in range(n):
            val = (1/c[i]*((Flow[n+m,i]+delta)**2)-(d/c[i])*(Flow[n+m,i]+delta)-(1/c[i]*(Flow[n+m,i]**2)-(d/c[i])*Flow[n+m,i]))/delta #delta流した時の1単位あたりのコスト
            Cost_matrix[n+m,i]=val
            val = (1/c[i]*((Flow[n+m,i]-delta)**2)-(d/c[i])*(Flow[n+m,i]-delta)-(1/c[i]*(Flow[n+m,i]**2)-(d/c[i])*Flow[n+m,i]))/delta #delta流した時の1単位あたりのコスト
            Cost_matrix[i,n+m]=val

        #2nd and subsequent iterations
        while delta>0.001:
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
                    val = (1/c[i]*((Flow[n+m,i]+delta)**2)-(d/c[i])*(Flow[n+m,i]+delta)-(1/c[i]*(Flow[n+m,i]**2)-(d/c[i])*Flow[n+m,i]))/delta
                    Cost_matrix[n+m,i]=val
                    val = (1/c[i]*((Flow[n+m,i]-delta)**2)-(d/c[i])*(Flow[n+m,i]-delta)-(1/c[i]*(Flow[n+m,i]**2)-(d/c[i])*Flow[n+m,i]))/delta #delta流した時の1単位あたりのコスト
                    Cost_matrix[i,n+m]=val

                if Cost_matrix[i,n+m]-potential[i]+potential[n+m] < -epsilon and Cap_matrix[i,n+m]>=delta:
                    Flow[n+m,i]-=delta
                    excess[n+m] += delta
                    excess[i] -= delta
                    Cap_matrix[n+m,i]+=delta
                    Cap_matrix[i,n+m]-=delta
                    val = (1/c[i]*((Flow[n+m,i]+delta)**2)-(d/c[i])*(Flow[n+m,i]+delta)-(1/c[i]*(Flow[n+m,i]**2)-(d/c[i])*Flow[n+m,i]))/delta
                    Cost_matrix[n+m,i]=val
                    val = (1/c[i]*((Flow[n+m,i]-delta)**2)-(d/c[i])*(Flow[n+m,i]-delta)-(1/c[i]*(Flow[n+m,i]**2)-(d/c[i])*Flow[n+m,i]))/delta #delta流した時の1単位あたりのコスト
                    Cost_matrix[i,n+m]=val

            # j-t
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

            # s-t
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
                    Cap_matrix[previous_nodes[tmp_node],tmp_node] -= delta
                    Cap_matrix[tmp_node,previous_nodes[tmp_node]] += delta
                    tmp_node=previous_nodes[tmp_node]
                for i in range(n):
                    val = (1/c[i]*((Flow[n+m,i]+delta)**2)-(d/c[i])*(Flow[n+m,i]+delta)-(1/c[i]*(Flow[n+m,i]**2)-(d/c[i])*Flow[n+m,i]))/delta #delta流した時の1単位あたりのコスト
                    Cost_matrix[n+m,i]=val
                    val = (1/c[i]*((Flow[n+m,i]-delta)**2)-(d/c[i])*(Flow[n+m,i]-delta)-(1/c[i]*(Flow[n+m,i]**2)-(d/c[i])*Flow[n+m,i]))/delta #delta流した時の1単位あたりのコスト
                    Cost_matrix[i,n+m]=val
                excess[start_node]-=delta
                excess[end_node]+=delta
            delta=0.5*delta
            for i in range(n):
                val = (1/c[i]*((Flow[n+m,i]+delta)**2)-(d/c[i])*(Flow[n+m,i]+delta)-(1/c[i]*(Flow[n+m,i]**2)-(d/c[i])*Flow[n+m,i]))/delta #delta流した時の1単位あたりのコスト
                Cost_matrix[n+m,i]=val
                val = (1/c[i]*((Flow[n+m,i]-delta)**2)-(d/c[i])*(Flow[n+m,i]-delta)-(1/c[i]*(Flow[n+m,i]**2)-(d/c[i])*Flow[n+m,i]))/delta #delta流した時の1単位あたりのコスト
                Cost_matrix[i,n+m]=val

        #Calculate the price corresponding to the flow
        price_proposed=np.zeros(n)
        for i in range(n):
            price_proposed[i]=-(1/c[i])*Flow[n+m,i]+d/c[i]
        computation_time_proposed=time.time()-start_time

        #start MAPS method
        #MAPS considers each area's requester to be the same.
        requester_pickup_location_x_2=tu_data[df_requesters[:,1].astype('int64')-1,0].reshape((df_requesters.shape[0],1))
        requester_pickup_location_y_2=tu_data[df_requesters[:,1].astype('int64')-1,1].reshape((df_requesters.shape[0],1))
        taxi_location_x_2=tu_data[df_taxis.astype('int64')-1,0].reshape((df_taxis.shape[0],1))
        taxi_location_y_2=tu_data[df_taxis.astype('int64')-1,1].reshape((df_taxis.shape[0],1))
        distance_ij_homogeneous=np.zeros((df_requesters.shape[0],df_taxis.size))
        for i in range(df_requesters.shape[0]):
            for j in range(df_taxis.size):
                azimuth, bkw_azimuth, distance = grs80.inv(requester_pickup_location_x_2[i], requester_pickup_location_y_2[i], taxi_location_x_2[j],taxi_location_y_2[j])
                distance_ij_homogeneous[i,j]=distance*0.001
        #set of area IDs where at least one requester exists
        ID_set=list(set(df_requesters[:,1]))
        #parameters to calculate the acceptance rate
        S_0_rate=1.5
        S_a=1/(S_0_rate-1)
        S_b=1+1/(S_0_rate-1)
        #the upperbound and lower bound of the price
        p_max=np.amax(df_requesters[:,4]/df_requesters[:,3])*S_0_rate
        p_min=np.amin(df_requesters[:,4]/df_requesters[:,3])
        #start MAPS algorithm
        start_time1=time.time()
        p_current=np.ones(len(ID_set))*p_max
        current_count=np.zeros(len(ID_set))
        Nr=np.zeros(len(ID_set))
        Nr_max=np.zeros(len(ID_set))
        for i in range(len(ID_set)):
            Nr_max[i]+=sum(df_requesters[:,1]==ID_set[i])
        dr_sum=np.zeros(len(ID_set))

        p_new=np.ones(len(ID_set))*p_max
        new_count=np.zeros(len(ID_set))
        delta_new=np.zeros(len(ID_set))

        num_requseters=len(df_requesters[:,0])
        num_taxis=len(df_taxis)
        edges = [set() for _ in range(num_requseters)]
        matched=[-1]*num_taxis
        matched_r=[-1]*num_requseters
        D_pre=np.zeros(len(ID_set))

        #difference between each price
        d_rate=0.05
        #number of price values
        d_number=int(np.trunc(np.log(p_max/p_min)/np.log(1+d_rate)))+1

        #average number of acceptances for each price
        S=np.ones([len(ID_set),d_number])*np.inf
        for r in range(len(ID_set)):
            p_tmp=p_max
            for k in range(d_number):
                accept_sum=0
                for [o_dist,to_am] in df_requesters[df_requesters[:,1]==ID_set[r]][:,[3,4]]:
                    if -S_a/to_am*p_tmp*o_dist+S_b>0:
                        if -S_a/to_am*p_tmp*o_dist+S_b<1:
                            accept_sum+=-S_a/to_am*p_tmp*o_dist+S_b
                        else:
                            accept_sum+=1
                S[r,k]=accept_sum/len(df_requesters[df_requesters[:,1]==ID_set[r]][:,1])
                p_tmp=p_tmp/(1+d_rate)

        #start first iteration
        #Calculate delta
        for r in range(len(ID_set)):
            dr_sum[r]=np.sum(df_requesters[:,3]*(df_requesters[:,1]==ID_set[r]))
            dr_Nr_sum=0
            r_count=0
            count=0
            while 1:
                if df_requesters[count,1]==ID_set[r]:
                    dr_Nr_sum+=df_requesters[count,3]
                    r_count+=1
                if r_count==Nr[r]+1:
                    Nr_plus_flag=1
                    break
                count+=1
            value_tmp=0.0
            p_tmp=p_max
            d_count=0
            while p_tmp >= p_min:
                C=dr_sum[r]
                D=dr_Nr_sum
                if value_tmp < np.amin([C*(p_tmp-alpha/s_taxi)*S[r,d_count],D*(p_tmp-alpha/s_taxi)]):
                    value_tmp=np.amin([C*(p_tmp-alpha/s_taxi)*S[r,d_count],D*(p_tmp-alpha/s_taxi)])
                    p_opt=p_tmp
                    opt_d_count=d_count
                p_tmp=p_tmp/(1+d_rate)
                d_count+=1
            delta_new[r]=(p_opt-alpha/s_taxi)*S[r,opt_d_count]-(p_current[r]-alpha/s_taxi)*S[r,int(current_count[r])]
            p_new[r]=p_opt
            new_count[r]=opt_d_count
            D_pre[r]=D

        #assume that it is possible to match a taxi with an requester within 2 km
        for i in range(num_requseters):
            for j in range(num_taxis):
                if distance_ij_homogeneous[i,j]<=2.0:
                    edges[i].add(j)

        #Find an augmented path
        feasible_flag=False
        max_index=np.argmax(delta_new)
        for i in range(n):
            if df_requesters[i,1]==ID_set[max_index] and matched_r[i]==-1:
                feasible_flag=dfs(i, set())
                if feasible_flag==True:
                    matched_r[i]=1
                    break

        #If there is an augmented path, change the match and calculate new delta
        if feasible_flag==True:
            Nr[max_index]+=1
            p_current[max_index]=p_new[max_index]
            current_count[max_index]=new_count[max_index]
            value_tmp=0.0
            p_tmp=p_max
            d_count=0

            #line 18--21 in Algorithm 2 of tong's paper
            if Nr[max_index]+1<=Nr_max[max_index]:
                C=dr_sum[max_index]
                D=0
                sum_num=0
                for i in range(len(df_requesters[:,1])):
                    if sum_num > Nr[max_index]+1:
                        break
                    if df_requesters[i,1]==ID_set[max_index]:
                        D+=df_requesters[i,3]
                        sum_num+=1
                while p_tmp >= p_min:
                    if value_tmp < np.amin([C*(p_tmp-alpha/s_taxi)*S[max_index,d_count],D*(p_tmp-alpha/s_taxi)]):
                        value_tmp=np.amin([C*(p_tmp-alpha/s_taxi)*S[max_index,d_count],D*(p_tmp-alpha/s_taxi)])
                        p_opt=p_tmp
                        opt_d_count=d_count
                    p_tmp=p_tmp/(1+d_rate)
                    d_count+=1
                delta_new[max_index]=(p_opt-alpha/s_taxi)*S[max_index,opt_d_count]-(p_current[max_index]-alpha/s_taxi)*S[max_index,int(current_count[max_index])]
                p_new[max_index]=p_opt
                new_count[max_index]=opt_d_count
            #line 16--17 in Algorithm 2 of tong's paper
            else:
                delta_new[max_index]=-1
                p_new[max_index]=-1
                new_count[max_index]=-1
        else:
            delta_new[max_index]=-1

        #second and subsequent iterations
        iter_num=0
        while 1:
            feasible_flag=False
            max_index=np.argmax(delta_new)
            if delta_new[max_index]<=0:
                break
            for i in range(n):
                if df_requesters[i,1]==ID_set[max_index] and matched_r[i]==-1:
                    feasible_flag=dfs(i, set())
                    if feasible_flag==True:
                        matched_r[i]=1
                    break
            if feasible_flag==True:
                Nr[max_index]+=1
                p_current[max_index]=p_new[max_index]
                current_count[max_index]=new_count[max_index]
                if Nr[max_index]+1<=Nr_max[max_index]:
                    value_tmp=0.0
                    p_tmp=p_max
                    d_count=0
                    C=dr_sum[max_index]
                    D=0
                    sum_num=0
                    for i in range(len(df_requesters[:,1])):
                        if sum_num > Nr[max_index]+1 or sum_num > Nr_max[max_index]:
                            break
                        if df_requesters[i,1]==ID_set[max_index]:
                            D+=df_requesters[i,3]
                            sum_num+=1
                    while p_tmp >= p_min:
                        if value_tmp < np.amin([C*(p_tmp-alpha/s_taxi)*S[max_index,d_count],D*(p_tmp-alpha/s_taxi)]):
                            value_tmp=np.amin([C*(p_tmp-alpha/s_taxi)*S[max_index,d_count],D*(p_tmp-alpha/s_taxi)])
                            p_opt=p_tmp
                            opt_d_count=d_count
                        p_tmp=p_tmp/(1+d_rate)
                        d_count+=1
                    delta_new[max_index]=(p_opt-alpha/s_taxi)*S[max_index,opt_d_count]-(p_current[max_index]-alpha/s_taxi)*S[max_index,int(current_count[max_index])]
                    p_new[max_index]=p_opt
                    new_count[max_index]=opt_d_count
                    D_pre[max_index]=D
                else:
                    delta_new[max_index]=-1
                    p_new[max_index]=-1
                    new_count[max_index]=-1
            else:
                delta_new[max_index]=-1
            iter_num+=1

        #set the price
        price_MAPS=np.zeros(n)
        for i in range(n):
            r=df_requesters[i,1]
            for h in range(len(ID_set)):
                if ID_set[h]==r:
                    price_MAPS[i]=p_current[h]*df_requesters[i,3]
                    break
        computation_time_MAPS=time.time()-start_time1

        #start LinUCB
        start_time2=time.time()

        df_ID=df_loc[df_loc["borough"] == place]
        PUID_set=list(set(df_ID.values[:,4]))
        DOID_set =list(set(df_ID.values[:,4]))

        price_LinUCB=np.zeros([n,1])
        Acceptance_probability_LinUCB=np.zeros([n,1])

        hour_onehot=np.zeros(10)
        hour_onehot[hour-10]=1
        max_arm=np.ones(n)*(-1)

        #line 4 in Algorithm 1 of chu's paper
        theta_0=np.linalg.solve(A_0,b_0)
        theta_1=np.linalg.solve(A_1,b_1)
        theta_2=np.linalg.solve(A_2,b_2)
        theta_3=np.linalg.solve(A_3,b_3)
        theta_4=np.linalg.solve(A_4,b_4)

        #make features X (line 5 in Algorithm 1 of chu's paper)
        for i in range(df_requesters.shape[0]):
            PUID_onehot=np.zeros(len(PUID_set))
            for j in range(len(PUID_set)):
                if df_requesters[i,1]==PUID_set[j]:
                    PUID_onehot[j]=1
            DOID_onehot=np.zeros(len(DOID_set))
            for j in range(len(DOID_set)):
                if df_requesters[i,2]==DOID_set[j]:
                    DOID_onehot[j]=1
            X=np.hstack([hour_onehot,PUID_onehot,DOID_onehot,df_requesters[i,3],df_requesters[i,5]])

            #line 7 and 9 in Algorithm 1 of chu's paper
            p_0=np.inner(theta_0,X)+UCB_alpha*np.sqrt(np.inner(X,np.linalg.solve(A_0,X)))
            max_p=p_0
            max_arm[i]=0
            p_1=np.inner(theta_1,X)+UCB_alpha*np.sqrt(np.inner(X,np.linalg.solve(A_1,X)))
            if p_1>max_p:
                max_p=p_1
                max_arm[i]=1
            p_2=np.inner(theta_2,X)+UCB_alpha*np.sqrt(np.inner(X,np.linalg.solve(A_2,X)))
            if p_2>max_p:
                max_p=p_2
                max_arm[i]=2
            p_3=np.inner(theta_3,X)+UCB_alpha*np.sqrt(np.inner(X,np.linalg.solve(A_3,X)))
            if p_3>max_p:
                max_p=p_3
                max_arm[i]=3
            p_4=np.inner(theta_4,X)+UCB_alpha*np.sqrt(np.inner(X,np.linalg.solve(A_4,X)))
            if p_4>max_p:
                max_p=p_4
                max_arm[i]=4

            price_LinUCB[i]=arm_price[int(max_arm[i])]*df_requesters[i,3]
            Acceptance_probability_LinUCB[i]=-2.0/df_requesters[i,4]*price_LinUCB[i]+3

        #line 10 in Algorithm 1 of chu's paper
        a_LinUCB=np.zeros(n)
        for i in range(n):
            tmp=np.random.rand()
            if tmp < Acceptance_probability_LinUCB[i]:
                a_LinUCB[i]=1
        [opt_value_LinUCB,matched_edges_LinUCB,reward_LinUCB]=value_eval(price_LinUCB,a_LinUCB)

        #line 11 and 12 in Algorithm 1 of chu's paper
        for i in range(df_requesters.shape[0]):
            PUID_onehot=np.zeros(len(PUID_set))
            for j in range(len(PUID_set)):
                if df_requesters[i,1]==PUID_set[j]:
                    PUID_onehot[j]=1
            DOID_onehot=np.zeros(len(DOID_set))
            for j in range(len(DOID_set)):
                if df_requesters[i,2]==DOID_set[j]:
                    DOID_onehot[j]=1
            X=np.hstack([hour_onehot,PUID_onehot,DOID_onehot,df_requesters[i,3],df_requesters[i,5]])
            if max_arm[i]==0:
                A_0+=np.outer(X, X)
                b_0+=X*reward_LinUCB[i]
            elif max_arm[i]==1:
                A_1+=np.outer(X, X)
                b_1+=X*reward_LinUCB[i]
            elif max_arm[i]==2:
                A_2+=np.outer(X, X)
                b_2+=X*reward_LinUCB[i]
            elif max_arm[i]==3:
                A_3+=np.outer(X, X)
                b_3+=X*reward_LinUCB[i]
            elif max_arm[i]==4:
                A_4+=np.outer(X, X)
                b_4+=X*reward_LinUCB[i]
        computation_time_LinUCB=time.time()-start_time2


        #compute the result for each method
        objective_value_proposed_list=np.zeros(num_eval)
        objective_value_MAPS_list=np.zeros(num_eval)
        objective_value_LinUCB_list=np.zeros(num_eval)

        Acceptance_probability_proposed=-2.0/df_requesters[:,4]*price_proposed+3
        Acceptance_probability_MAPS=-2.0/df_requesters[:,4]*price_MAPS+3

        for k in range(num_eval):
                a_proposed=np.zeros(n)
                a_MAPS=np.zeros(n)
                a_LinUCB=np.zeros(n)
                for i in range(n):
                    tmp=np.random.rand()
                    if tmp < Acceptance_probability_proposed[i]:
                        a_proposed[i]=1
                    if tmp < Acceptance_probability_MAPS[i]:
                        a_MAPS[i]=1
                    if tmp < Acceptance_probability_LinUCB[i]:
                        a_LinUCB[i]=1
                [opt_value,matched_edges,reward]=value_eval(price_proposed,a_proposed)
                objective_value_proposed_list[k]=opt_value
                [opt_value_MAPS,matched_edges_MAPS,reward]=value_eval(price_MAPS,a_MAPS)
                objective_value_MAPS_list[k]=opt_value_MAPS
                [opt_value_LinUCB,matched_edges_LinUCB,reward_LinUCB]=value_eval(price_LinUCB,a_LinUCB)
                objective_value_LinUCB_list[k]=opt_value_LinUCB

        np.append(objective_value_proposed_list,np.average(objective_value_proposed_list))
        np.append(objective_value_MAPS_list,np.average(objective_value_MAPS_list))
        np.append(objective_value_LinUCB_list,np.average(objective_value_LinUCB_list))

        time_proposed_list.append(computation_time_proposed)
        time_MAPS_list.append(computation_time_MAPS)
        time_LinUCB_list.append(computation_time_LinUCB)

    print(tt_tmp+1, '/', simulataion_range, 'iterations end')

with open('../results/Average_result_PL_place=%s_day=%s_interval=%f%s.csv'%(place, day, time_interval ,time_unit), mode='w') as Flow:
    writer = csv.writer(Flow)
    writer.writerow([np.average(objective_value_proposed_list),np.average(time_proposed_list)])
    writer.writerow([np.average(objective_value_MAPS_list),np.average(time_MAPS_list)])
    writer.writerow([np.average(objective_value_LinUCB_list),np.average(time_LinUCB_list)])
