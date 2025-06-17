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
#from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import datetime
import collections
import csv
import pickle
import os


alpha=18
s_taxi=25

base_price=5.875
a=np.array([0.6,0.8,1,1.2,1.4])

arm_price=base_price*a

df_loc=pd.read_csv("../data/locationID_location_all.csv")

tu_data=df_loc.values[:,6:8]
dist_matrix = np.zeros([df_loc.values[:,6:8].shape[0],df_loc.values[:,6:8].shape[0]])
for i in range(df_loc.values[:,6:8].shape[0]):
    for j in range(df_loc.values[:,6:8].shape[0]):

        azimuth, bkw_azimuth, distance = grs80.inv(tu_data[i,0], tu_data[i,1], tu_data[j,0], tu_data[j,1])
        dist_matrix[i,j]=distance
dist_matrix=dist_matrix*0.001

num_eval=100
place="Bronx"

year=2019
month=7

df = pd.read_csv("../green_tripdata_2019-07.csv",parse_dates=['lpep_pickup_datetime','lpep_dropoff_datetime'])
df=df[['lpep_pickup_datetime','lpep_dropoff_datetime','PULocationID', 'DOLocationID','trip_distance','total_amount']]
df=pd.merge(df, df_loc, how="inner" ,left_on="PULocationID",right_on="LocationID")
df=df[(df["trip_distance"] >10**(-3))&(df["total_amount"] >10**(-3))&(df["borough"] == place)&(df["PULocationID"] < 264)&(df["DOLocationID"] < 264)]
df1=df[['borough','PULocationID', 'DOLocationID','trip_distance','total_amount','lpep_pickup_datetime','lpep_dropoff_datetime']]
df = pd.read_csv("../yellow_tripdata_2019-07.csv",parse_dates=['tpep_pickup_datetime','tpep_dropoff_datetime'])
df=pd.merge(df, df_loc, how="inner" ,left_on="PULocationID",right_on="LocationID")
df=df[(df["trip_distance"] >10**(-3))&(df["total_amount"] >10**(-3))&(df["borough"] == place)&(df["PULocationID"] < 264)&(df["DOLocationID"] < 264)]
df2=df[['borough','PULocationID', 'DOLocationID','trip_distance','total_amount','tpep_pickup_datetime','tpep_dropoff_datetime']]

df_ID=df_loc[df_loc["borough"] == place]
PUID_set=list(set(df_ID.values[:,4]))
DOID_set =list(set(df_ID.values[:,4]))

D_a=np.zeros([0,10+len(PUID_set)+len(DOID_set)+3])
c_a=np.zeros([0,1])

A_0=0
A_1=0
A_2=0
A_3=0
A_4=0
b_0=0
b_1=0
b_2=0
b_3=0
b_4=0

for day_tmp in range(31):
    day=day_tmp+1
    hour_start=10
    hour_end=20
    minute_0=0
    second_0=0
    day_start_time=datetime.datetime(year,month,day,hour_start,minute_0,second_0)
    day_end_time=datetime.datetime(year,month,day,hour_end,minute_0,second_0)

    df1_day=df1[(df1["lpep_pickup_datetime"] > day_start_time)& (df1["lpep_pickup_datetime"] < day_end_time)]
    df2_day=df2[(df2["tpep_pickup_datetime"] > day_start_time)& (df2["tpep_pickup_datetime"] < day_end_time)]

    for tt_tmp in range(120):
        tt=tt_tmp*5
        h,m=divmod(tt,60)
        hour=10+h
        minute=m
        second=0
        set_time=datetime.datetime(year,month,day,hour,minute,second)
        if minute+5<60:
            after_5min=datetime.datetime(year,month,day,hour,minute+5,second)
        else:
            after_5min=datetime.datetime(year,month,day,hour+1,minute+5-60,second)

        df1_minu=df1_day[(df1_day["lpep_pickup_datetime"] > set_time)& (df1_day["lpep_pickup_datetime"] < after_5min)]
        df2_minu=df2_day[(df2_day["tpep_pickup_datetime"] > set_time)& (df2_day["tpep_pickup_datetime"] < after_5min)]

        sdf1_minu=df1_day[(df1_day["lpep_dropoff_datetime"] > set_time)& (df1_day["lpep_dropoff_datetime"] < after_5min)]
        sdf2_minu=df2_day[(df2_day["tpep_dropoff_datetime"] > set_time)& (df2_day["tpep_dropoff_datetime"] < after_5min)]

        df_ex_pu=np.concatenate([df1_minu.values, df2_minu.values])
        df_ex_do=np.concatenate([sdf1_minu.values, sdf2_minu.values])

        df_ex_pu[:,3]=df_ex_pu[:,3]*1.60934

        req_pu_loc_x=tu_data[df_ex_pu[:,1].astype('int64')-1,0].reshape((df_ex_pu.shape[0],1))+np.random.normal(0, 0.00306, (df_ex_pu.shape[0], 1))
        req_pu_loc_y=tu_data[df_ex_pu[:,1].astype('int64')-1,1].reshape((df_ex_pu.shape[0],1))+np.random.normal(0, 0.000896, (df_ex_pu.shape[0], 1))

        work_loc_x=tu_data[df_ex_do[:,1].astype('int64')-1,0].reshape((df_ex_do.shape[0],1))+np.random.normal(0, 0.00306, (df_ex_do.shape[0], 1))
        work_loc_y=tu_data[df_ex_do[:,1].astype('int64')-1,1].reshape((df_ex_do.shape[0],1))+np.random.normal(0, 0.000896, (df_ex_do.shape[0], 1))

        n=df_ex_pu.shape[0]
        m=df_ex_do.shape[0]

        print(n,m)

        T_dist=np.zeros((df_ex_pu.shape[0],df_ex_do.shape[0]))
        for i in range(df_ex_pu.shape[0]):
            for j in range(df_ex_do.shape[0]):
                azimuth, bkw_azimuth, distance = grs80.inv(req_pu_loc_x[i], req_pu_loc_y[i], work_loc_x[j],work_loc_y[j])
                T_dist[i,j]=distance*0.001

        W=np.zeros((df_ex_pu.shape[0],df_ex_do.shape[0]))
        Cost_matrix=np.ones([n+m+2,n+m+2])*np.inf
        for i in range(df_ex_pu.shape[0]):
            for j in range(df_ex_do.shape[0]):
                W[i,j]=-(T_dist[i,j]+df_ex_pu[i,3])/s_taxi*alpha
                Cost_matrix[i,n+j]=-W[i,j]
                Cost_matrix[n+j,i]=W[i,j]

        time_consume=np.zeros([n,1])
        for i in range(n):
            time_consume[i]=(df_ex_pu[i,6]-df_ex_pu[i,5]).seconds

        df_ex_pu=np.hstack([df_ex_pu[:,0:5],time_consume])

        random_arm=[random.randrange(5) for i in range(n)]

        P=arm_price[random_arm]*df_ex_pu[:,3]
        Pr=-2.0/df_ex_pu[:,4]*P+3
        reward=np.zeros(n)

        def value_eval(P,a_sample):
            group1 = range(n)
            group2 = range(n,n+m)
            g_post = nx.Graph()
            g_post.add_nodes_from(group1, bipartite=1)
            g_post.add_nodes_from(group2, bipartite=0)
            for i in range(n):
                if a_sample[i]==1:
                    for j in range(m):
                        val = P[i]+W[i,j]
                        g_post.add_edge(i, j+n, weight=val)
            d_post = nx.max_weight_matching(g_post)
            opt_value=0.0
            for (i, j) in d_post:
                if i>j:
                    jtmp=j
                    j=i-n
                    i=jtmp
                else:
                    j=j-n
                opt_value+=P[i]+W[i,j];
                reward[i]=P[i]+W[i,j]

            return [opt_value,d_post,reward]

        a_vec=np.zeros(n)
        for i in range(n):
            tmp=np.random.rand()
            if tmp < Pr[i]:
                a_vec[i]=1

        [opt_value,d_postost,reward]=value_eval(P,a_vec)

        hour_onehot=np.zeros(10)
        hour_onehot[hour-10]=1
        for i in range(df_ex_pu.shape[0]):
            PUID_onehot=np.zeros(len(PUID_set))
            for j in range(len(PUID_set)):
                if df_ex_pu[i,1]==PUID_set[j]:
                    PUID_onehot[j]=1
            DOID_onehot=np.zeros(len(DOID_set))
            for j in range(len(DOID_set)):
                if df_ex_pu[i,2]==DOID_set[j]:
                    DOID_onehot[j]=1
            X=np.hstack([hour_onehot,PUID_onehot,DOID_onehot,df_ex_pu[i,3],df_ex_pu[i,5]])

            if random_arm[i]==0:
                A_0+=np.outer(X, X)
                b_0+=X*reward[i]
            elif random_arm[i]==1:
                A_1+=np.outer(X, X)
                b_1+=X*reward[i]
            elif random_arm[i]==2:
                A_2+=np.outer(X, X)
                b_2+=X*reward[i]
            elif random_arm[i]==3:
                A_3+=np.outer(X, X)
                b_3+=X*reward[i]
            elif random_arm[i]==4:
                A_4+=np.outer(X, X)
                b_4+=X*reward[i]

with open('A_0_07', 'wb') as web:
  pickle.dump(A_0, web)
with open('b_0_07', 'wb') as web:
  pickle.dump(b_0, web)
with open('A_1_07', 'wb') as web:
  pickle.dump(A_1, web)
with open('b_1_07', 'wb') as web:
  pickle.dump(b_1, web)
with open('A_2_07', 'wb') as web:
  pickle.dump(A_2, web)
with open('b_2_07', 'wb') as web:
  pickle.dump(b_2, web)
with open('A_3_07', 'wb') as web:
  pickle.dump(A_3, web)
with open('b_3_07', 'wb') as web:
  pickle.dump(b_3, web)
with open('A_4_07', 'wb') as web:
  pickle.dump(A_4, web)
with open('b_4_07', 'wb') as web:
  pickle.dump(b_4, web)
