#-*- coding: utf-8 -*-

import os,sys
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram,fcluster,cophenet
from pyqsar.minisom import MiniSom
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score,silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib.patches import RegularPolygon
from matplotlib import cm, colorbar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from math import sqrt
from matplotlib import pyplot as plt
import datetime
import random
import copy
import math
from numpy import ndarray
import sklearn.linear_model as lm
from sklearn import preprocessing, datasets, linear_model 
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from bokeh.plotting import figure, output_file, show, ColumnDataSource, output_notebook
from bokeh.models import HoverTool, BoxSelectTool
from io import StringIO
from ipywidgets.widgets import interact, ToggleButtons
from scipy.stats import pearsonr
from pyqsar import data_setting as ds
import plotly.express as px
from tqdm import notebook
import pickle
import panel as pn
from panel_chemistry.widgets import JSMEEditor
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from padelpy import from_smiles
import datetime
import molplotly
from notebook import notebookapp
from bokeh.plotting import figure, output_file, show, ColumnDataSource, output_notebook
from bokeh.models import HoverTool, Line, Circle
from bokeh.resources import INLINE
from bokeh.transform import factor_cmap
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def split_xy(ext=None):
    df = ds.read_csv(ext,index='ID')
    X = df.iloc[:,2:]
    y = pd.DataFrame(df.loc[:,'EP'])
    
    return X,y

def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)


def cluster_dataframe(X_data,file):
    fr = open(file,'r')
    a1 = fr.readlines()
    fr.close()
    clust_dict = {}
    for data in a1:
        clust_num = data.split('\t')[0].split(' ')[1]
        clust_des = data.split('\t')[1].strip().split(' ')
    
        for i in clust_des:
            clust_dict[i] = int(clust_num)
    
    header = list(X_data.columns.values)
    my_priority = {}
    for i in range(len(header)):
        my_priority[header[i]] = i
    clust_dict_sorted = sorted(clust_dict.items(), key=(lambda x: my_priority[x[0]]))
    feature = []
    cluster = []
    for i in clust_dict_sorted:
        feature.append(i[0])
        cluster.append(i[1])
        
    cluster_output = DataFrame({'Features':feature , 'cluster':cluster})
    
    return cluster_output

class FeatureCluster:
    """
    Make features(decriptors) clusters based on hierarchical clustering method

    Parameters
    ----------
    X_data : pandas DataFrame , shape = (n_samples, n_features)
    link : str , kind of linkage (single or complete or average)
    cut_d : int, depth in cluster(dendrogram)

    Sub functions
    -------
    set_cluster(self)
    cluster_dist(self)
    """
    def __init__(self,X_data,):
        self.cluster_info = []
        self.assignments = 0
        self.X_data = X_data
        self.abs_corre = abs(X_data.corr())


    def cophenetic(self) :
        """
        Calculate cophenetic correlation coefficient of linkages
    
        Parameters
        ----------
        X_data : pandas DataFrame , shape = (n_samples, n_features)
    
        Returns
        -------
        None
        """
        X_data = self.X_data
        abs_corre = abs(X_data.corr())
        Z1 = linkage(abs_corre, method='average')
        Z2 = linkage(abs_corre, method='complete')
        Z3 = linkage(abs_corre, method='single')
        c1, coph_dists1 = cophenet(Z1, pdist(abs_corre))
        c2, coph_dists2 = cophenet(Z2, pdist(abs_corre))
        c3, coph_dists3 = cophenet(Z3, pdist(abs_corre))
        print ('average linkage cophenet:', c1)
        print ('complete linkage cophenet:',c2)
        print ('single linkage cophenet:',c3)
        
    def set_cluster(self,link,cut_d) :
        """
        Make input of feature selection function

        Returns
        -------
        assignments : dic, shape = (n_features)
        return cluster information as a input of feature selection function
        """

        Z = linkage(self.abs_corre, method=link)
        self.assignments = fcluster(Z,cut_d,'distance') - 1

        cluster_output = DataFrame({'Feature':list(self.X_data.columns.values) , 'cluster':self.assignments})
        
        fw = open('%s_hierarchical.cluster'%ds.data_name,'w')
        nc = list(cluster_output.cluster.values)
        nnc = max(nc)
        name = list(cluster_output.Feature.values)
        cludic = {}
        for i in range(0,len(nc)):
            k = name[i]
            v = nc[i]
            cludic[k] = v
        for t in range(0,nnc+1):
            vv = []
            vv = [key for key, value in list(cludic.items()) if value == t]    #Find Key by Value in Dictionary
            self.cluster_info.append(vv)
            print ('\n','\x1b[1;46m'+'Cluster'+'\x1b[0m',t,vv, end=' ')
            fw.write('Cluster %s\t'%t)
            for des in vv:
                fw.write(des+' ')
            fw.write('\n')
        fw.close()
        print('\n')
        print('Cluster info file','\033[1m'+'%s_hierarchical.cluster'%ds.data_name+'\033[0m','file saved')
        return self.assignments

    # cluster correlation coefficient distribution
    def cluster_dist(self) :
        """
        Show dendrogram of correlation coefficient distribution of each cluster

        Returns
        -------
        None
        """
        dist_box = {}

        assignments = self.cluster_info
        cluster_output = DataFrame({'Feature':list(self.X_data.columns.values) , 'cluster':self.assignments})
        nc = list(cluster_output.cluster.values)
        name = list(cluster_output.Feature.values)

        clu_hist  = {}
        cluster= []

        for i in range(0,len(nc)):
            k = name[i]
            v = nc[i]
            clu_hist [k] = v
        for t in range(max(nc)):
            vv = []
            vv = [key for key, value in list(clu_hist.items()) if value == t]    #Find Key by Value in Dictionary
            cluster.append(vv)

        for val, s in enumerate(cluster) :
            desc_set = s
            c = len(desc_set)
            if c == 1 :
                pass
            else:
                tay = self.abs_corre.loc[desc_set,desc_set]
                t =  np.array(tay)

                av = (t.sum()-c)/2
                aver = av/((c*c-c)/2)
                dist_box[aver] = val
        
        dist_array = np.array(list(dist_box.keys()))       
        cnt, bins, patches = plt.hist(dist_box.keys(),edgecolor='black')
        plt.ylabel('Frequency')
        plt.xlabel('Correlation coefficient of each cluster')
        
        buttons = {}
        for num in range(len(cnt)):
            Range = []
            Range.append(bins[num])
            Range.append(bins[num+1])
            buttons[num] = Range
        labels = np.array(list(buttons.keys()))
        index = np.where(cnt==0)[0]
        labels = list(np.delete(labels,index))
        

        bin_centers = np.diff(bins)*0.5 + bins[:-1]
        for fr, x, patch in zip(cnt,bin_centers,patches):
            height = fr
            plt.annotate("{}".format(height),
                        xy = (x, height),
                        xytext = (0,0.2),
                        textcoords = 'offset points',
                        ha = 'center',
                        va = 'bottom'
                        )

        def func(label):
            val_range = buttons[int(label)]
            cluster_index = np.where((dist_array>=val_range[0]) & (dist_array<val_range[1]))
            print('\033[1m'+'Bins : %s\t'%str(val_range) + 'Count :%d'%cnt[int(label)] + '\033[0m')
        
            for i in dist_array[cluster_index]:
                print('Cluster %d'%dist_box[i], end='\t')
                des = cluster[dist_box[i]]
                for j in des:
                    print(j, end=' ')
                data = self.X_data.loc[:,des]
                g = sns.pairplot(data,kind='reg',diag_kind='kde')
                g.map_lower(corrfunc)
                g.map_upper(corrfunc)                
                plt.show()
                print ('\n')
            
        
        interact(func, label=ToggleButtons(options=labels,
                                         description = 'Bins Index'))
        
class FeatureCluster_KMeans:
    
    def __init__(self,X_data,n_cluster=500, random_state =0, init='k-means++',algorithm='auto'):
        self.cluster_info = []
        self.assignments = 0
        self.init = init
        self.algorithm = algorithm
        self.abs_corre = abs(X_data.corr())
        self.X_data = X_data
        self.train = X_data.iloc[:,3:].T
        self.train.columns = X_data.index.values
        self.n_cluster = n_cluster
        self.random_state = random_state
        
        
    def set_cluster(self):
        model = KMeans(self.n_cluster, random_state = self.random_state)
        self.train['Cluster'] = model.fit_predict(self.train)
        self.assignment = np.array(self.train['Cluster'])
        cluster_output = DataFrame({'Feature':list(self.train.index) , 'cluster':self.assignment})
        
        #save_data = self.train.T
        #save_data = save_data.rename(index={save_data.shape[0]-1:'cluster'})
        #save_data.to_csv('KMeansCluster.csv',sep=',',index=True)
        fw = open('%s_kmeans.cluster'%ds.data_name,'w')
       
        nc = list(cluster_output.cluster.values)
        nnc = max(nc)
        name = list(cluster_output.Feature.values)
        cludic = {}
        for i in range(0,len(nc)):
            k = name[i]
            v = nc[i]
            cludic[k] = v
            
        for t in range(0,nnc+1):
            vv = []
            vv = [key for key, value in list(cludic.items()) if value == t]    #Find Key by Value in Dictionary
            self.cluster_info.append(vv)
            print ('\n','\x1b[1;46m'+'Cluster'+'\x1b[0m',t,vv, end=' ')
            fw.write('Cluster %s\t'%t)
            for des in vv:
                fw.write(des+' ')
            fw.write('\n')
            
        fw.close()
        print('\n')
        
        print('Cluster info file','\033[1m'+'%s_kmeans.cluster'%ds.data_name+'\033[0m','file saved')
        return self.assignment

    def cluster_dist(self) :
        """
        Show dendrogram of correlation coefficient distribution of each cluster

        Returns
        -------
        None
        """
        X_data = self.X_data.iloc[:,3:]
        abs_corre = abs(X_data.corr())
        dist_box = {}
        assignments = self.cluster_info
        cluster_output = DataFrame({'Feature':list(self.train.index) , 'cluster':self.assignment})
        nc = list(cluster_output.cluster.values)
        name = list(cluster_output.Feature.values)

        clu_hist  = {}
        cluster= []

        for i in range(0,len(nc)):
            k = name[i]
            v = nc[i]
            clu_hist [k] = v
        for t in range(0,max(nc)+1):
            vv = []
            vv = [key for key, value in list(clu_hist.items()) if value == t]    #Find Key by Value in Dictionary
            cluster.append(vv)
        
        for val, s in enumerate(cluster) :
            desc_set = s
            c = len(desc_set)
            if c == 1 :
                pass
            else:
                tay = abs_corre.loc[desc_set,desc_set]
                t =  np.array(tay)

                av = (t.sum()-c)/2
                aver = av/((c*c-c)/2)
                dist_box[aver] = val
        
        dist_array = np.array(list(dist_box.keys()))       
        cnt, bins, patches = plt.hist(dist_box.keys(),edgecolor='black')
        plt.ylabel('Frequency')
        plt.xlabel('Correlation coefficient of each cluster')
        
        buttons = {}
        for num in range(len(cnt)):
            Range = []
            Range.append(bins[num])
            Range.append(bins[num+1])
            buttons[num] = Range
        labels = np.array(list(buttons.keys()))
        index = np.where(cnt==0)[0]
        labels = list(np.delete(labels,index))
        

        bin_centers = np.diff(bins)*0.5 + bins[:-1]
        for fr, x, patch in zip(cnt,bin_centers,patches):
            height = fr
            plt.annotate("{}".format(height),
                        xy = (x, height),
                        xytext = (0,0.2),
                        textcoords = 'offset points',
                        ha = 'center',
                        va = 'bottom'
                        )


        def func(label):
            val_range = buttons[int(label)]
            cluster_index = np.where((dist_array>=val_range[0]) & (dist_array<val_range[1]))
            print('\033[1m'+'Bins : %s\t'%str(val_range) + 'Count :%d'%cnt[int(label)] + '\033[0m')
        
            for i in dist_array[cluster_index]:
                print('Cluster %d'%dist_box[i], end='\t')
                des = cluster[dist_box[i]]
                for j in des:
                    print(j, end=' ')
                data = self.X_data.loc[:,des]
                g = sns.pairplot(data,kind='reg',diag_kind='kde')
                g.map_lower(corrfunc)
                g.map_upper(corrfunc)
                plt.show()
                
                print ('\n')
                




        interact(func, label=ToggleButtons(options=labels,
                                         description = 'Bins Index'))
    
    def model_evaluation(self):
        train_samples = silhouette_samples(self.train,self.train['Cluster'])
        self.train['silhouette_coeff'] = train_samples 
        print('silhouette average score : %f'%silhouette_score(self.train,self.train['Cluster']))
        group = self.train.groupby('Cluster')['silhouette_coeff'].mean()
        print('silhouette socre group by \n',group)
        fw = open('kmeans.eval','w')
        fw.write('silhouette average score : %f'%silhouette_score(self.train,self.train['Cluster']))
        for index, val in enumerate(group):
            fw.write('Cluster %d\t%f\n'%(index,val))
        plt.hist(group)
        
        group_array = np.array(group)
        index_overlapping = np.where(group_array ==0)[0]
        index_wrong = np.where(group_array < 0)[0]
        fw.write('\n')
        fw.write('overlapped Cluseter Num : ')
        fw.write(str(list(index_overlapping)))
        fw.write('\n')
        fw.write('wrong/incorrect Cluster Num : ')
        fw.write(str(list(index_wrong)))
        fw.close()
        print('\n')
        print('overlapped Cluseter Num : ',list(index_overlapping))
        print('wrong/incorrect Cluster Num : ',list(index_wrong))
        
        if len(index_wrong) > 1:
            print('try Clustering again')
        print('\n')
        print('kmeans.eval file saved')
class FeatureCluster_Minisom:
    
    def __init__(self,X_data, error='qe',random_seed=0, train=1000, neighborhood_function='gaussian',topology='hexagonal'):
        self.X_data = X_data
        self.abs_corre = abs(X_data.corr())
        self.data = X_data.iloc[:,3:].T
        self.data.columns = X_data.index.values
        self.error = error
        self.neighborhood_function = neighborhood_function
        self.random_seed = random_seed
        self.input_len = self.data.shape[1]
        self.train = train
        self.topology = topology
        self.assignment=0
        self.cluster_info=[]
        self.abs_corre = abs(X_data.corr())
        

    def parameter_combine(self,min_size,max_size):
        s_time = pd.Timestamp.now()
        print('시작시간:',s_time,'\n')

        #원하는 파라미터 조합 리스트화
        map_n= [n for n in range(min_size,max_size)]
        para_sigma= [np.round(sigma*0.1,2) for sigma in range(1,11)]
        para_learning_rate= [np.round(learning_rate*0.1,2) for learning_rate in range(1,10)]

        #결과 값을 담을 리스트 res 생성
        res = []
        #모든 조합에 대해 모델 생성 및 qe,te값 계산
        for n in map_n:
            for sigma in para_sigma:
                for lr in para_learning_rate:
            
                    try:
                        #랜덤으로 초기값을 설정하는 경우
                        estimator = MiniSom(n,n,self.input_len,sigma =sigma, learning_rate = lr, 
                                            topology=self.topology,random_seed=self.random_seed)
                        estimator.random_weights_init(self.data.values)
                        estimator.train(self.data.values,self.train,random_order=True)
                        winner_coordinates = np.array([estimator.winner(x) for x in self.data.values]).T
                        cluster_index = np.ravel_multi_index(winner_coordinates,(n,n))
                        if self.error == 'qe':
                            qe = estimator.quantization_error(self.data.values)
                            res.append([str(n)+'x'+str(n),sigma,lr,'random_init',qe,len(np.unique(cluster_index))])
                        elif self.error == 'te':
                            te = estimator.topographic_error(self.data.values)
                            res.append([str(n)+'x'+str(n),sigma,lr,'random_init',qe,len(np.unique(cluster_index))])
                        
               

                        #pca로 초기값을 설정하는 경우
                        estimator = MiniSom(n,n,self.input_len,sigma =sigma, learning_rate = lr
                                            ,topology=self.topology, random_seed=self.random_seed)
                        estimator.pca_weights_init(self.data.values)
                        estimator.train(self.data.values,self.train,random_order=True)
                        winner_coordinates = np.array([estimator.winner(x) for x in self.data.values]).T
                        cluster_index = np.ravel_multi_index(winner_coordinates,(n,n))
                        if self.error == 'qe':
                            qe = estimator.quantization_error(self.data.values)
                            res.append([str(n)+'x'+str(n),sigma,lr,'random_init',qe,len(np.unique(cluster_index))])
                        elif self.error == 'te':
                            te = estimator.topographic_error(self.data.values)
                            res.append([str(n)+'x'+str(n),sigma,lr,'random_init',qe,len(np.unique(cluster_index))])
                
                    except ValueError as e:
                        print(e)
            
        #결과 데이터프레임 생성 및 sorting 
        df_res = pd.DataFrame(res,columns=['map_size','sigma','learning_rate','init_method','qe','n_cluster']) 
        df_res.sort_values(by=[self.error],ascending=True,inplace=True,ignore_index=True)
       

        #시각화를 위한 lineplot 생성
#        plt.figure(figsize=(20,10))
#        sns.lineplot(data = df_res)

        e_time = pd.Timestamp.now()
        print('\n종료시간:',e_time,'\n총 소요시간:',e_time-s_time)

        return df_res
    
    def set_cluster(self,map_n,sigma,lr,init):
        estimator = MiniSom(map_n,map_n,self.input_len,
                            sigma=sigma,learning_rate=lr,
                            topology=self.topology,
                            random_seed = self.random_seed,
                            neighborhood_function=self.neighborhood_function,
                            activation_distance='euclidean')
        if init == 'pca':
            estimator.pca_weights_init(self.data.values)
        elif init == 'random':
            estimator.random_weights_init(self.data.values)
        else:
            print ('Possible Init : pca, random')
            raise IndexError
        
        estimator.train(self.data.values,self.train,random_order=True)
        
        if self.error == 'qe':
            print('Quantization Error :',estimator.quantization_error(self.data.values))
        elif self.error == 'te':
            print('Topology Error :',estimator.topographic_error(self.data.values))
        
        winner_coordinates = np.array([estimator.winner(x) for x in self.data.values]).T
        cluster_index = np.ravel_multi_index(winner_coordinates,(map_n,map_n))

        clust_uniq = np.unique(cluster_index)
        clust_dict = {}
        for index, val in enumerate(clust_uniq):
            clust_dict[val] = index
            
        for index, i in enumerate(cluster_index):
            if clust_dict[i] == i:
                pass
            else:
                cluster_index[index] = clust_dict[i]       
    
        self.assignment = cluster_index
        
        cluster_output = DataFrame({'Feature':list(self.data.index) , 'cluster':self.assignment})

        fw = open('%s_som.cluster'%ds.data_name,'w')
        nc = list(cluster_output.cluster.values)
        nnc = max(nc)
        name = list(cluster_output.Feature.values)
        cludic = {}
        for i in range(0,len(nc)):
            k = name[i]
            v = nc[i]
            cludic[k] = v
        blank_cluster = []    
        for t in range(0,nnc+1):
            vv = []
            vv = [key for key, value in list(cludic.items()) if value == t]    #Find Key by Value in Dictionary
            self.cluster_info.append(vv)
            print ('\n','\x1b[1;46m'+'Cluster'+'\x1b[0m',t,vv, end=' ')

            fw.write('Cluster %s\t'%t)
            for des in vv:
                fw.write(des+' ')
            fw.write('\n')
        fw.close()
        
        print('\nCluster info file','\033[1m'+'%s_som.cluster'%ds.data_name+'\033[0m','file saved')
        
        
        xx, yy = estimator.get_euclidean_coordinates()
        umatrix = estimator.distance_map()
        weights = estimator.get_weights()
        f = plt.figure(figsize=(10,10))
        ax = f.add_subplot(111)
        ax.set_aspect('equal')
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                wy = yy[(i, j)] * 2 / np.sqrt(3) * 3 / 4
                if self.topology == 'hexagonal':
                    shape = RegularPolygon((xx[(i, j)], wy), 
                                         numVertices=6, 
                                         radius=.95 / np.sqrt(3),
                                         facecolor=cm.Blues(umatrix[i, j]), 
                                         alpha=.4, 
                                         edgecolor='gray')
                elif self.topology == 'rectangular':
                    shape = RegularPolygon((xx[(i, j)], wy), 
                                         numVertices=4,
                                         radius=.95 / np.sqrt(2.6),
                                         orientation = 0.785398163,
                                         facecolor=cm.Blues(umatrix[i, j]), 
                                         alpha=.4, 
                                         edgecolor='gray')                    
                plot = ax.add_patch(shape)
       
    
    
        cnt=[]
        for c in np.unique(cluster_index):
                x_= [estimator.convert_map_to_euclidean(estimator.winner(x))[0] + 
                     (2*np.random.rand(1)[0]-1)*0.4 for x in self.data.values[cluster_index==c]]
                y_= [estimator.convert_map_to_euclidean(estimator.winner(x))[1] + 
                     (2*np.random.rand(1)[0]-1)*0.4 for x in self.data.values[cluster_index==c]]
                y_= [(i* 2 / np.sqrt(3) * 3 / 4) for i in y_]
    
                plot = sns.scatterplot( x = x_,  y= y_ ,label='cluster='+str(c),alpha=.7)
        plt.legend(bbox_to_anchor=(1.4,1))
        df_cnt = pd.DataFrame(cnt,columns=['cluster이름','개수'])
        df_cnt

        #x축,y축 간격 설정 
        xrange = np.arange(weights.shape[0])
        yrange = np.arange(weights.shape[1])
        plot = plt.xticks(xrange-.5, xrange)
        plot = plt.yticks(yrange * 2 / np.sqrt(3) * 3 / 4, yrange)

        #차트 우측에 color bar생성
        divider = make_axes_locatable(plt.gca())
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
        cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Blues, 
                            orientation='vertical', alpha=.4)
        cb1.ax.get_yaxis().labelpad = 16
        plot = cb1.ax.set_ylabel('distance from neurons in the neighborhood',
                          rotation=270, fontsize=16)
        plot = plt.gcf().add_axes(ax_cb)

        #이미지 저장
        plt.savefig('som_seed_%s_%s.png'%(self.error,self.topology))
        
        
        return self.assignment

    def cluster_dist(self) :
        """
        Show dendrogram of correlation coefficient distribution of each cluster

        Returns
        -------
        None
        """
        X_data = self.X_data.iloc[:,3:]
        data = self.data
        abs_corre = abs(X_data.corr())
        dist_box = {}

        assignments = self.cluster_info
        cluster_output = DataFrame({'Feature':list(data.index) , 'cluster':self.assignment})
        nc = list(cluster_output.cluster.values)
        name = list(cluster_output.Feature.values)

        clu_hist  = {}
        cluster= []

        for i in range(len(nc)):
            k = name[i]
            v = nc[i]
            clu_hist [k] = v
        for t in np.unique(nc):
            vv = []
            vv = [key for key, value in list(clu_hist.items()) if value == t]    #Find Key by Value in Dictionary
            cluster.append(vv)
        
        for val, s in enumerate(cluster) :
            desc_set = s
            c = len(desc_set)
            if c == 1 :
                pass
            else:
                tay = abs_corre.loc[desc_set,desc_set]
                t =  np.array(tay)

                av = (t.sum()-c)/2
                aver = av/((c*c-c)/2)
                dist_box[aver] = val
        
        dist_array = np.array(list(dist_box.keys()))       
        cnt, bins, patches = plt.hist(dist_box.keys(),edgecolor='black')
        plt.ylabel('Frequency')
        plt.xlabel('Correlation coefficient of each cluster')
        
        buttons = {}
        for num in range(len(cnt)):
            Range = []
            Range.append(bins[num])
            Range.append(bins[num+1])
            buttons[num] = Range
        labels = np.array(list(buttons.keys()))
        index = np.where(cnt==0)[0]
        labels = list(np.delete(labels,index))
        

        bin_centers = np.diff(bins)*0.5 + bins[:-1]
        for fr, x, patch in zip(cnt,bin_centers,patches):
            height = fr
            plt.annotate("{}".format(height),
                        xy = (x, height),
                        xytext = (0,0.2),
                        textcoords = 'offset points',
                        ha = 'center',
                        va = 'bottom'
                        )

        def func(label):
            val_range = buttons[int(label)]
            cluster_index = np.where((dist_array>=val_range[0]) & (dist_array<val_range[1]))
            print('\033[1m'+'Bins : %s\t'%str(val_range) + 'Count :%d'%cnt[int(label)] + '\033[0m')
        
            for i in dist_array[cluster_index]:
                print('Cluster %d'%dist_box[i], end='\t')
                des = cluster[dist_box[i]]
                for j in des:
                    print(j, end=' ')
                data = self.X_data.loc[:,des]
                g = sns.pairplot(data,kind='reg',diag_kind='kde')
                g.map_lower(corrfunc)
                g.map_upper(corrfunc)      
                plt.show()
                print ('\n')
        
        interact(func, label=ToggleButtons(options=labels,
                                         description = 'Bins Index'))
        
        
def Draw_epoch():
    
    file_list = [file for file in os.listdir() if file.endswith('.log')]
    print('%-7s %-7s'%('Index','File Name'))
    for index, i in enumerate(file_list):
        print(f'{index:<7} {i}')
    print('\n')
    clust = input('Enter Clust Info File Index : ')
  
    print('\x1b[48;5;226m'+f'{file_list[int(clust)]}'+'\x1b[0m','   file selected\n')    
    f = open(file_list[int(clust)], 'r')
    lines = f.readlines()
    f.close()

    _r2_log = []
    _rmse_log = []
    for line in lines:
        line = line.strip()
        line = line.split(',')
        _r2_log.append(float(line[0]))
        _rmse_log.append(float(line[1]))

    fig, host = plt.subplots(figsize=(10.0, 7.0))
    host.set_title("Epoch Graph")

    x = range(len(_r2_log))

    color = 'tab:red'
    host.set_xlabel("Generation")
    host.set_ylabel('RMSE', color=color)
    h, = host.plot(x, _rmse_log, '-', label="RMSE", color=color)
    host.set_ylim(min(_rmse_log)*0.95, max(_rmse_log)*1.05)
    host.tick_params(axis='x')
    host.tick_params(axis='y', labelcolor=color)

    ax1 = host.twinx()
    color = 'tab:blue'
    ax1.set_ylabel('R^2', color=color)
    p1, = ax1.plot(x, _r2_log, '-', label="R^2", color=color)
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='y', labelcolor=color)
    #ax1.xaxis.set_ticks([])

    #lns = [h, p1]
    #host.legend(handles=lns, loc='lower right')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #plt.savefig(fig_name+".png")
    plt.show()

    return

def selection_mc(X_data,y_data,model='MLR',pop_info=None,learning=50000,bank=200,component=4,pntinterval=1000):
    """
    Feature selection algorothm with single core

    Parameters
    ----------
    X_data : pandas DataFrame , shape = (n_samples, n_features)
    y_data : pandas DataFrame , shape = (n_samples,)
    cluster_info : Return value of clustering.FeatureCluster.set_cluster()
    model : default='regression' / when user use default value, operate this function for regression model /
    when user use other value like 'Classification', operate this function for classification model/
    This function only have two options : regression(default) or classification
    learning : Number of learning
    bank : Size of bank(pool) in selection operation
    component : Number of feature in prediction model
    pntinterval : print currnt score and selected features set once every 'pntinterval' times

    Returns
    -------
    list, result of selected best feature set
    """
    now = datetime.datetime.now()
    nowTime = now.strftime('%H:%M:%S')
    print ('Start time : ',nowTime)

    file_list = [file for file in os.listdir() if file.endswith('.cluster')]
    print('%-7s %-7s'%('Index','File Name'))
    for index, i in enumerate(file_list):
        print(f'{index:<7} {i}')
    print('\n')
    clust = input('Enter Clust Info File Index : ')
  
    print('\x1b[48;5;226m'+f'{file_list[int(clust)]}'+'\x1b[0m','   file selected\n')
    cluster_output = cluster_dataframe(X_data,file_list[int(clust)])
    method = ['kmeans','som','hierarchical']
    for i in method:
        if i in file_list[int(clust)]:
            met = i
            break

    
    if model == 'MLR' :
        print ('\x1b[1;42m'+'MLR'+'\x1b[0m')
        #print(100 => 23:26:25 [0.6140169067392781, 0.8674773725953273])
        print("                       R^2                  RMSE")
        y_mlr = lm.LinearRegression()
        e_mlr = lm.LinearRegression()
    elif model == 'PLS' :
        print ('\x1b[1;42m'+'PLS'+'\x1b[0m')
        #print(100 => 23:26:25 [0.6140169067392781, 0.8674773725953273])
        print("                       R^2                  RMSE")
        y_mlr = PLSRegression(n_components=component)
        e_mlr = PLSRegression(n_components=component)
    else :
        print ('\x1b[1;42m'+'Classification'+'\x1b[0m')
        y_mlr = SVC(kernel='rbf', C=1.0 ,gamma=0.1 ,random_state=0)
        e_mlr = SVC(kernel='rbf', C=1.0 ,gamma=0.1 ,random_state=0)

        

    nc = list(cluster_output.cluster.values)
    name = list(cluster_output.Features.values)
    values = np.unique(cluster_output['cluster'].values)
    X_data = X_data.loc[:,name]
    clu_hist  = {}
    cluster= []
    
    for i in range(0,len(nc)):
        k = name[i]
        v = nc[i]
        clu_hist[k] = v
    for t in values:
        vv = []
        vv = [key for key, value in list(clu_hist.items()) if value == t]    #Find Key by Value in Dictionary
        cluster.append(vv)


    if pop_info==None:
        ini_desc_bank= []
        index_sort_bank = []

        while len(ini_desc_bank) < bank :
            leenlis = list(range(len(values)))
            ini_desc = []
            index_sort = []

            for j in range(component) :
                clu_n = random.choice(leenlis)
                leenlis.remove(clu_n)

                index_sort.append(clu_n)
                ini_desc.append(random.choice(cluster[clu_n]))
                index_sort.sort()
                ini_desc.sort()

                if index_sort not in index_sort_bank :
                    index_sort_bank.append(index_sort)
                    ini_desc_bank.append(ini_desc)
    elif pop_info!=None:
        ini_desc_bank= []
        for _temp in pop_info:
            ini_desc_bank.append(_temp[2])

    uni_desc = []
    for i in ini_desc_bank :
        x = DataFrame(X_data, columns=i)
        xx = np.array(x, dtype=np.float64)
        y = np.array(y_data, dtype=np.float64)

        #y_mlr = lm.LinearRegression()
        y_mlr.fit(xx,y_data.values.ravel())
        score = y_mlr.score(xx,y)
        pred_y = y_mlr.predict(xx)
        rmse = mean_squared_error(y, pred_y, squared=False)

        mid = []
        mid.append(score)
        mid.append(rmse)
        mid.append(i)

        uni_desc.append(mid)

    top_ranker = uni_desc
    n = 0
    total_r2_log = []
    total_rmse_log = []
    for s in notebook.tqdm(range(learning)) :
        n = n + 1
        evoldes = []
        for h in top_ranker :
            evoldes.append(h)


        for j,r,i in top_ranker :    # j = r2 , r = rmse, i = desc set
            #child = []
            group_n = []
            for y in i :
                gn= clu_hist[y]-1
                group_n.append(gn)

            sw_index = random.randrange(0,component)
            while 1 :
                sw_group = random.randrange(0,len(values))
                if sw_group in group_n :
                    continue

                else :
                    break

            b_set = copy.deepcopy(i)
            b_set[sw_index] = random.choice(cluster[sw_group])
            b_set.sort()
            x = DataFrame(X_data, columns=b_set)
            xx = np.array(x, dtype=np.float64)
            y = np.array(y_data, dtype=np.float64)
            e_mlr = lm.LinearRegression()
            e_mlr.fit(xx,y_data.values.ravel())
            score = e_mlr.score(xx,y)
            pred_y = e_mlr.predict(xx)
            rmse = mean_squared_error(y, pred_y, squared=False)
            mid = []
            mid.append(score)
            mid.append(rmse)
            mid.append(b_set)
            evoldes.append(mid)
        #print evoldes ,len(evoldes)

        rank_filter = []
        #중복지우는부분
        for i in evoldes :
            if i in rank_filter :
                pass
            else :
                rank_filter.append(i)

        rank_filter.sort(reverse= True)
        top_ranker =  rank_filter[:bank]

        total_r2_log.append(top_ranker[0][0])
        total_rmse_log.append(top_ranker[0][1])

        if n % pntinterval == 0 :
            tt = datetime.datetime.now()
            #print (n , '=>', tt.strftime('%H:%M:%S') , top_ranker[0])
            print ('%7d'%n , '=>', tt.strftime('%H:%M:%S') , '[%7.4f,%7.4f]'%(top_ranker[0][0],top_ranker[0][1]), top_ranker[0][2])

    for index,i in enumerate(top_ranker[0]):
        if index == 0:
            print('%-8s : %7.4f'%('R2',i))
        elif index == 1:
            print('%-8s : %7.4f'%('RMSE',i))
        elif index == 2:
            print('%-8s : %s'%('Cluster',i))
    #print('%-8s : '%'Final', '[%7.4f,%7.4f]'%(top_ranker[0][0],top_ranker[0][1]), top_ranker[0][2])
    #print (top_ranker[0])
    clulog = []
    for j,r,i in top_ranker :
            #child = []
            group_n = []
            for y in i :
                gn= clu_hist[y]
                clulog.append(gn)
            break
    print ("Model's cluster info",clulog)
    fi = datetime.datetime.now()
    fiTime = fi.strftime('%H:%M:%S')
    #print ('Finish time : ',fiTime)

    _epoch_log_str = ""
    for i in range(len(total_r2_log)):
        _epoch_log_str += "%7.4f,%7.4f\n"%(total_r2_log[i], total_rmse_log[i])

    if pop_info==None:
        fw = open("%s_mc_%s_%s.log"%(ds.data_name,model,met),'w')
        print("%s_mc_%s_%s.log  is saved!"%(ds.data_name,model,met))
    elif pop_info :
        fw = open("%s_mc_%s_%s.log"%(ds.data_name,model,met),'a')
        print("%s_mc_%s_%s.log  is updated!"%(ds.data_name,model,met))
    fw.write(_epoch_log_str)
    fw.close()

    return top_ranker[0][2], top_ranker


#####################################################################################
def Initialization(n_pop_size, n_feature):

    pop_index = []
    while len(pop_index) < n_pop_size:
        individual = []
        while len(individual) < n_feature:
            i = random.randint(0, len(cluster)-1)
            if i in individual: continue
            individual.append(i)
        individual.sort()

        if individual in pop_index: continue
        pop_index.append(individual)

    population = []
    for _index in pop_index:
        individual = []
        for i in _index:
            if len(cluster[i]) == 1:
                individual.append(cluster[i][0])
            else:
                j = random.randint(0, len(cluster[i])-1)
                individual.append(cluster[i][j])
        population.append(individual)

    return population
#####################################################################################
def Evaluation(individual):
    #print(individual)
    x = X_train.loc[:,individual].values
    y = y_train.values
    mlr = LinearRegression()
    mlr.fit(x,y)
    score = mlr.score(x, y)

    pred_y = mlr.predict(x)
    rmse = mean_squared_error(y, pred_y, squared=False)

    #fit_value = (math.e ** (1/(rmse**6))) * score
    fit_value = (1/(rmse**4)) * score
    #fit_value = score

    return fit_value
#####################################################################################
def R2_RMSE(individual):

    x = X_train.loc[:,individual].values
    y = y_train.values
    mlr = LinearRegression()
    mlr.fit(x,y)
    score = mlr.score(x, y)

    pred_y = mlr.predict(x)
    rmse = mean_squared_error(y, pred_y, squared=False)

    return score, rmse
#####################################################################################
def delete_duplication(population):

    _temp = []
    for individual in population:
        individual.sort()
        _temp_str = '*'.join(individual)
        _temp.append(_temp_str)

    _temp = list(set(_temp))

    edited_population = []
    for _temp_str in _temp:
        individual = _temp_str.split('*')
        edited_population.append(individual)

    return edited_population
#####################################################################################
def Add_Individual(population, n_pop_size, n_feature):

    new_population = population[:]
    while len(new_population) < n_pop_size*2 :

        _index = []
        while len(_index) < n_feature:
            i = random.randint(0, len(cluster)-1)
            if i in _index: continue
            _index.append(i)
        _index.sort()

        individual = []
        for i in _index:
            if len(cluster[i]) == 1:
                individual.append(cluster[i][0])
            else:
                j = random.randint(0, len(cluster[i])-1)
                individual.append(cluster[i][j])

        if individual in new_population: continue

        new_population.append(individual)

    return new_population
#####################################################################################
def Rank_by_Score(population, n_pop_size, n_feature):

    population = delete_duplication(population)
    #if len(population) < n_pop_size*2 :
    #    population = Add_Individual(population, n_pop_size, n_feature)

    for individual in population:
        #print(individual)
        fit_value = Evaluation(individual)
        individual.insert(0,fit_value)

    population.sort()
    ranked_population = population[::-1]

    fitScore_list = []
    for sc_individual in ranked_population:
        fitScore_list.append(sc_individual[0])

    fitScore_ratio_list = []
    _sum_fitScore = sum(fitScore_list)
    for i in range(len(fitScore_list)):
        fitScore_ratio_list.append(fitScore_list[i] / _sum_fitScore)

    #print("%-8s%-12s%-12s%-12s%s "%("Rank","Fit Val","RMSE","R2","Portion"))
    #print("%-12s%-12s%s : "%("RMSE","R2","Portion"),end='')
    #print("[%s %s %s] : "%("RMSE","R2","Portion"),end='')
    cnt = 0
    _first_r2 = 0
    _first_rmse = 0
    for sc_individual in ranked_population:
        #if cnt < 5 or cnt > len(ranked_population)-6:
        if cnt in list(range(0,50,10)) :
            _fit = sc_individual[0]
            _r2, _rmse = R2_RMSE(sc_individual[1:])
            #print("%-12.6f%-12.6f%-12.6f%.6f%% "%(_fit, _rmse, _r2, fitScore_ratio_list[cnt]), sc_individual[1:])
            #print("%-8d%-12.5f%-12.6f%-12.6f%.6f%% "%(cnt+1, _fit, _rmse, _r2, fitScore_ratio_list[cnt]*100))

            if cnt==0:
                _first_r2, _first_rmse = R2_RMSE(sc_individual[1:])
                #print("%-12.6f%-12.6f%.6f%% "%(_rmse, _r2, fitScore_ratio_list[cnt]*100))
                print("%10.6f %10.6f %10.6f%% "%(_rmse, _r2, fitScore_ratio_list[cnt]*100))

        del sc_individual[0]
        cnt += 1

    return ranked_population, fitScore_ratio_list, fitScore_list, _first_r2, _first_rmse
#####################################################################################
def roulette_sel(score_ratio_list):
    index_num = random.randrange(len(score_ratio_list))
    probability = random.random()
    return index_num, probability
#####################################################################################
def Selection_Mate(population, n_pop_size, fitScore_ratio_list):

    mate_list = []
    while len(mate_list) < int((n_pop_size*2-50)/2) :
        i1,p1 = roulette_sel(fitScore_ratio_list)
        #print("Mate 1 : ",p1, fitScore_ratio_list[i1])
        if p1 <= fitScore_ratio_list[i1]:
        #if p1 > 0:
            mate1 = population[i1]
            while True:
                #i2,p2 = roulette_sel(fitScore_ratio_list)
                #if p2 <= fitScore_ratio_list[i2]:
                i2 = random.randrange(len(fitScore_ratio_list))
                if i2 == i1: continue

                mate2 = population[i2]
                break

            _mate = [mate1, mate2]
            _mate.sort()
            if _mate in mate_list: continue
            mate_list.append(_mate)

    return mate_list
#####################################################################################
def Crossover(population, mate_list):

    cross_pop = []
    for mate in mate_list:
        if len(mate[0])==2:
            new_1 = [mate[0][0], mate[1][1]]
            new_2 = [mate[1][0], mate[0][1]]
        elif len(mate[0]) > 2:
            cross_site = random.randint(1,len(mate[0])-2)
            #cross_site = np.random.randint(1,len(population[0]))
            new_1 = mate[0][:cross_site] + mate[1][cross_site:]
            new_2 = mate[1][:cross_site] + mate[0][cross_site:]

        cross_pop.append(new_1)
        cross_pop.append(new_2)
    #new_cross_pop = population + cross_pop
    #print("### Mate & Crossover count : %d"%len(mate_list))
    return cross_pop
#####################################################################################
def Mutation(individual):

    muta_individual = individual[:]
    _i = random.randint(0, len(muta_individual)-1)

    for j in range(len(cluster)):
        if muta_individual[_i] in cluster[j]:
            while True:
                k = random.randint(0, len(cluster)-1)
                if k!=j: break
            if len(cluster[k]) > 1:
                n = random.randint(0, len(cluster[k])-1)
                muta_individual[_i] = cluster[k][n]
            else:
                muta_individual[_i] = cluster[k][0]
            break

    a='''
    for j in range(len(cluster)):
        if len(cluster[j]) > 1:
            if muta_individual[_i] in cluster[j]:
                _temp_cluster = cluster[j][:]
                _temp_cluster.remove(muta_individual[_i])

                k = random.randint(0, len(_temp_cluster)-1)
                muta_individual[_i] = _temp_cluster[k]
                break

        else :
            if muta_individual[_i] == cluster[j][0]:
                k = 0
                while True:
                    k = random.randint(0, len(cluster)-1)
                    if k!=j: break
                muta_individual[_i] = cluster[k][0]
                break
    '''

    return muta_individual
#####################################################################################
def diversity(population):

    _index_pop = []
    for individual in population:
        _index_indi = []
        for des in individual:

            for i in range(len(cluster)):
                if len(cluster[i]) > 1:
                    if des in cluster[i]:
                        _index_indi.append(i)
                        break
                else :
                    if des == cluster[i][0]:
                        _index_indi.append(i)
                        break
        _index_pop += _index_indi

    _index_pop = list(set(_index_pop))
    #print("Diversity : ",len(_index_pop),"/",len(cluster))

    return len(_index_pop)
#####################################################################################
def selection_ga(X_data, y_data, model='MLR', pop_info=None, n_pop_size=300, N_generation=5000, component=4):

    if model == 'MLR' :
        print ('\x1b[1;42m'+'MLR'+'\x1b[0m')
        #print(100 => 23:26:25 [0.6140169067392781, 0.8674773725953273])
        #print("                       R^2                  RMSE")
        y_mlr = LinearRegression()
        e_mlr = LinearRegression()
    elif model == 'PLS' :
        print ('\x1b[1;42m'+'PLS'+'\x1b[0m')
        #print(100 => 23:26:25 [0.6140169067392781, 0.8674773725953273])
        #print("                       R^2                  RMSE")
        #y_mlr = lm.LinearRegression()
        #e_mlr = lm.LinearRegression()

    _sel_Num = component
    cxpb=1.0
    mupb=0.05
    
    file_list = [file for file in os.listdir() if file.endswith('.cluster')]
    print('%-7s %-7s'%('Index','File Name'))
    for index, i in enumerate(file_list):
        print(f'{index:<7} {i}')
    print('\n')
    clust = input('Enter Clust Info File Index : ')

    print('\x1b[48;5;226m'+f'{file_list[int(clust)]}'+'\x1b[0m','   file selected\n')
    cluster_output = cluster_dataframe(X_data,file_list[int(clust)])
    method = ['kmeans','som','hierarchical']
    for i in method:
        if i in file_list[int(clust)]:
            met = i
            break
    global X_train
    global y_train
    global cluster
    nc = list(cluster_output.cluster.values)
    name = list(cluster_output.Features.values)
    X_train = X_data.loc[:,name]
    y_train = y_data
    values = np.unique(cluster_output['cluster'].values)
    clu_hist  = {}
    cluster= []

    for i in range(0,len(nc)):
        k = name[i]
        v = nc[i]
        clu_hist[k] = v
    for t in values:
        vv = []
        vv = [key for key, value in list(clu_hist.items()) if value == t]    #Find Key by Value in Dictionary
        cluster.append(vv)

    print("---------------  Initialization  ---------------")
    if pop_info==None:
        parent = Initialization(n_pop_size=n_pop_size, n_feature=_sel_Num)
        child = Initialization(n_pop_size=n_pop_size, n_feature=_sel_Num)
        population = parent + child
    elif pop_info!=None:
        population = pop_info[:]

    fitScore_list = []
    for individual in population: fitScore_list.append(Evaluation(individual))
    fitScore_ratio = []
    _sum_fitScore = sum(fitScore_list)
    for i in range(len(fitScore_list)): fitScore_ratio.append(fitScore_list[i] / _sum_fitScore)

    avg_fit_list = []
    total_r2_log = []
    total_rmse_log = []
    n=0
    for s in notebook.tqdm(range(N_generation)):
        
        ## Selection
        #print("Input size of Selection : ",len(population))
        diversity_num =diversity(population)
        mate_list = Selection_Mate(population, int(len(population)/2), fitScore_ratio)

        ## Next Generation
        #next_gen_Num = 10
        parent = population[:50]

        ## CrossOver
        #if random.random() < cxpb:
        child = Crossover(population, mate_list)
        #print("### CrossOver is done")

        ## Mutation
        #muta_cnt = 0
        for i in range(len(child)):
            individual = child[i]
            #if random.random() < mupb:
            #    muta_cnt += 1
            muta_indi = Mutation(individual)
            child[i] = muta_indi
        #print("### Muataion Count : ",muta_cnt)

        population = parent + child
        population, fitScore_ratio, fitScores, first_R2, first_RMSE = Rank_by_Score(population, n_pop_size=n_pop_size, n_feature=_sel_Num)
        total_r2_log.append(first_R2)
        total_rmse_log.append(first_RMSE)
        if n == N_generation : break
        #print(" %d Generation  : "%(n+1),end='')
        if s == 0:
            print("%10s  %10s%10s   %10s"%('Generation', 'RMSD','R2','Portion'))
        print("%10d : "%(n+1),end='')

        a='''
        if n > 20 and len(avg_fit_list)>=next_gen_Num+1 :
            avg = round(sum(avg_fit_list[0-1-next_gen_Num:-1])/next_gen_Num,4)
            last = round(avg_fit_list[-1],4)
            if avg == last:
                print("$$$$$  Migration   $$$$$")
                new_pop = Initialization(n_pop_size=n_pop_size*2-next_gen_Num, n_feature=_sel_Num)
                population = population[:next_gen_Num] + new_pop
                avg_fit_list = []
        avg_fit_list.append(sum(fitScores[:next_gen_Num])/next_gen_Num)
        '''

        if diversity_num < len(cluster)*2/3 :
            #print("$$$$$  Migration   $$$$$")
            new_pop = Initialization(n_pop_size=n_pop_size*2-10, n_feature=_sel_Num)
            population = population[:10] + new_pop
        n=n+1
    #####################################################################################
    print("---------------  End of Generation  ---------------")
    fitScore_list = []
    for individual in population:
        fitScore_list.append(Evaluation(individual))
    _sum_fitScore = sum(fitScore_list)
    fitScore_ratio_list = []
    for i in range(len(fitScore_list)):
        fitScore_ratio_list.append(fitScore_list[i] / _sum_fitScore)
    print("%-8s  %-6s  %-6s  %s "%("Fit Val","RMSE","R2","Portion"))
    for i in range(3):
        _r2, _rmse = R2_RMSE(population[i])
        print("%-8.4f  %-6.4f  %-6.4f  %.3f%% "%(fitScore_list[i], _rmse, _r2, fitScore_ratio_list[i]*100), population[i])

    _epoch_log_str = ""
    for i in range(len(total_r2_log)):
        _epoch_log_str += "%f,%f\n"%(total_r2_log[i], total_rmse_log[i])

    if pop_info==None:
        fw = open("%s_ga_%s_%s.log"%(ds.data_name,model,met),'w')
        print("%s_ga_%s_%s.log  is saved!"%(ds.data_name,model,met))
    elif pop_info :
        fw = open("%s_ga_%s_%s.log"%(ds.data_name,model,met),'a')
        print("%s_ga_%s_%s.log  is updated!"%(ds.data_name,model,met))
    fw.write(_epoch_log_str)
    fw.close()

    return population[0], population

class GetModel:
    
    def __init__(self,feature_set,model_algo,n_component):
        self.data = ds.read_csv('.train','ID')
        self.X_data = self.data.iloc[:,3:]
        self.y_data = pd.DataFrame(self.data.loc[:,'EP'], columns=np.array(['EP']))
        self.feature_set = feature_set
        self.model = model_algo
        self.n_component = n_component
        
        self.index = []
        for i in self.feature_set:
            self.index.append(int(np.where(self.X_data.columns.values == i)[0][0]))
    
    def k_fold(self,k=5,run=1000):
        """
        Repeat K-fold 'run' times and summary the best implementation.
    
        Parameters
        ----------
        X_data : pandas DataFrame , shape = (n_samples, n_features)
        y_data : pandas DataFrame , shape = (n_samples,)
        feature_set : list, set of features that make up model
        run : int, number of implementation
        k : int, k of 'K'-Fold
    
        Returns
        -------
        None
        """
        gingerbreadman = []
        n=0
        while n < run :
            n += 1
            Q2=[]
            Q2_pearson=[]
            R2=[]
            R2_pearson=[]
            coef = []
            intercept = []
            trainset_index=[]
            testset_index=[]
            x = self.X_data.loc[:,self.feature_set].values
            y = self.y_data.values
            kf = KFold(n_splits=k,shuffle=True)
            predY = np.zeros_like(y)

            for train,test in kf.split(x):
                scaler = MinMaxScaler()
                scaler.fit(x[train])
                xtrain = scaler.transform(x[train])
                xtest = scaler.transform(x[test])

                if self.model == "MLR":
                    clf = LinearRegression()
                elif self.model == "PLS":
                    clf = PLSRegression(n_components=len(self.feature_set))
            
                clf.fit(xtrain,y[train])
                predY[test] = clf.predict(xtest)

                rs = clf.score(xtrain,y[train])
                R2.append(rs)
                qs = clf.score(xtest,y[test])
                Q2.append(qs)
                
                _temp_perdict_y = []
                for _list in clf.predict(xtrain):
                    for value in _list:
                        _temp_perdict_y.append(value)
                _temp_y = []
                for _list in y[train]:
                    for value in _list:
                        _temp_y.append(value)
                pearson_r, p = stats.pearsonr(np.array(_temp_perdict_y), np.array(_temp_y))
                rs = pearson_r**2
                R2_pearson.append(rs)
    
                _temp_perdict_y = []
                for _list in clf.predict(xtest):
                    for value in _list:
                        _temp_perdict_y.append(value)
                _temp_y = []
                for _list in y[test]:
                    for value in _list:
                        _temp_y.append(value)
                pearson_r, p = stats.pearsonr(np.array(_temp_perdict_y), np.array(_temp_y))
                qs = pearson_r**2
                Q2_pearson.append(qs)
                #print(rs)
                #print("---------------------------")
    
                coe = clf.coef_
                coef.append(coe)
    
                if self.model == "MLR":
                    inte = clf.intercept_
                    intercept.append(inte)
                elif self.model == "PLS":
                    intercept.append([float(0)])
    
                trainset_index.append(train)
                testset_index.append(test)
    
            rmse = np.sqrt(mean_squared_error(predY,y))
            maxq2 = np.max(Q2)
            index = Q2.index(maxq2)
            mid = []
            mid.append(np.mean(np.array(Q2))) #ginger[0]
            mid.append(np.mean(np.array(R2))) #ginger[1]
            mid.append(rmse) #ginger[2]
            mid.append(coef[index]) #ginger[3]
            mid.append(intercept[index]) #ginger[4]
            mid.append(trainset_index[index])
            mid.append(testset_index[index])
            mid.append(np.mean(np.array(Q2_pearson)))
            mid.append(np.mean(np.array(R2_pearson)))
            gingerbreadman.append(mid)
    
        gingerbreadman.sort()
        best =  gingerbreadman[-1]
    
        print ('sklearn R^2CV mean: {:.6}'.format(best[1]))
        print ('sklearn Q^2CV mean: {:.6}'.format(best[0]))
        print ('RMSE CV : {:.6}'.format(best[2]))
        print ('Features set =', self.feature_set)

        train_ind = best[5]
        test_ind = best[6]

        #print trainind,testind
    
        pred_plotY = np.zeros_like(y)
        g_mlr = LinearRegression()
        g_mlr.fit(x[train_ind],y[train_ind])

        pred_plotY[train_ind] = g_mlr.predict(x[train_ind])
        pred_plotY[test_ind] = g_mlr.predict(x[test_ind])

        label_train = np.repeat('Train',len(y[train_ind]))
        label_test = np.repeat('Test',len(y[test_ind]))
        
        ID = self.X_data.index.values
        train_frame = pd.DataFrame(np.array([y[train_ind].T[0],pred_plotY[train_ind].T[0],label_train,ID[train_ind]]).T,
                                  columns = np.array(['Actual','Predict','Label','ID']))
        test_frame = pd.DataFrame(np.array([y[test_ind].T[0],pred_plotY[test_ind].T[0],label_test,ID[test_ind]]).T,
                                  columns = np.array(['Actual','Predict','Label','ID']))        
        final_frame = pd.concat([train_frame,test_frame])
        work_path = os.getcwd().replace(sys.path[0]+'/','')        
        LABEL = np.unique(final_frame['Label'].values)
        color_cmap = factor_cmap('label', palette=['red','blue'], factors=LABEL)
        source = ColumnDataSource(data=dict(x=np.around(final_frame['Actual'].values.astype('float64'),2),
                                            y=np.around(final_frame['Predict'].values.astype('float64'),2),
                                            desc=final_frame['ID'].values.astype('str'),
                                            imgs=['%s/Image/%s.png'%(work_path,x) for x in final_frame['ID'].values],
                                            label = final_frame['Label'].values
                                           )
                                 )
        hover = HoverTool(
                tooltips="""
                <div>
                    <div>
                        <img
                            src="@imgs" height="100" alt="@imgs" width="100"
                            style="float: left; margin: 0px 15px 15px 0px;"
                            border="2"
                        ></img>
                    </div>
                    <div>
                        <span style="font-size: 17px; font-weight: bold;">@desc</span>
                        <span style="font-size: 15px; color: #966;">[$index]</span>
                    </div>
                    <div>
                        <span style="font-size: 15px;">EP</span>
                        <span style="font-size: 15px; color: #696;">@x</span>
                    </div>
                    <div>
                        <span style="font-size: 15px;">Predict</span>
                        <sapn style="font-size: 15px; color: #696;">@y</span>
                    </div>
                </div>
                """,
                names=['Scatter']
            )
        p = figure(title='Cross Validation Best Train/Validation Figure',plot_width=1000, plot_height=600,tools=[hover])
        p.scatter('x','y',source=source, name='Scatter', size=10,fill_color=color_cmap,legend_group='label')
        
                                            
        par1 = np.polyfit(train_frame['Actual'].values.astype('float64'), train_frame['Predict'].values.astype('float64'), 1, full=True)
        slope1=par1[0][0]
        intercept1=par1[0][1]
        y_predicted1 = [slope1*i + intercept1  for i in train_frame['Actual'].values.astype('float64')]

        par2 = np.polyfit(test_frame['Actual'].values.astype('float64'), test_frame['Predict'].values.astype('float64'), 1, full=True)
        slope2=par2[0][0]
        intercept2=par2[0][1]
        y_predicted2 = [slope2*i + intercept2  for i in test_frame['Actual'].values.astype('float64')]

        p.line(train_frame['Actual'].values.astype('float64'),y_predicted1,line_color='blue')
        p.line(test_frame['Actual'].values.astype('float64'),y_predicted2, line_color='red')
        p.legend.location = 'top_left'
        #p.legend.label_text_align='left'
        show(p)
        
        CV_R2 = float(best[1])
        CV_Q2 = float(best[0])
        CV_RMSE = float(best[2])
        
        self.CV_R2 = CV_R2
        self.CV_Q2 = CV_Q2
        self.CV_RMSE = CV_RMSE
       

    
    def features_table(self) :
        """
        Show feature vlaues table

        Returns
        -------
        table
        """
        feature_num = len(self.feature_set)
        desc = DataFrame(self.X_data, columns=self.feature_set)
        result = pd.concat([desc, self.y_data], axis=1, join='inner')
        error_val = np.array(result['EP'].values) - self.pred_y
        result.insert(feature_num+1,'Predict',self.pred_y)
        result.insert(feature_num+2,'Error',error_val)
        
        
        
        return result
    
    def train_plot(self) :
        """
        Show prediction training plot

        Returns
        -------
        None
        """
        x = self.X_data.loc[:,self.feature_set]
        y = pd.DataFrame(self.y_data, columns=np.array(['EP']))
        if self.model == "MLR":
            g_mlrr = LinearRegression()
            
        elif self.model == "PLS":
            g_mlrr = PLSRegression(n_components=self.n_component)
        g_mlrr.fit(x,y)
        score = g_mlrr.score(x,y)
        self.score = score
        pred_y = g_mlrr.predict(x).T[0]
        mol_id = x.index.values
        y.insert(1,'Predict',pred_y)
        y.insert(2,'Name',self.data.index.values)
        y.insert(3,'Smiles',self.data['SMI'].values)
        self.y = y
        output_notebook(INLINE)
        work_path = os.getcwd().replace(sys.path[0]+'/','')
        source = ColumnDataSource(
                            data=dict(
                                x=np.around(y['EP'].values,2),
                                y=np.around(y['Predict'].values,2),
                                desc=y['Name'].values.astype('str'),
                                imgs=['%s/Image/%s.png'%(work_path,x) for x in y['Name'].values]
                                )
                            )
        hover = HoverTool(
                tooltips="""
                <div>
                    <div>
                        <img
                            src="@imgs" height="100" alt="@imgs" width="100"
                            style="float: left; margin: 0px 15px 15px 0px;"
                            border="2"
                        ></img>
                    </div>
                    <div>
                        <span style="font-size: 17px; font-weight: bold;">@desc</span>
                        <span style="font-size: 15px; color: #966;">[$index]</span>
                    </div>
                    <div>
                        <span style="font-size: 15px;">EP</span>
                        <span style="font-size: 15px; color: #696;">@x</span>
                    </div>
                    <div>
                        <span style="font-size: 15px;">Predict</span>
                        <sapn style="font-size: 15px; color: #696;">@y</span>
                    </div>
                </div>
                """,
                names=['Scatter']
            )
        p = figure(plot_width=1000, plot_height=600,tools=[hover])
        p.circle('x','y',source=source, name='Scatter', size=10)        
        
        if self.model == 'PLS':
            coef = g_mlrr.coef_.T[0]
       
        else:
            coef = g_mlrr.coef_[0]
        
        par1 = np.polyfit(y['EP'].values.astype('float64'), y['Predict'].values.astype('float64'), 1, full=True)
        slope1=par1[0][0]
        intercept1=par1[0][1]
        y_predicted1 = [slope1*i + intercept1  for i in y['EP'].values.astype('float64')]
        p.line(y['EP'].values.astype('float64'),y_predicted1,line_color='blue')
        self.pred_y = pred_y
        self.coef = coef

        display(pd.DataFrame(coef,index=np.array(self.feature_set),columns=np.array(['Coef Value'])).T)
        show(p)
        
    def feature_corr(self) :
        """
        Correlation coefficient of features table

        Returns
        -------
        table
        """
        X = self.X_data.loc[:,self.feature_set]
        result = pd.concat([X, self.y_data], axis=1, join='inner')
        g = sns.pairplot(result,kind='reg',diag_kind='kde')
        g.map_lower(corrfunc)
        g.map_upper(corrfunc)      
        plt.show()
        
        return result.corr()

    
    def save(self):
        data_name = ds.data_name
        X_train = self.X_data
        y_train = self.y_data
        file_name = ds.load('.info')
        coef = self.coef
        pred_y = self.pred_y          
        header = X_train.columns.values

        with open(file_name,'rb') as fr:
            scaler = pickle.load(fr)
        which_scale = file_name.split('.info')[0].split('_')[-1]
        if which_scale == 'minmax':

            data_scale = scaler.scale_[self.index]
            minimum = scaler.min_[self.index]
            data_range = scaler.data_range_[self.index]
            data_min = scaler.data_min_[self.index]
            data_max = scaler.data_max_[self.index]
            description = np.array(['Per feature relative scaling of the data. Equivalent to (max - min) / (X.max(axis=0) - X.min(axis=0))',
                                    'Per feature adjustment for minimum. Equivalent to min - X.min(axis=0) * self.scale_',
                                    'Per feature minimum seen in the data',
                                    'Per feature maximum seen in the data',
                                    'Per feature range seen in the data',
                                    'Coef value of each feature'])
            frame = {'Scale':data_scale,'Min':minimum,'Data Range':data_range,'Data Min':data_min,'Data Max':data_max,'Coef':coef}
            info = pd.DataFrame(frame,index=np.array(self.feature_set)).T
            info.loc[:,'Description'] = description
            
        elif which_scale == 'standard':           
            data_scale = scaler.scale_[self.index]
            mean = scaler.mean_[self.index]
            var = scaler.var_[self.index]
            description = np.array(['Per feature relative scaling of the data to achieve zero mean and unit variance',
                                    'The mean value for each feature in the training set',
                                    'The variance for each feature in the training set',
                                    'Coef value of each feature',
                                    'PaDEL descriptor index'])
            frame = {'Scale':data_scale,'Mean':mean,'Var':var, 'Coef':coef,'Padel Index':self.index}
            info = pd.DataFrame(frame,index=np.array(self.feature_set)).T
            info.loc[:,'Description'] = description

        
        elif which_scale == 'robust':
            data_scale = scaler.scale_[self.index]
            center = scaler.center_[self.index]
            description = np.array(['The (scaled) interquartile range for each feature in the training set (Same as IQR).',
                                    'The median value for each feature in the training set.'])
            frame = {'Scale':data_scale,'Center':center}
            info = pd.DataFrame(frame,index=np.array(self.feature_set)).T
            info.loc[:,'Description'] = description
        
        result = self.features_table()
        
        R2_CV = self.CV_R2
        Q2_CV = self.CV_Q2
        RMSE_CV = self.CV_RMSE
        
        CV_df = pd.DataFrame(data=np.array([R2_CV,Q2_CV,RMSE_CV]))
        CV_df = CV_df.rename(index={0:'R2_CV',1:'Q2_CV',2:'RMSE_CV'},columns={0:'Value'})
        CV_df.loc['Model_Algo'] = self.model
        CV_df.loc['Preprocessing'] = which_scale
        
        algorithm = input('Selected Algorithm in Feature Selection (MC/GA) : ')
        model = input('Selected Model (MLR/PLS) : ')
        
        file_name = input('Enter model name to  save as\n(default) %s_%s_%s.model\n- : '%(ds.data_name,algorithm,model))
        if file_name == '':
            file_name = '%s_%s_%s.model'%(ds.data_name,algorithm,model)
        else:
            file_name = file_name + '_%s_%s.model'%(algorithm,model)
            
        info.to_csv(file_name, mode='w',index=True, index_label='Row')
        fw = open(file_name,'a')
        fw.write('--------------------------------\n')
        fw.close()
        
        result.to_csv(file_name,mode='a', index=True, index_label='ID')
        fw = open(file_name,'a')
        fw.write('--------------------------------\n')
        fw.close()
        
        CV_df.to_csv(file_name,mode='a', index=True, index_label = 'Row')
        
        print('%s file saved'%file_name)

        
def load_model():
    model_name = ds.load('.model')
    
    fr = open(model_name,'r')
    a = np.array(fr.readlines())
    division_spot = np.where(a=='--------------------------------\n')[0]
    _temp1 = a[:division_spot[0]]
    _temp2 = a[division_spot[0]+1:division_spot[1]]
    _temp3 = a[division_spot[1]+1:]
    
    fw = open('_temp1','w')
    for line in _temp1:
        fw.write(line)
    fw.close()
    
    Scaler = pd.read_csv('_temp1', index_col='Row')
    os.remove('_temp1')
    
    fw = open('_temp2','w')
    for line in _temp2:
        fw.write(line)
    fw.close()
    
    Feature = pd.read_csv('_temp2', index_col='ID')
    os.remove('_temp2')
    
    fw = open('_temp3','w')
    for line in _temp3:
        fw.write(line)
    fw.close()
    
    CV = pd.read_csv('_temp3',index_col='Row')
    os.remove('_temp3')

    display(Scaler, Feature, CV)
    
    return Scaler, Feature, CV

class ModelTest:
    def __init__(self):
        Scaler , Feature, CV = load_model()
        self.scaler = Scaler
        self.Feature = Feature
        self.CV = CV
        self.method = CV.loc['Preprocessing'].values[0]
        self.model_algo = CV.loc['Model_Algo',:].values[0]
        
        if self.model_algo == "MLR":
            self.g_mlr = LinearRegression()
        elif self.model_algo == 'PLS':
            self.g_mlr = PLSRegression(n_components=len(Feature.columns.values)-3)
    
    def model_test(self,test=True, scaled=True):
    
        Feature = self.Feature
        Scaler = self.scaler
        CV = self.CV
        X_train = Feature.iloc[:,:-3]
        y_train = pd.DataFrame(Feature['EP'])
        feature_name = X_train.columns.values
        model_algo = CV.loc['Model_Algo',:].values[0]
              
        g_mlr = self.g_mlr
        g_mlr.fit(X_train,y_train)
        
        
        train_val = np.repeat('Train',X_train.shape[0])
        train_scatter = Feature.loc[:,['EP','Predict']]
        train_scatter.insert(2,'Name',X_train.index.values)
        train_scatter.insert(3,'Label',train_val)
        
        if test==True:
            X_test, y_test = split_xy('.test')
            
            X_test = X_test.loc[:,feature_name]
            if scaled==False: 
                if self.method == 'standard':
                    for i in feature_name:
                        scaled = (X_test[i].values - Scaler.loc['Mean',i]) / sqrt(Scaler.loc['Var',i])
                        X_test[i] = scaled
                elif self.method == 'minmax':
                    for i in feature_name:
                        scaled = (X_test[i].values - Scaler.loc['Data Min',i]) / (Scaler.loc['Data Max',i] - Scaler.loc['Data Min',i])
                        X_test[i] = scaled      
            pred_test_y = pd.DataFrame(g_mlr.predict(X_test), index=X_test.index.values, columns=np.array(['Predict']))
            test_val = np.repeat('Test',X_test.shape[0])
            test_scatter = pd.concat([y_test,pred_test_y],axis=1)
            test_scatter.insert(2,'Name',X_test.index.values)
            test_scatter.insert(3,'Label',test_val)
                
            final = pd.concat([train_scatter,test_scatter])

            work_path = os.getcwd().replace(sys.path[0]+'/','')        
            LABEL = np.unique(final['Label'].values)
            color_cmap = factor_cmap('label', palette=['red','blue'], factors=LABEL)
            source = ColumnDataSource(data=dict(x=np.around(final['EP'].values.astype('float64'),2),
                                                y=np.around(final['Predict'].values.astype('float64'),2),
                                                desc=final['Name'].values.astype('str'),
                                                imgs=['%s/Image/%s.png'%(work_path,x) for x in final['Name'].values],
                                                label = final['Label'].values
                                               )
                                     )
            hover = HoverTool(
                    tooltips="""
                    <div>
                        <div>
                            <img
                                src="@imgs" height="100" alt="@imgs" width="100"
                                style="float: left; margin: 0px 15px 15px 0px;"
                                border="2"
                            ></img>
                        </div>
                        <div>
                            <span style="font-size: 17px; font-weight: bold;">@desc</span>
                            <span style="font-size: 15px; color: #966;">[$index]</span>
                        </div>
                        <div>
                            <span style="font-size: 15px;">EP</span>
                            <span style="font-size: 15px; color: #696;">@x</span>
                        </div>
                        <div>
                            <span style="font-size: 15px;">Predict</span>
                            <sapn style="font-size: 15px; color: #696;">@y</span>
                        </div>
                    </div>
                    """,
                    names=['Scatter']
                )
            p = figure(title='Train/Validation Set Figure',plot_width=1000, plot_height=600,tools=[hover])
            p.scatter('x','y',source=source, name='Scatter', size=10,fill_color=color_cmap,legend_group='label')
            
                                                
            par1 = np.polyfit(train_scatter['EP'].values.astype('float64'), train_scatter['Predict'].values.astype('float64'), 1, full=True)
            slope1=par1[0][0]
            intercept1=par1[0][1]
            y_predicted1 = [slope1*i + intercept1  for i in train_scatter['EP'].values.astype('float64')]
    
            par2 = np.polyfit(test_scatter['EP'].values.astype('float64'), test_scatter['Predict'].values.astype('float64'), 1, full=True)
            slope2=par2[0][0]
            intercept2=par2[0][1]
            y_predicted2 = [slope2*i + intercept2  for i in test_scatter['EP'].values.astype('float64')]
    
            p.line(train_scatter['EP'].values.astype('float64'),y_predicted1,line_color='blue')
            p.line(test_scatter['EP'].values.astype('float64'),y_predicted2, line_color='red')
            p.legend.location = 'bottom_right'
            self.plot = p
            show(p)
            pearson_r, p = stats.pearsonr(pred_test_y['Predict'].values, np.array(y_test['EP'].values))
            rmse = mean_squared_error(np.array(y_test), np.array(pred_test_y['Predict'].values), squared=False)
            print('Test R2 : %5f'%pearson_r**2)
            print('Test RMSE : %5f'%rmse)
            
            
            
        elif test == False:
            display(Feature.iloc[:,-3:])
            display(CV.iloc[:3,:])
            
    def editor(self):
        pn.extension("jsme", sizing_mode="stretch_width")
        editor = JSMEEditor(height=500, format="smiles")
        values = pn.Param(editor, parameters=["value"], widgets={
            "value": {"type": pn.widgets.TextAreaInput,"height":50,"width":400,"name":"SMILES","disabled":True}
            })
        
        btn = pn.widgets.Button(name='Evaluation',button_type='primary')
        text = pn.widgets.TextAreaInput(value='Ready',height=320, width=400)
        checkbox = pn.widgets.Checkbox(name='New File?',value=True)
        file_name = pn.widgets.TextAreaInput(placeholder='Enter File Name; File Extension is .csv ', height=30,width=400)
        
        global result_val
        result_val = ''
        def result(event):
            global result_val
            try:
                feature = self.Feature.columns.values[:-3]
                scaler = self.scaler
                method = self.method
                smiles = editor.value
                num = '%d'.zfill(5)%btn.clicks
                Draw.MolToFile(Chem.MolFromSmiles(smiles),'Image/%s_%s.png'%(file_name.value,num))
                descriptors = from_smiles(smiles)
                content = ''
                val_list=[]
                
                if method == 'standard':
                    for i in feature:
                        scaled = (float(descriptors[i]) - scaler.loc['Mean',i]) / sqrt(scaler.loc['Var',i])
                        val_list.append(scaled)
                        content += '%s\t%s\t%s\n'%(i,str(round(float(descriptors[i]),2)),round(scaled,2))
                elif method == 'minmax':
                    for i in feature:
                        scaled = (float(descriptors[i]) - scaler.loc['Data Min',i]) / (scaler.loc['Data Max',i] - scaler.loc['Data Min',i]) 
                        val_list.append(scaled)
                        content += '%s\t%s\t%s\n'%(i,str(round(float(descriptors[i]),2)),round(scaled,2)) 
                val_list = pd.DataFrame(np.array(val_list),index=feature).T
                result_val += '{0} Molecule\n'.format(btn.clicks)
                result_val += 'Smiles : ' + smiles + '\n'
                result_val += 'Feature\tVal\tScaled\n'
                result_val += content
                ep = self.g_mlr.predict(val_list)
                val_list.insert(0,'Smiles',smiles)
                val_list.insert(1,'Predict',ep)
                val_list.insert(2,'Name','%s_%s'%(file_name.value,num))
                if checkbox.value == True:
                    val_list.to_csv('%s.csv'%file_name.value,mode='w',index=False, header=True)
                else:
                    val_list.to_csv('%s.csv'%file_name.value,mode='a',index=False, header=False)
                result_val += 'Predict = %s\n\n'%str(round(ep[0][0],2))
                result_val = result_val.expandtabs(20)
                text.value = result_val
                checkbox.value = False
            except:
                text.value = 'Some error has occured, check smiles or file name'
        btn.on_click(result)
        
        display(pn.Column(pn.Row(editor,pn.Column(values,checkbox,file_name,text)),btn))

    def new_molecule(self,file_name):
        data = pd.read_csv(file_name)
        min_val = np.min(self.Feature['EP'].values)
        work_path = os.getcwd().replace(sys.path[0]+'/','')
        source = ColumnDataSource(data=dict(x=np.repeat(min_val,data.shape[0]),
                                           y=data['Predict'].values,
                                           label = np.repeat('External',data.shape[0]),
                                           desc=data['Name'].values,
                                           imgs=['%s/Image/%s.png'%(work_path,x) for x in data['Name'].values],
                                          )
                                )
        #plot = Circle(x='x',y='y',fill_color='green',name='Scatter',size=10)
        p = self.plot
        p.circle(x='x',y='y',fill_color='green',name='Scatter',size=10,source=source,legend_label='External')
        
        show(p)
                                 
        
