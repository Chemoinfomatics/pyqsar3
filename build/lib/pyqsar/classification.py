#-*- coding: utf-8 -*-
# for classification model_selection

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import pickle
from pyqsar import data_setting as ds

class Predict:
    
    def __init__(self,X_train,y_train,X_test,y_test,n, scaled=True):
        self.X_train = X_train
        self.y_train = y_train
        if scaled == True:
            self.X_test = X_test
            self.y_test = y_test
        else:
            scaler_file = ds.load('.info')
            with open(scaler_file,'rb') as fr:
                scaler = pickle.load(fr)
            self.X_test = pd.DataFrame(scaler.fit_transform(X_test),index=X_test.index.values,columns=X_test.columns.values)
            self.y_test = y_test
        self.n = n
        self.model_list = {}
        #self.parameter = param
    
    def feature_selection(self,**parameter):
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test
        n = self.n
        header = X_train.columns.values
        
        model = XGBClassifier(**parameter)
        eval_set = [(X_test,y_test)]
        model.fit(X_train, y_train, eval_set = eval_set, verbose=False)
        print(model)
        print('\n')
        importances = model.feature_importances_
        import_index = np.argsort(importances)[::-1]

        print('\033[1m' + 'XGBoost Feature Importances' + '\033[0m')
        print('%-15s %s'%('Feature','Importances'))
        for top in range(n):
            print('%-15s %f'%(header[import_index[top]],importances[import_index[top]]))
        XGBoost_Feature = header[import_index][:4]    
        display(X_train.loc[:,XGBoost_Feature].corr())
        print('\n')    
        
        
        
        mutual_list = []
        for cnt in range(10):
            fs = SelectKBest(score_func=mutual_info_classif, k=n)
            frame = fs.fit_transform(X_train, y_train['EP'].values)
            for feature in fs.get_feature_names_out():
                mutual_list.append(feature)
        
        mutual_list = np.array(mutual_list)
        feature_cnt = {}
        for feature in np.unique(mutual_list):
            feature_cnt[feature] = np.where(mutual_list == feature)[0].shape[0]
        feature_cnt = dict(sorted(feature_cnt.items(), key=lambda item: item[1], reverse=True))
        print('\033[1m' + 'Mutual Count (x10)' + '\033[0m')
        print('%-15s %s'%('Feature','Count'))
        count = 0
        Mutual_Feature = []
        for feature, cnt in feature_cnt.items():
            print('%-15s %d'%(feature, cnt))
            Mutual_Feature.append(feature)
            count = count + 1
            if count == n:
                break
        display(X_train.loc[:,Mutual_Feature].corr())
        print('\n')
        
        filter_index = np.where(importances !=0)[0]
        filter_feature = header[filter_index]
        X_train_filter = X_train.loc[:,filter_feature]
        
        mutual_list_filter = []
        for cnt in range(10):
            fs = SelectKBest(score_func=mutual_info_classif, k=n)
            frame = fs.fit_transform(X_train_filter, y_train['EP'].values)
            for feature in fs.get_feature_names_out():
                mutual_list_filter.append(feature)
        
        mutual_list_filter = np.array(mutual_list_filter)
        feature_cnt_filter = {}
        for feature in np.unique(mutual_list_filter):
            feature_cnt_filter[feature] = np.where(mutual_list_filter == feature)[0].shape[0]
        feature_cnt_filter = dict(sorted(feature_cnt_filter.items(), key=lambda item: item[1], reverse=True))
        print('\033[1m' + 'Mutual Count After XGBoost (x10)' + '\033[0m')
        print('%-15s %s'%('Feature','Count'))
        count = 0
        Mutual_Feature_Filter = []
        for feature, cnt in feature_cnt_filter.items():
            print('%-15s %d'%(feature, cnt))
            Mutual_Feature_Filter.append(feature)
            count = count + 1
            if count == n:
                break    
        display(X_train.loc[:,Mutual_Feature_Filter].corr())
        print('\n')
        
        select_feature = int(input('Enter Feature to use (0-XGBoost, 1-Mutual, 2-Mutual_Filter) : '))
        
        if select_feature == 0:
            self.feature = np.array(XGBoost_Feature,dtype='str')
        elif select_feature == 1:
            self.feature = np.array(Mutual_Feature,dtype='str')
        elif select_feature == 2:
            self.feature = np.array(Mutual_Feature_Filter,dtype='str')
    
    def predict(self, use_model, show=True, **parameter):
        pd.set_option('display.max_rows',None)
        X_train = self.X_train.loc[:,self.feature]
        y_train = self.y_train
        X_test = self.X_test.loc[:,self.feature]
        y_test = self.y_test  
        total = pd.concat([X_train,X_test])
        ID = total.index.values
        self.ID = ID
        eval_set = [(X_test,y_test)]
        
        if use_model == 'XGBoost':
            if 'eval_metric' in parameter.keys():
                model = XGBClassifier(**parameter)
                model.fit(X_train,y_train, eval_set = eval_set,verbose=False)
            else:
                model = XGBClassifier(**parameter)
                model.fit(X_train,y_train, eval_metric='logloss', eval_set=eval_set,verbose=False)
            
        elif use_model == 'RandomForest':
            model = RandomForestClassifier(**parameter)
            model.fit(X_train,y_train['EP'].values)
        
        elif use_model == 'DecisionTree':
            model = DecisionTreeClassifier(**parameter)
            model.fit(X_train,y_train['EP'].values)
          
        pred_y = model.predict(X_train)
        pro_y = model.predict_proba(X_train)
        label_train = np.repeat('Train',len(pred_y))
        
        pred_y_test = model.predict(X_test)
        pro_y_test = model.predict_proba(X_test)
        label_test = np.repeat('Test',len(pred_y_test))
        
        total_pred = pd.DataFrame(np.concatenate([pred_y,pred_y_test]),index=ID, columns=np.array(['Predict']))
        total_pro = pd.DataFrame(np.concatenate([pro_y,pro_y_test]),index=ID,columns=np.array(['0_Pro','1_Pro']))
        total_label = pd.DataFrame(np.concatenate([label_train, label_test]),index=ID, columns=np.array(['Label']))
        total_y = pd.concat([y_train,y_test])
        total_ID = pd.DataFrame(ID,index=ID,columns=np.array(['ID']))
        final_frame = pd.concat([total,total_y,total_pred,total_pro,total_label,total_ID],axis=1)
        if show == True:
            display(final_frame)
        #self.final_frame = final_frame


        
        
        train_confusion = confusion_matrix(y_train,pred_y)
        test_confusion = confusion_matrix(y_test,pred_y_test)
        
        train_confusion = pd.DataFrame(train_confusion, index=np.array(['Class 0','Class 1']), columns =np.array(['Class 0', 'Calss 1']))
        test_confusion = pd.DataFrame(test_confusion, index=np.array(['Class 0','Class 1']), columns =np.array(['Class 0', 'Calss 1']))
        print('Train Accuracy : %f'%accuracy_score(y_train,pred_y))
        print('Train F1 Score : %f'%f1_score(y_train,pred_y))        
        print('Train Confusion Matrix')
        display(train_confusion)
        print('Test Accuracy : %f'%accuracy_score(y_test,pred_y_test))
        print('Test F1 Score : %f'%f1_score(y_test,pred_y_test))
        print('Test Confusion Matrix')
        display(test_confusion)
        
        result = []
        result.append(pred_y)
        result.append(pred_y_test)
        final_result = []
        if use_model == 'XGBoost':
            final_result.append(result)
            final_result.append(final_frame)
            self.model_list[use_model] = final_result
        elif use_model == 'RandomForest':
            final_result.append(result)
            final_result.append(final_frame)
            self.model_list[use_model] = final_result
        elif use_model == 'DecisionTree':
            final_result.append(result)
            final_result.append(final_frame)
            self.model_list[use_model] = final_result
                             
    def plot(self):
        for model_list in self.model_list.keys():
            print('\033[1m' + '%s Classification Plot'%model_list + '\033[0m')
            print()
            df = self.model_list[model_list][1]

            ID = self.ID
            pd.options.plotting.backend = "plotly"
            
            df_0 = df[df['EP']==0]
            df_1 = df[df['EP']==1]
            groups = np.unique(df['Label'].values)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            class_val = np.unique(df['EP'].values)
            cnt=0
            class0_index = df_0.shape[0]
            fig.update_layout(xaxis2= {'anchor': 'y', 'overlaying': 'x', 'side': 'top'})
            label_color = {'Train':'rgba(246, 78, 139, 0.3)','Test':'rgba(58, 71, 80, 0.3)'}
            for group in groups:
                df_group = df_0[df_0['Label']==group]
                if cnt ==0:
                    cnt = df_group.shape[0]
                    fig.add_trace(go.Scatter(x=np.arange(0,cnt,1), y=df_group['0_Pro'],mode='markers', 
                                  name='Class 0 %s'%group,text=['ID : %s'%str(x) for x in df_group['ID']]),secondary_y=False)
                else:
                    fig.add_trace(go.Scatter(x=np.arange(cnt,class0_index+1,1), y=df_group['0_Pro'],mode='markers', 
                                  name='Class 0 %s'%group,text=['ID : %s'%str(x) for x in df_group['ID']]),secondary_y=False)
                cnt_0, bins_0 = np.histogram(df_group['0_Pro'],bins=12,range=(0,1))
                cnt_0 = -cnt_0        
                fig.add_trace(go.Bar(x=cnt_0,y=bins_0,xaxis='x2',orientation='h', name='Class 0 %s'%group,
                                     text=['[%.2f,%.2f]'%(bins_0[x],bins_0[x+1]) for x in range(12)],
                                     marker=dict(color=label_color[group],line=dict(color='rgba(58, 71, 80, 1.0)', width=1))))
                fig.update_layout(barmode='stack')
            label_color = {'Train':'rgba(135,206,250, 0.3)','Test':'rgba(206,126,0, 0.3)'}    
            cnt=0
            for group in groups:
                df_group = df_1[df_1['Label']==group]
                if cnt ==0:
                    cnt = df_group.shape[0]
                    fig.add_trace(go.Scatter(x=np.arange(class0_index+2,class0_index+cnt+2,1), y=df_group['1_Pro'],mode='markers', 
                                  name='Class 1 %s'%group,text=['ID : %s'%str(x) for x in df_group['ID']]),secondary_y=True)
                else:
                    fig.add_trace(go.Scatter(x=np.arange(class0_index+cnt+2,df.shape[0]+2,1), y=df_group['1_Pro'],mode='markers', 
                                  name='Class 1 %s'%group,text=['ID : %s'%str(x) for x in df_group['ID']]),secondary_y=True)
                cnt_1, bins_1 = np.histogram(df_group['1_Pro'],bins=12,range=(0,1))
                fig.add_trace(go.Bar(x=cnt_1,y=bins_1,xaxis='x2',yaxis='y2',orientation='h', name='Class 1 %s'%group,
                                     text=['[%.2f,%.2f]'%(bins_0[x],bins_0[x+1]) for x in range(12)],
                                     marker=dict(color=label_color[group],line=dict(color='rgba(58, 71, 80, 1.0)', width=1))))
                fig.update_layout(barmode='stack')
                
            #fig.add_trace(go.Scatter(x=np.arange(0,len(ID),1), y=df_0['0_Pro'],mode='markers', text=df_0['ID']),secondary_y=False)
            #fig.add_trace(go.Scatter(x=np.arange(0,len(ID),1), y=df_1['1_Pro'],mode='markers', text=df_1['ID']),secondary_y=True)
            fig['layout']['yaxis2']['autorange']='reversed'
            #fig.update_layout(barmode='stack')
            
            #cnt_0, bins_0 = np.histogram(df_0['0_Pro'])
            #cnt_0 = -cnt_0
            
            #cnt_1, bins_1 = np.histogram(df_1['1_Pro'])
            #max_cnt = max(np.max(abs(cnt_0)),np.max(abs(cnt_1)))
    
            #fig.add_trace(go.Bar(x=cnt_0,y=bins_0,xaxis='x2',orientation='h',
            #                     marker=dict(color='rgba(246, 78, 139, 0.3)',line=dict(color='rgba(246, 78, 139, 0.3)', width=3))))
            #fig.add_trace(go.Bar(x=cnt_1,y=bins_1,xaxis='x2',yaxis='y2',orientation='h',
            #                     marker=dict(color='rgba(58, 71, 80, 0.3)',line=dict(color='rgba(58, 71, 80, 0.3)', width=3))))
            fig.add_shape(type='line',
                          x0=0,
                          y0=0.5,
                          x1=df.shape[0]+2,
                          y1=0.5,
                          line=dict(color='rgba(191,188,188,1)',dash='dot'),
                          xref='x',
                          yref='y'
                         )
            fig.add_shape(type='line',
                          x0=class0_index+1,
                          y0=0,
                          x1=class0_index+1,
                          y1=1,
                          line=dict(color='rgba(191,188,188,1)',dash='dot'),
                          xref='x',
                          yref='y'
                         )
    
            fig.update_layout(xaxis2 = dict(tickmode='auto',
                                            range=(-df_0.shape[0]-1,df_1.shape[0]+1),
                                            title='Count'
                                           ),
                              xaxis1 = dict(title='x axis',
                                           range=(-1,df.shape[0]+2)
                                           ),
                              yaxis1 = dict(title='Probability Class 0'),
                              yaxis2 = dict(title='Probability Class 1')
                           )
                                          
          
            #fig.update_layout(xaxis2=dict(tickmode='array',
            #                              tickvals = np.arange(-max_cnt,max_cnt,10)))
            #fig1.add_trace(df_1.plot.scatter(x='EP',y='1_Pro', color='Label',hover_data=['ID']))
            #fig.add_annotation(x=0.4,y=0.8, text="Class 1", showarrow=False)
            #fig.add_annotation(x=0.6,y=0.2, text="Class 0", showarrow=False)
            fig.show()
            
    def comparison(self):
        y_train = self.y_train
        y_test = self.y_test
        for model_list in self.model_list.keys():
            print('\033[1m' + '%s Result'%model_list + '\033[0m')
            result = self.model_list[model_list][0]
            
            train_confusion = confusion_matrix(y_train,result[0])
            test_confusion = confusion_matrix(y_test,result[1])
            
            train_confusion = pd.DataFrame(train_confusion, index=np.array(['Class 0','Class 1']), columns =np.array(['Class 0', 'Calss 1']))
            test_confusion = pd.DataFrame(test_confusion, index=np.array(['Class 0','Class 1']), columns =np.array(['Class 0', 'Calss 1']))
            
            print('Train Accuracy : %f'%accuracy_score(y_train,result[0]))
            print('Train F1 Score : %f'%f1_score(y_train,result[0]))
            print('Train Confusion Matrix')            
            display(train_confusion)
            
            print('Test Accuracy : %f'%accuracy_score(y_test,result[1]))
            print('Test F1 Score : %f'%f1_score(y_test,result[1]))
            print('Test Confusion Matrix')
            display(test_confusion)
            
            print('\n')
            print('\n')