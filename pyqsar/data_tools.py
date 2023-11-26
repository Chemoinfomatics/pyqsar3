#-*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from pandas.plotting import scatter_matrix
from ipywidgets import interact, interactive
import ipywidgets as widgets
from IPython.display import display, Image
from math import isinf
from ipywidgets import interact, interact_manual
import os, sys
from pyqsar import data_setting as ds
import pickle
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools

__version__='2023-11-20'

def man(image_file):
    img = Image(sys.path[0]+'/man/'+image_file)
    display(img)

def VCHECK():
    import scipy
    import rdkit 
    import sklearn
    import bokeh
    import py3Dmol
    import IPython
    import ipywidgets
    import pubchempy
    #import xgboost
    #from mpl_toolkits import basemap
    #dt.Ver() #2023-11-20
    #! python -V #3.9.16
    #print ('pyqsar:', dt.__version__)
    print ('pyqsar:', __version__)
    print ('Numpy :', np.__version__)
    print ('Scipy :', scipy.__version__)
    print ('pandas :', pd.__version__)
    print ('RDkit :', rdkit.__version__)
    print ('sklearn :', sklearn.__version__)
    print ('bokeh :', bokeh.__version__)
    print ('py3Dmol :', py3Dmol.__version__)
    print ('IPython :', IPython.__version__)
    print ('ipywidgets :', ipywidgets.__version__)
    #print ('mpl_toolkits :', basemap.__version__)
    print ('pubchempy :', pubchempy.__version__)
    #print ('xgboost :', xgboost.__version__)
    # Python Version Check, 2022.02 Rdkit doesn't works with version 3.8 or higher
    return

def uCheck(filtered):
    import matplotlib.pyplot as plt
    unique=[]
    for column in filtered.columns:
        unique_values = filtered[column].unique()
        nunique=len(unique_values)
        if nunique <10:
            unique.append(nunique)
    plt.hist(unique,bins=100)
    plt.title ('Number of Unique Values')
    plt.xlabel ('Number of Unique Values')
    plt.ylabel ('Frequency')
    plt.show()

class FilteringTools:
    def __init__(self,ext='.csv'):
        self.X_data = ds.read_csv(ext)
        self.X_data_ = self.X_data.copy()
        self.nan_list = []
        self.novar_list = []
        self.inf_list = []
        self.etc_list = []
        self.etc_val_list = []

    def check_etc(self):
        X_data = self.X_data
        header = list(X_data.columns.values)[3:]
        id_list = X_data['ID'].values
        etc_list = []
        cnt = 0
        for id_val in id_list:
            check_data = X_data[X_data['ID']==id_val]
            for des in header:
                value = check_data[des].values
                try:
                    np.float(value)
                except:
                    print('%s has %s in %s'%(str(des),str(value),str(id_val)))
                    cnt = cnt + 1
                    if value not in etc_list:
                        etc_list.append(value)
        if cnt == 0:
            print('No ETC Values in Data')
        print('\n')
        print('ETC Value List')
        for etc_val in etc_list:
            print(etc_val)
        self.etc_val_list = etc_list
        
        return self.etc_val_list
    
    def rm_nan(self):
        X_data = self.X_data
        header = X_data.columns.values[3:]

        col = X_data.isnull().any()
        col_index = col.index[col.values == True]
        X_data = X_data.drop(col_index,axis=1)
        print('%-10s: %d'%('NaN',col_index.shape[0]))
        self.X_data = X_data
        return self.X_data
    
    def rm_novar(self):
        X_data = self.X_data
        zero_desc = []
        header = list(X_data.columns.values)[3:]
        cnt = 0
        try:
            for des in header:
                variance = np.var(X_data[des].values)
                if variance == 0:
                    self.novar_list.append(des)
                    zero_desc.append(des)
                    cnt = cnt +1
            X_data = X_data.drop(zero_desc, axis=1)
            print('%-10s: %d'%('No Var',cnt))
                
        except TypeError:
            print('%-10s: ETC value in Data'%'No Var')
        
        self.X_data = X_data
        
        return self.X_data
    
    def rm_inf(self):
        X_data = self.X_data
        header = list(X_data.columns.values)[3:]
        cnt = 0
        inf_desc = []
        try:
            for des in header:
                val = np.array(X_data[des])
                if np.isinf(val).any() == True:
                    self.inf_list.append(des)
                    cnt = cnt +1
                    inf_desc.append(des)
                    
            X_data = X_data.drop(inf_desc, axis=1)
            print('%-10s: %d'%('Inf',cnt))
            
        except TypeError:
            print('%-10s: ETC value in Data'%"Inf")
        
        self.X_data = X_data
        
        return self.X_data
    
    def rm_etc(self,*value):
        X_data = self.X_data
        header = X_data.columns.values
        
        for etc_val in value:
            col = X_data.iloc[:,3:].isin([etc_val]).any()
            col_index = col.index[col.values == True]

            
            print('%-10s: %d'%(etc_val,col_index.shape[0]))
        
            X_data = X_data.drop(col_index,axis=1)
        
        self.X_data = X_data
        return self.X_data
    
    def rm_all(self,*method):
        method = list(method)
        X_data = self.X_data
        if "Nan" in method:
            X_data = self.rm_nan()
            method.remove("Nan")
        
        if "Inf" in method:
            X_data = self.rm_inf()
            method.remove("Inf")
            
        if "Novar" in method:
            X_data = self.rm_novar()
            method.remove("Novar")
        for val in method:
            X_data = self.rm_etc(val)
        self.X_data = X_data
        
        print('\n')
        self.check_etc()
        print('\n\n')

        
        print('%s -> %s'%(self.X_data_.shape,self.X_data.shape))
        return self.X_data
    
    def save(self):
        ds.save(self.X_data,'filtering')
        return
    
class ScalingTools:
    def __init__(self,scale,ext='.csv'):
        self.X_data = ds.read_csv(ext)
        self.method = scale
    
    
    def train_scaler(self):
        scale_method = self.method
        X_data = self.X_data
        
        id_list = X_data['ID']
        ep_list = X_data['EP']
        smi_list = X_data['SMI']
        
        X_data = X_data.iloc[:,3:]
        header = list(X_data.columns.values)
        
        if scale_method == 'minmax':
            scaler = MinMaxScaler()
        elif scale_method == 'robust':
            scaler = RobustScaler()
        elif scale_method == 'standard':
            scaler = StandardScaler()
        else:
            print('Select Scaler in minmax, robust, standard')
        self.scaler = scaler
        scale_data = pd.DataFrame(scaler.fit_transform(X_data),columns = header)
        scale_data.insert(0,"ID",id_list)
        scale_data.insert(1,"EP",ep_list)
        scale_data.insert(2,'SMI',smi_list)
        
        self.scale_data = scale_data
        with open(f'{ds.data_name}_{scale_method}.info','wb') as fw:
            pickle.dump(scaler,fw)
        print(f'Scaler File : {ds.data_name}_{scale_method}.info file saved')
        

        return scale_data
            
    def check_outlier(self,standard):
        global X_data
        X_data = self.scale_data
        header = X_data.columns.values[3:]
        id_list = X_data['ID']
        ep_list = X_data['EP']
        smi_list = X_data['SMI']

        descriptor={}
        for des in header:
            value = X_data[des].values
            mol_index = np.where(abs(value) > standard)[0]
        
            if len(mol_index) != 0:
                descriptor[des] = mol_index

        fw = open('%s_%s_%d.out'%(ds.data_name,self.method,standard),'w')
        for key, value in descriptor.items():
            for index in value:
                fw.write('%-3s\t%s\n'%('DES',str(key)))
                fw.write('%-3s\t%s\n'%('ID',str(id_list[index])))
                fw.write('%-3s\t%s\n'%('EP',str(ep_list[index])))
                fw.write('%-3s\t%s\n'%('SMI',str(smi_list[index])))
                fw.write('%-3s\t%s\n'%('VAL',str(X_data[key][index])))

        fw.close()


        def histogram(Descriptor):
            global X_data
            PaDEL_Description(Descriptor)
            plt.figure(figsize=(10,5))
            array = X_data[Descriptor].values
            length = round(max(array)) - round(min(array))
            hist = plt.hist(array, range=[round(min(array))-1,round(max(array))+1], bins = length+2, edgecolor='black')
            plt.axvline(x=standard, color='red',linestyle='dashed')
            plt.axvline(x=-standard, color='red',linestyle='dashed')
                
            cnt, bins, patches = hist
            bin_centers = np.diff(bins)*0.5 + bins[:-1]
            for fr, x, patch in zip(cnt,bin_centers,patches):
                if fr == 0:
                    continue
                else:
                    height = fr
                    plt.annotate("{}".format(height),
                                xy = (x, height),
                                xytext = (0,0.2),
                                textcoords = 'offset points',
                                ha = 'center',
                                va = 'bottom'
                                )    
                        
            plt.show()
            over_index = descriptor[Descriptor]
            over_smi = smi_list[over_index]
            over_id = id_list[over_index]
            over_ep = ep_list[over_index]
            over_mol = [Chem.MolFromSmiles(x) for x in over_smi]
            over_val = X_data[Descriptor][over_index]
            legend = ['ID : %s\tVAL : %.2f\t'%(str(x),z) for x,z in zip(over_id,over_val)]
            img = Draw.MolsToGridImage(over_mol, molsPerRow=4, legends=legend)
            display(img)
    
            def deletion(clikced_button : widgets.Button) -> None:
                global X_data
                X_data = X_data.drop(Descriptor,axis=1)
                del descriptor[Descriptor]
                
                return X_data
            
            button = widgets.Button(description='Deletion')
            button.on_click(deletion)
            display(button)
            self.scale_data = X_data
            return X_data
        
        widgets.interact(histogram,Descriptor=descriptor.keys())
        
        
        
    def outplot(self,figsize=(2000,2000)):
        global X_data
        X_data = self.scale_data
        
        ids_list = X_data['ID']
        ep_list = X_data['EP']
        smi_list = X_data['SMI']
        
        FilePath = os.getcwd()
        FileExt = '.out'
        print(FilePath)
        directory_file = [file for file in os.listdir(FilePath) if file.endswith(FileExt)]
        
        print('%-10s   File Name'%'File Index')
        for index, file_name in enumerate(directory_file):
            print(f'{index:<10}   {file_name}')
            
        file = input('Enter File Index Num of Above List : ')
        file_name = directory_file[int(file)]
        save_name = file_name.split(FileExt)[0]
        print(f'{file_name} is selected')
        
        use_file = open(file_name,'r')
        a = use_file.readlines()
        des_list = []
        id_list = []
        for i in range(int(len(a)/5)):
            des_line = a[5*i]
            id_line = a[5*i+1]
            
            des_list.append(des_line.split('\t')[1].strip())
            id_list.append(id_line.split('\t')[1].strip())
            
        id_unique = np.unique(id_list)
        final = {}
        for i in id_unique:
            id_pos = np.where(np.array(id_list)==i)[0]
            final[i] = np.array(des_list)[id_pos]
            
        def descriptor(id_value):
            global X_data

 
            pos = np.where(np.array(ids_list).astype('str') == id_value)[0]
            smiles = smi_list[pos].values[0]
            mol = Chem.MolFromSmiles(smiles)
            for i in final[id_value]:
                PaDEL_Description(i)
                print('Descriptor Mean : %.2f'%np.mean(self.X_data[i].values))
                print('Descriptor Var : %.2f'%np.var(self.X_data[i].values))
                print('Descriptor Value : %.2f'%self.X_data[i][pos])
                print('Scaled Value : %.2f'%X_data[i][pos])
                print('\n')            
            display(Draw.MolToImage(mol))

            
            def deletion2(clikced_button : widgets.Button) -> None:
                global X_data
                X_data = X_data.drop(pos[0],axis=0)
                return X_data
            
            button = widgets.Button(description='Deletion')
            button.on_click(deletion2)
            display(button)
            self.scale_data = X_data
            return X_data            
            
        widgets.interact(descriptor,id_value=widgets.Dropdown(options=final.keys(),description='ID:'))
        
    
    def save(self):
        
        ds.save(self.scale_data,'scaling')
        
        return
def data_split_new(ext=None,test_num=0, Block=False):
    total_frame = ds.read_csv(ext)
    data_cnt = int(total_frame.shape[0])
    test_cnt = test_num

    spoint= data_cnt-test_cnt
    if Block == True :
        df_train= total_frame.iloc[:spoint, :]
        df_test= total_frame.iloc[spoint:, :]
        df_train.to_csv("%s_B.train"%ds.data_name,index=False)
        df_test.to_csv("%s_B.test"%ds.data_name,index=False)

        print ("Total : %d"%data_cnt )
        print ("Train : %s_B.train %d"%(ds.data_name,int(df_train.shape[0])) )
        print ("Test  : %s_B.test %d"%(ds.data_name,int(df_test.shape[0])))
    else :
        from sklearn.utils import shuffle
        df = shuffle(total_frame)
        df_train= df.iloc[:spoint, :]
        df_test= df.iloc[spoint:, :]
        df_train.to_csv("%s.train"%ds.data_name,index=False)
        df_test.to_csv("%s.test"%ds.data_name,index=False)

        print ("Total : %d"%data_cnt )
        print ("Train : %s.train %d"%(ds.data_name,int(df_train.shape[0])) )
        print ("Test  : %s.test %d"%(ds.data_name,int(df_test.shape[0])))
    return

                     
def data_split(ext=None,test_num=0, Random=True):
    total_frame = ds.read_csv(ext)
    data_cnt = int(total_frame.shape[0])
    test_cnt = test_num

    if Random == False :
        spoint= data_cnt-test_cnt
        df_train= total_frame.iloc[:spoint, :]
        df_test= total_frame.iloc[spoint:, :]
        df_train.to_csv("%s_B.train"%ds.data_name,index=False)
        df_test.to_csv("%s_B.test"%ds.data_name,index=False)

        print ("Total : %d"%data_cnt )
        print ("Train : %s_B.train %d"%(ds.data_name,int(df_train.shape[0])) )
        print ("Test  : %s_B.test %d"%(ds.data_name,int(df_test.shape[0])))
        return

    all_index = np.array(range(0,data_cnt))

    train_index = ""
    train_frame_scaled = ""
    test_index = ""
    test_frame_scaled = ""
    iter_num = 10

    padel_frame = total_frame.iloc[:, 3:]
    for n in range(iter_num):

        test_index = np.random.choice(all_index, test_cnt, replace=False, p=None)

        test_index = list(test_index)
        test_index.sort()
        #print(type(test_index), type(test_index[0]))
        #print(test_index)

        train_index = list(set(list(all_index)) - set(test_index))
        train_index.sort()
        #print(type(train_index), type(train_index[0]))
        #print(train_index)

        train_frame_scaled = padel_frame.iloc[train_index]
        abs_corre = abs(train_frame_scaled.corr())

        drop_index = []
        for des, row in abs_corre.iterrows():
            if sum(list(row.isna()))==abs_corre.shape[0]:
                drop_index.append(des)
                #print(n)
        print ('drop_index :',drop_index)
        print("try%4d"%(n+1))
        if n < iter_num-1 :
            if len(drop_index) > 0:
                print(drop_index)
                print("--------------------")
                continue
            else:
                #print("Before : ",train_frame_scaled.shape)
                #print("After : ",train_frame_scaled.shape)
                #test_frame = padel_frame.iloc[test_index]
                test_frame_scaled = padel_frame.iloc[test_index]
                break

        else:
            if len(drop_index) > 0:
                print(drop_index)
                #print("Before : ",train_frame_scaled.shape)
                for des in drop_index:
                    del train_frame_scaled[des]
                #print("After : ",train_frame_scaled.shape)
            else:
                aaaaa=0
                #print("Before : ",train_frame_scaled.shape)
                #print("After : ",train_frame_scaled.shape)

        #test_frame = padel_frame.iloc[test_index]
        test_frame_scaled = padel_frame.iloc[test_index]

    _id = total_frame["ID"]
    _ep = total_frame["EP"]
    _smi = total_frame["SMI"]

    #_id = total_frame["Name"]
    #_ep = total_frame["end_point"]
    #print(_ep.iloc[test_index].values)
    ################################################################################
    train_frame_scaled.insert(0, "ID", _id.iloc[train_index].values)
    train_frame_scaled.insert(1, "EP", _ep.iloc[train_index].values)
    train_frame_scaled.insert(2, "SMI", _smi.iloc[train_index].values)

    print("train : ",train_frame_scaled.shape)
    train_frame_scaled.to_csv("%s.train"%ds.data_name,index=False)
    print("%s.train  is saved."%ds.data_name)
    ################################################################################
    test_frame_scaled.insert(0, "ID", _id.iloc[test_index].values)
    test_frame_scaled.insert(1, "EP", _ep.iloc[test_index].values)
    test_frame_scaled.insert(2, "SMI", _smi.iloc[test_index].values)

    print("test : ",test_frame_scaled.shape)
    test_frame_scaled.to_csv("%s.test"%ds.data_name,index=False)
    print("%s.test   is saved."%ds.data_name)

    return

def _PaDEL_Description_Setting():
    #file = pd.read_excel('%s/PaDEL/Descriptors.xls'%sys.path[0],sheet_name='Detailed')
    file = pd.read_excel('%s/Desc/Descriptors.xls'%sys.path[0],sheet_name='Detailed')
    header = file['Descriptor Java Class'].dropna()
    header = header.reset_index(drop=True)
 
    cnt = -1
    
    for i in range(len(file['Descriptor Java Class'])):
        if pd.isna(file['Descriptor Java Class'][i]) == False:
            cnt = cnt + 1
            
        elif pd.isna(file['Descriptor Java Class'][i]) == True:
            file['Descriptor Java Class'][i] = header[cnt]
    
    possible_descript = list(file.columns.values)
    psssible_descript = possible_descript.remove('Descriptor')
    
    return file, possible_descript

def PaDEL_Description(*Descriptor):

    DataFrame, header = _PaDEL_Description_Setting()
    final = pd.DataFrame()
    for des in Descriptor:
        data = DataFrame.loc[DataFrame['Descriptor']==des]
        print ('Descriptor : %s'%des)
        for description in header:
            print (f'{description:<20} : {data[description].values}')
    return    


def NonNumricFilter (X_data):
    print('Start : ', X_data.shape)
    sel_desc = []
    for des in X_data:
        des_dtype = X_data[des].dtype
        if des_dtype == np.array([1]).dtype or des_dtype == np.array([1.0]).dtype:
            sel_desc.append(des)
    X_data = X_data[sel_desc]
    sel_desc = []
    for des in X_data:
        my_list = list(X_data[des].values)
        if sum(my_list) != 0:
            sel_desc.append(des)
    X_data = X_data[sel_desc]
    print('Filterd :', X_data.shape)
    return X_data, sel_desc

#def SDFgeneration(df, output_file):
def SDFgeneration():
    file_name = ds.load('.csv')
    output_file=file_name.split('.')[0]+'.sdf'
#    print (file_name, output_file)
    df=pd.read_csv(file_name)
    smi_df=df[['ID','EP','SMI']]
    print (smi_df)
    # Create a Pandas DataFrame to store the data
    #data = {'SMILES': smiles_list}
    #df = pd.DataFrame(data)

    # Add a column for RDKit Mol objects
    smi_df['Molecule'] = smi_df['SMI'].apply(Chem.MolFromSmiles)
#    smi_df['Molecule'] = df.loc[:,('SMI')].apply(Chem.MolFromSmiles)

    # Write the SDF file
    PandasTools.WriteSDF(smi_df, output_file, molColName='Molecule', idName='ID', properties=list(smi_df.columns))


