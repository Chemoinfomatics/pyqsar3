from pubchempy import get_compounds
from rdkit.Chem import AllChem, inchi, Draw
from pathlib import Path
from tqdm import notebook
from rdkit import Chem
from requests import get
from ipywidgets import interact
from tabulate import tabulate
import os, sys
import zipfile
import pubchempy as pcp
import time
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import re
import ipywidgets
################################################################################

def set_directory():
    homedir = sys.path[0] + '/'
    workdir = input('Enter Folder Name : ')
    try:
        os.mkdir(homedir+workdir)
        os.chdir(homedir+workdir)
        print('Working Directory : %s'%os.getcwd())
    except FileExistsError:
        print ('Directory Exists')
        os.chdir(homedir+workdir)
        print('Working Directory : %s'%os.getcwd())
    
    
def data_download(file_name='Qsar Data.zip'):
    link = input('Download Link : ')
    with open(file_name, "wb") as file:  # open in binary mode
        response = get(link)                 # get request
        file.write(response.content)         # write to file  
        
    with zipfile.ZipFile(file_name, 'r') as files:  # select zip file
        files.extractall(os.getcwd())                # setting extract path
        
def file_list(extension=None,path=None,):
    if path == None:
        path = os.getcwd()

    all_files = []
    for root, dir, file in os.walk(path):
        all_files.append(file)
    
    if extension == None:
        current = all_files[0]
        print('\033[1m'+'Path :'+path+'\033[0m')
        current.sort()
        print('%-7s %s'%('Index','File Name'))
        file_table = []
        for index, file in enumerate(current):
            print(f'{index:^7} {file}')
        print('\n')
        print('\n')

        return current
    
    else:
        ext_file = [file for file in all_files[0] if file.endswith(extension)]
        print('\033[1m'+'Path :'+path+'\033[0m')
        if len(ext_file) == 0:
            print('No %s File'%extension)
        else:
            ext_file.sort()
            print('%-7s %s'%('Index','File Name'))
            for index, file in enumerate(ext_file):
                print(f'{index:^7} {file}')

            return ext_file

def save(data,prog=None):
    pros = {'parse':'_p.csv',
            'sdf':'.sdf',
            '3D_sdf':'_3D.sdf',
            '2D_sdf':'_2D.sdf',
            'padel':'.padel',
            'filtering':'_f.csv',
            'outlier':'_o.csv',
            'scaling':'_s.csv'}
    try:
        if prog == None:
            print('%-7s %-10s %s'%('Index','Steps','Default'))
            for index, pro in enumerate(pros):
                print(f'{index:^7} {pro:<10} {pros[pro]}')
            num = input('Enter progress number : ')
            prog = list(pros.keys())[int(num)]
        
        
        file_name = input('File name to save %s as\n(default) %s\n\n-'%(prog,data_name+pros[prog]))

        
        if pros[prog] == '.padel' or pros[prog] == '.sdf':
            if file_name == '':
                file_name = data_name+pros[prog]
            else:
                file_name = file_name+pros[prog]

        else:
            seperator = pros[prog].split('.')[0]

            extension = '.'+ pros[prog].split('.')[1]
            if file_name == '':

                name_split = data_name.split(seperator)
                if name_split[-1] == data_name:
                    file_name = data_name+pros[prog]
                elif name_split[-1] == '':
                    file_name = data_name+'1'+ extension
                else:
                    file_num = int(name_split[-1]) + 1 
                    file_name = name_split[0] + seperator + str(file_num) + extension
            else:
                file_name = file_name+pros[prog]        
        if type(data) == pd.DataFrame:
            data.to_csv(file_name,index=False)
        elif type(data) == list:
            fw = open(file_name,'w')
            for line in data:
                fw.write(line)
            fw.close()

        print(f'{file_name} is saved')
        print('\n')
        return file_name
    
    except KeyError:
        print('Steps Argument Should Be : ',end='')
        for i in pros.keys():
            print('\'%s\''%i,end=' ')
    

def load(ext=None):
    current = file_list(extension=ext)
    if current == None:
        pass
        return
    else:
        if ext == None:
            load_file = input('Enter file index to load : ')
        else:
            load_file = input('Enter %s file index to load : '%ext)
    
        data_file = current[int(load_file)]
        print('\n')
        return data_file

def read_csv(ext=None,index=False):
    ext_file = load(ext)
    global data, data_name
    if ext_file == None:
        pass
    else:
        data = pd.read_csv(ext_file,index_col=index)
        data_name = ext_file.split(ext)[0]
    
        return data


def parse(ID, EP, SMI, ext=None):
    #print(globals())
    if 'data' in globals():
        df = data
    else:
        df = read_csv(ext)
    df = df.loc[:,[ID,EP,SMI]]
    df = df.rename(columns={ID:'ID',EP:'EP',SMI:'SMI'})
    df['ID'] = df['ID'].astype('str')
    display(df)   
    
    return df

def ep2class(data):
    ep_list = data['EP'].values
    
    for class_val, val in enumerate(np.unique(ep_list)):
        print('%3s = %d'%(val, class_val))
        ep_list = np.where(ep_list==val,class_val,ep_list)
    
    data['EP'] = ep_list
    display(data)
    
    return data
#'pandas version under 1.4.0'

def _get_compound_values():
    tree = ET.parse('compounds/compounds.xml')
    root = tree.getroot()
    ns = '{http://www.qsardb.org/QDB}'
    data = {}
    labels = []
    cnt = 0
    for child in root:
        values = {}
        for subchild in child:
            label = subchild.tag
            label = re.sub(ns,'',label)
            value = subchild.text
        
            if label.lower() not in labels:
                labels.append(label.lower())
            
            values[label.lower()] = value
        
        data[cnt] = values
        cnt = cnt + 1
    
    X_data = pd.DataFrame(data).T
    
    return X_data, labels

def _get_ep_values():
    data = pd.DataFrame()
    X_data, _ = _get_compound_values()
    file_list = os.listdir('properties')
    for k in file_list:
        if k.endswith('.xml') == True:
            file_list.remove(k)
    print(file_list)
    proper_name = input('Enter Property Name : ')

    for i in file_list:
        if i.startswith(proper_name) == True:
            value_data = pd.read_csv(f'{os.getcwd()}/properties/{i}/values',sep='\t')
            header = value_data.columns.values
            value_data = value_data.rename(columns={header[0]:'id',header[1]:'EP'})
            value_data['id'] = value_data['id'].astype('str')
            data = pd.concat([data,value_data],ignore_index=True)
        else:
            continue
            
    data1 = pd.DataFrame()
    file_list = os.listdir('predictions')
    for k in file_list:
        if k.endswith('.xml') == True:
            file_list.remove(k)
    print(file_list)
    use_file_list = []
    while True:
        model_name = input('Enter Model Name (Enter \'End\' if finished selecting) : ')
        if model_name == 'End':
            break
        elif model_name not in file_list:
            print('Enter Correct Model Name')
        else:
            use_file_list.append(model_name)
            
    for i in use_file_list:
        value_data = pd.read_csv(f'{os.getcwd()}/predictions/{i}/values',sep='\t')
        header = value_data.columns.values
        value_data = value_data.rename(columns={header[0]:'id',header[1]:'EP'})
        value_data['id'] = value_data['id'].astype('str')
        data1 = pd.concat([data1,value_data],ignore_index=True)
            
            
    
    data2 = pd.merge(data,data1['id'],on='id')
    X_data = pd.merge(X_data,data2,on='id')
    
    return X_data

def get_data(data_name,selection=False):
    start = time.time()
    
    _,labels = _get_compound_values()
    X_data = _get_ep_values()
    indexes = [id for id in X_data['id']]
    use_index = []
    identifiers = ['cid','inchi','inchikey','cas','smiles','sdf','name']
    my_priority = {'inchi':0,'inchikey':1,'cid':2,'smiles':3,'sdf':4,'cas':5,'name':6}
    usable = list(set(labels) & set(identifiers))
    usable = sorted(usable, key=(lambda x: my_priority[x]))
    
    print('\nUsing label :',usable)
    data_dict = {}
    for identifier in usable:
        print('\033[1m'+identifier+'\033[0m')
        data = pd.Series(data=np.array(X_data[identifier]),index=X_data['id'])
        for id_name in indexes:
            try:
                if identifier=='cas':
                    compounds = pcp.get_compounds(data[id_name],'name')
                    if len(compounds) > 1:
                        if selection == True:
                            print(compounds)
                            a = input('Enter Index to select : ')
                            compounds = [compounds[int(a)]]
                        else:
                            compounds = [compounds[0]]
                    
                    for compound in compounds:
                        smiles = compound.canonical_smiles
                    if smiles == None:
                        raise
                    data_dict[id_name] = smiles
                    use_index.append(id_name)
                    print(f'{id_name:<15}{smiles}')
                else:
                    compounds = pcp.get_compounds(data[id_name],identifier)
                    if len(compounds) > 1:
                        if selection == True:
                            print(compounds)
                            a = input('Enter Index to select :')
                            compounds = [compounds[int(a)]]
                        else:
                            compounds = [compounds[0]]
                    
                    for compound in compounds:
                        smiles = compound.canonical_smiles
                    if smiles == None:
                        raise
                    data_dict[id_name] = smiles
                    use_index.append(id_name)
                    print(f'{id_name:<15}{smiles}')
            except:
                if identifier == 'inchi' and type(data[id_name]) != float:
                    Mol = inchi.MolFromInchi(data[id_name])
                    Smiles = AllChem.MolToSmiles(Mol)
                    data_dict[id_name] = Smiles
                    use_index.append(id_name)
                    print(f'{id_name:<15}{Smiles}')
                else:
                    print(f'Error in {id_name} molecule using {identifier}')
                    pass
        indexes = list(set(indexes).difference(use_index))
    
    print('Left Molecule Index :',indexes)
    
    c = pd.DataFrame({'id':data_dict.keys(),'smiles':data_dict.values()})
    X_data = pd.merge(X_data,c,on='id')
    
    X_data = X_data.rename(columns={'id':'ID','smiles':'SMI'})
    X_data.to_csv(f'{data_name}.csv',sep=',',index=False)
    print(f'{data_name}.csv file saved')
    
    end = time.time()
    print(f"{end - start:.5f} sec")
    return X_data

def Gen3DfromSMI(smiles, maxIters=500):
    '''Convert SMILES to rdkit.Mol with 3D coordinates'''
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    else:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        confs = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters)
    
    energy = round(confs[0][1],4)

    return mol, energy
    
def get_conformer(ext=None, All=True):
    data = read_csv(ext)
    data['ID'] = data['ID'].astype(str)
    smi_list = data['SMI']
    id_list = data['ID']
    ep_list = data['EP']

    if All == True:
        confs = []
        energy = []
        cnt = 0
        na_cnt = 0
        Fail = {}

        for s in notebook.tqdm(smi_list):
            try:
                m, eng = Gen3DfromSMI(s)
                cnt = cnt+1
            except:
                if m is not None:
                    mol_index = smi_list[smi_list == s].index[0]
                    try:
                        m = Chem.MolFromSmiles(s)
                        m = Chem.AddHs(m)
                        AllChem.Compute2DCoords(m)
                        eng = np.nan
                    except:
                        Fail[mol_index] = 'Fail Optimize'

                else:
                    mol_index = smi_list[smi_list == s].index[na_cnt]
                    Fail[mol_index] = 'Empty Molecule'
                    na_cnt += 1

            confs.append(m)
            energy.append(eng)

        print("%d Molecules Optimized"%cnt)
        print("%d Molecules Fail to Optimized"%len(Fail))

        if len(Fail) != 0:
            print('%-10s %-15s %s'%('Mol Index', 'Reason', 'SMI'))
            fw = open('Fail_Optimize','w')
            for mol, reason in Fail.items():
                fw.write('%-10s %-15s %s'%(mol, reason, smi_list[mol]))
                print('%-10s %-15s %s'%(mol, reason, smi_list[mol]))
            fw.close()

        sdf_line=[]
        sdf_Lines=''
        cnt = 0
        
        try:
            os.mkdir('Image')
        except:
            print('Image Folder Already Exist')
            
        for mol, en, _id in zip(confs,energy,id_list):
            Draw.MolToFile(mol,'%s/Image/%s.png'%(os.getcwd(),_id))
            _temp = Chem.MolToMolBlock(mol)
            _temp = _temp.split('\n')
            _temp[0] = _id
            _temp = '\n'.join(_temp)

            sdf_Lines += _temp
            sdf_Lines += '>\t<SMI>\n'
            sdf_Lines += str(smi_list[cnt]) +'\n'
            sdf_Lines += '\n'
            sdf_Lines += '>\t<EP>\n'
            sdf_Lines += str(ep_list[cnt]) +'\n'
            sdf_Lines += '\n'
            sdf_Lines += '>\t<ENERGY>\n'
            sdf_Lines += str(en) +'\n'
            sdf_Lines += '\n'
            sdf_Lines += "$$$$\n"
            cnt = cnt + 1
        sdf_line.append(sdf_Lines)
        print('\n')
        save(sdf_line,prog='sdf')
        

    else:
        mol_3D = {}
        mol_2D = {}
        Fail = {}
        energy = []

        for s in notebook.tqdm(smi_list):
            try:
                mol_index = smi_list[smi_list ==s].index[0]
                m, eng = Gen3DfromSMI(s)
                mol_3D[mol_index] = m
                energy.append(eng)
            except:
                if m is not None:
                    mol_index = smi_list[smi_list == s].index[0]
                    try:
                        m = Chem.MolFromSmiles(s)
                        m = Chem.AddHs(m)
                        AllChem.Compute2DCoords(m)
                        mol_2D[mol_index] = m
                    except:
                        Fail[mol_index] = 'Fail Optimize'

                else:
                    mol_index = smi_list[smi_list == s].index[na_cnt]
                    Fail[mol_index] = 'Empty Molecule'
                    na_cnt += 1

        print("%d Molecules Optimized"%len(mol_3D))
        print("%d Molecules have 2D Coords"%len(mol_2D))
        print("%d Molecules Fail to Optimized"%len(Fail))

        if len(Fail) !=0:
            print('%-10s %-15s %s'%('Mol Index', 'Reason', 'SMI'))
            fw = open('Fail_Optimize','w')
            for mol, reason in Fail.items():
                fw.write('%-10s %-15s %s'%(mol, reason, smi_list[mol]))
                print('%-10s %-15s %s'%(mol, reason, smi_list[mol]))
            
        sdf_line=[]
        sdf_Lines=''
        try:
            os.mkdir('Image')
        except:
            print('Image Folder Already Exist')
            
        for mol, en, index in zip(mol_3D.values(),energy,mol_3D.keys()):
            Draw.MolToFile(mol,'%s/Image/%s.png'%(os.getcwd(),id_list[index]))
            _temp = Chem.MolToMolBlock(mol)
            _temp = _temp.split('\n')
            _temp[0] = id_list[index]
            _temp = '\n'.join(_temp)

            sdf_Lines += _temp
            sdf_Lines += '>\t<SMI>\n'
            sdf_Lines += str(smi_list[index]) +'\n'
            sdf_Lines += '\n'
            sdf_Lines += '>\t<EP>\n'
            sdf_Lines += str(ep_list[index]) +'\n'
            sdf_Lines += '\n'
            sdf_Lines += '>\t<ENERGY>\n'
            sdf_Lines += str(en) +'\n'
            sdf_Lines += '\n'
            sdf_Lines += "$$$$\n"

        sdf_line.append(sdf_Lines)
        print('\n')
        save(sdf_line,'3D_sdf')
################################################################################
        sdf_line = []
        sdf_Lines = ''
        cnt = 0

        for index, mol in mol_2D.items():
            Draw.MolToFile(mol,'%s/Image/%s.png'%(os.getcwd(),id_list[index]))
            _temp = Chem.MolToMolBlock(mol)
            _temp = _temp.split('\n')
            _temp[0] = id_list[index]
            _temp = '\n'.join(_temp)

            sdf_Lines += _temp
            sdf_Lines += '>\t<SMI>\n'
            sdf_Lines += str(smi_list[index]) + '\n'
            sdf_Lines += '\n'
            sdf_Lines += '>\t<EP>\n'
            sdf_Lines += str(ep_list[index]) + '\n'
            sdf_Lines += '\n'
            sdf_Lines += "$$$$\n"
  
        sdf_line.append(sdf_Lines)
        
        save(sdf_line,'2D_sdf')
        
        
def show_sdf(*showlist):
    file_name = load('.sdf')
    frame = [x for x in Chem.SDMolSupplier(file_name, removeHs=False) if x is not None]
    Name = [x.GetProp("_Name") for x in frame]
    if len(showlist) == 1:
        prop = [x.GetPropsAsDict()[showlist[0]] for x in frame]
        legend = ['%s : %s'%(str(x),str(y)) for x,y in zip(Name,prop)]
    elif len(showlist) == 2:
        prop1 = [x.GetPropsAsDict()[showlist[0]] for x in frame]
        prop2 = [x.GetPropsAsDict()[showlist[1]] for x in frame]
        legend = ['%s : %s : %s'%(str(x),str(y),str(z)) for x, y, z in zip(Name, prop1, prop2)]
    else:
        prop1 = [x.GetPropsAsDict()[showlist[0]] for x in frame]
        prop2 = [x.GetPropsAsDict()[showlist[1]] for x in frame]
        prop3 = [x.GetPropsAsDict()[showlist[2]] for x in frame]
        legend = ['%s : %s : %s : %s'%(str(x),str(y),str(z),str(w)) for x,y,z,w in zip(Name,prop1,prop2,prop3)]
    img = Draw.MolsToGridImage(frame, molsPerRow=3,
                                       subImgSize=(300,300),
                                       legends=legend,
                                       maxMols=9999)
    display(img)
        
def run_PaDEL(n_cpu=4, fingerprint=False, desc_2d=True, desc_3d=False):
    
    command = "java -jar -Djava.awt.headless=true %s/PaDEL/PaDEL-Descriptor.jar "%sys.path[0]
    command += "-threads %d "%n_cpu

    if desc_2d==True:
        command += "-2d "
    if desc_3d==True:
        command += "-3d "
    if fingerprint==True:
        command += "-fingerprints "
    
    file_name = load('.sdf')
    frame = [x for x in Chem.SDMolSupplier(file_name, removeHs=False) if x is not None]
    id_list = [x.GetProp("_Name") for x in frame]
    ep_list = [x.GetPropsAsDict()['EP'] for x in frame]
    smi_list = [x.GetPropsAsDict()['SMI'] for x in frame]
    global data_name
    data_name = file_name.split('.sdf')[0]
    command += "-dir %s -file PaDEL.csv -retainorder"%file_name
    print("Runnig PaDEL .....")
    os.system(command)
    print("Finished !")

    _temp_padel = pd.read_csv("PaDEL.csv", sep=",")
    del _temp_padel["Name"]
    
    
    _temp_padel.insert(0, "ID", id_list)
    _temp_padel.insert(1, "EP", ep_list)
    _temp_padel.insert(2, "SMI", smi_list)
    #_temp_padel["pKa"] = ep_list
    
    #_temp_padel.to_csv("%s.padel"%qstart,index=False)
    
    os.system("rm PaDEL.csv")
    
    
    return _temp_padel

def check_PaDEL(data, over_value):
    header = data.columns.values[3:]
    id_list = data['ID'].values
    for i in header:
        val = data[i].values
        pos = np.where((val > over_value) & (val != np.inf))[0]
        if len(pos) != 0:
            print('In %s medain value : %.2f'%(i,np.quantile(val,0.75)))
            
            for j in pos:
                print('%s molecule has %.4f, so convert to %.2f'%(id_list[j],val[j],over_value))
            print('\n')
            val[pos] = over_value
            data[i] = val
            
    return data

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
    print(header)
    final = pd.DataFrame()
    for des in Descriptor:
        data = DataFrame.loc[DataFrame['Descriptor']==des]
        print ('Descriptor : %s'%des)
        for description in header:
            print ('%s : %s'%(description, data[description].values))
        print ('\n')    
        final = final.append(data)

    final = final.reset_index(drop=True)
    final.to_csv('Description.csv',index=False)
    print('Descriptors\' Description.csv is saved')
    return final


#def SDFgeneration():
#    file_name = load('.csv')
#    output_file=file_name.split('.')[0]+'.csv'
#    df=pd.read_csv(file_name)
#    smi_df=df[['ID','EP','SMI']]
#    print (smi_df)
#    # Create a Pandas DataFrame to store the data
#    #data = {'SMILES': smiles_list}
#    #df = pd.DataFrame(data)
#
#    # Add a column for RDKit Mol objects
#    smi_df['Molecule'] = smi_df['SMI'].apply(Chem.MolFromSmiles)
##    smi_df['Molecule'] = df.loc[:,('SMI')].apply(Chem.MolFromSmiles)
#
#    # Write the SDF file
#    PandasTools.WriteSDF(smi_df, output_file, molColName='Molecule', idName='ID', properties=list(smi_df.columns))


