#-*- coding: utf-8 -*-
import os

import pandas as pd
import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series

from sklearn.metrics import mean_squared_error , r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDConfig
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem.Subshape import SubshapeBuilder,SubshapeAligner,SubshapeObjects
from rdkit.Chem import rdFMCS

from ipywidgets import interact
import py3Dmol
import ipywidgets

from bokeh.plotting import figure, output_file, show, ColumnDataSource, output_notebook
from bokeh.models import HoverTool, BoxSelectTool

from pyqsar import data_setting as ds

class DrawMols():
    """
    Tool of drawing the molecule

    Parameters
    ----------
    path : str , file path of sdf file

    Sub functions
    -------
    show(self,index=[])
    save_img (self,index=[])
    common_substr (self,index=[])
    show_substr(self,substr_info)
    """
    def __init__(self, ID=[]):
        self.path = ds.load('.sdf')
        #print (self.path)
        self.mols = [x for x in Chem.SDMolSupplier(self.path,removeHs=False) if x is not None]
        self.index = ID
        self.id_list = [x.GetProp("_Name") for x in self.mols]
        self.ep_list = [x.GetPropsAsDict()['EP'] for x in self.mols]
        self.smi_list = [x.GetPropsAsDict()['SMI'] for x in self.mols]       

    def show(self):
        """
        Draw the molecule of the user-specified index

        Parameters
        ----------
        index : list, specified index that user want to draw

        Returns
        -------
        Molecule images

        """
        index_list = self.index
        id_list = self.id_list
        ep_list = self.ep_list
        idx = []
        for i in index_list:
            idx.append(self.id_list.index(str(i)))
        
        pickimg = []   
        cdk2mols = self.mols
        for m in cdk2mols: tmp=AllChem.Compute2DCoords(m)
        for i in idx :
            img = cdk2mols[i]
            pickimg.append(img)
        
        return Draw.MolsToGridImage(pickimg[:],molsPerRow=4,legends=['%s : %s'%(str(id_list[x]),str(ep_list[x])) for x in idx],maxMols=len(idx))

    def save_img (self) :
        """
        Save the molecule of the user-specified index as image files

        Parameters
        ----------
        index : list, specified index that user want to save

        Returns
        -------
        list, saved file names

        """
        idx = self.idx
        imglist = []
        mols = self.mols
        for m in mols: tmp=AllChem.Compute2DCoords(m)
        for i in index :
            moll = mols[i]
            Draw.MolToFile(moll,'%d.png'%(i))
            #img =  Draw.MolsToGridImage(moll)
            #img.save('%s%d.png'%(save_path,i))
            imglist.append('%d.png'%(i))
        return imglist

    def common_substr (self) :
        """
        Finding common structure among specified index molecules

        Parameters
        ----------
        index : list, specified molecule's index

        Returns
        -------
        Smart string of common structure

        """
        index_list = self.index
        id_list = self.id_list
        ep_list = self.ep_list
        
        idx = []
        for i in index_list:
            idx.append(self.id_list.index(str(i)))
            
        pickmol = []
        mols = self.mols
        for i in idx :
            mol = mols[i]
            pickmol.append(mol)
        res=rdFMCS.FindMCS(pickmol)
        mol = Chem.MolFromSmarts(res.smartsString)
        return res.smartsString

    def show_substr(self,substr_info):
        """
        Draw the common structure using smart string

        Parameters
        ----------
        substr_info : str, return value of common_substr()

        Returns
        -------
        Image of common structure
        """
        return Chem.MolFromSmarts(substr_info)
    
    def show_3D(self):
        mols = self.mols
        index_list = self.index
        id_list = self.id_list
        ep_list = self.ep_list
        
        idx = []
        for i in index_list:
            idx.append(self.id_list.index(str(i)))

        pick_mol = []
    
        def style_selector(ID,s):
            idx = id_list.index(str(ID))
            print( 'EP :',ep_list[idx])
        
            return MolTo3DView(mols[idx], style=s).show()
               
        interact(style_selector, 
                 ID=ipywidgets.Dropdown(options=self.index, description='ID'),
                 s=ipywidgets.Dropdown(
                     options=['line','stick','sphere'],
                     value='line',
                     description='Style:'))


def MolTo3DView(mol, size=(700, 300), style="stick", surface=False, opacity=0.5):
    """Draw molecule in 3D

    Args:
    ----
        mol: rdMol, molecule to show
        size: tuple(int, int), canvas size
        style: str, type of drawing molecule
               style can be 'line', 'stick', 'sphere', 'carton'
        surface, bool, display SAS
        opacity, float, opacity of surface, range 0.0-1.0
    Return:
    ----
        viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
    """
    assert style in ('line', 'stick', 'sphere', 'carton')
    mblock = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=size[0], height=size[1])
    viewer.addModel(mblock, 'mol')
    viewer.setStyle({style:{}})
    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
    viewer.zoomTo()
    return viewer



def mol_plot(X_data, y_data, feature_set, imglist):
    """
    Draw the interactive prediction graphe with molecule images

    Parameters
    ----------
    X_data : pandas DataFrame , shape = (n_samples, n_features)
    y_data : pandas DataFrame , shape = (n_samples,)
    feature_set : list, set of features that make up model
    imglist : return value of save_mol()

    Returns
    -------
    Interactive prediction graphe
    """
    output_notebook()
    x = X_data.loc[:,feature_set].values
    Ay = y_data.values
    ipred_plotY = np.zeros_like(Ay)
    ig_mlrr = LinearRegression()
    ig_mlrr.fit(x,Ay)
    Py = ig_mlrr.predict(x)
    ppy = []
    aay = []
    for i in Py :
        ppy.append(i[0])
    for j in Ay :
        aay.append(j[0])

    source = ColumnDataSource(
    data=dict(x=aay,y=ppy, imgs = imglist)
    )
    hover = HoverTool(
        tooltips="""
        <div>
            <div>
                <img
                    src="@imgs" height="100" alt="@imgs" width="180"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                    ></img>
                    </div>
                    <div>
                    <span style="font-size: 17px; font-weight: bold;">@desc</span>
                    <span style="font-size: 15px; color: #966;">[$index]</span>
                    </div>
                    <div>
                    <span style="font-size: 15px;">Location</span>
                    <span style="font-size: 10px; color: #696;">($x, $y)</span>
                    </div>
                    </div>
                    """
                    )

    p = figure(plot_width=600, plot_height=600, tools=[hover],
    title="Predict & Actual")
    p.circle('x', 'y', size=20, source=source, color="green", alpha=0.5 )
    show(p)
