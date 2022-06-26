"""MLP neural network block and DataLoader"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection  import train_test_split

class MLP(nn.Module):
    """MLP class from the original paper implemented in Python with PyTorch"""
    def __init__(self):
        super(MLP, self).__init__()
        self.fc_md_0=nn.Sequential(
            nn.Linear(1604+167+37,300),
            nn.Tanh(),
            nn.Linear(300,300),
            nn.ReLU()
        )
        self.fc_fp_0=nn.Sequential(
            nn.Linear(1024,1200),
            nn.ReLU(),
        )
        self.fc_res_1=nn.Sequential(
            nn.Linear(1200,1200),
            nn.Dropout(0.05),
            nn.ReLU(),
            nn.Linear(1200,1200),
            nn.Dropout(0.05),
            nn.ReLU()
        )
        self.fc_res_2=nn.Sequential(
            nn.Linear(1200,1200),
            nn.Dropout(0.05),
            nn.ReLU(),
            nn.Linear(1200,1200),
            nn.Dropout(0.05),
            nn.ReLU()
        )
        self.fc_3=nn.Sequential(
            nn.Linear(1500,600),
            nn.ReLU(),
            nn.Linear(600,1),
            nn.Identity()
        )
    
    def forward(self,md,fp,col,maccs):
        x=self.fc_md_0(torch.cat([md,col,maccs],dim=1))
        y=self.fc_fp_0(fp)
        z=self.fc_res_1(y)+y
        y=self.fc_res_2(z)+z
        x=torch.cat([x,y],dim=1)
        return self.fc_3(x)

class MLP_Dataset(Dataset):
    """Dataset class for MLP neural network.
    It uses pre-generated molecular descriptors, molecular fingerprints, 
    MACCS keys of a molecule along with encoding types of columns."""
    def __init__(self, dataframe,descriptors,fingerprints,unique_formulas,maccs,mask=None): 
        def _make_index(formulas, unique_formulas):
            d0=dict(zip(unique_formulas,range(len(unique_formulas))))
            d={}
            for i,val in enumerate(formulas):
                idx=d0.get(val,None)
                if idx is None:
                    print(f"Formula error: {val}")
                d[i]=idx
            return d
        
        def _encode_column(col_type):
            res = np.zeros((37))
            res[col_type] = 1
            if col_type > 14:
                res[-1] = 1
            return res

        if mask is not None:
            df=dataframe.iloc[mask].copy()
        else:
            df=dataframe
        self.maccs=maccs
        self.md = descriptors
        self.fp=fingerprints
        self.index=_make_index(df["Formula"].values, unique_formulas)
        self.ris = df["RI"].values/1000
        cols = df["ColType"].values
        self.en_cols=np.vstack(list(map(_encode_column,cols)))

    def __getitem__(self, index):
        descriptors = torch.FloatTensor(self.md[self.index[index]])
        fingerprints= torch.FloatTensor(self.fp[self.index[index]])
        maccs=torch.FloatTensor(self.maccs[self.index[index]])
        col_encoded = torch.FloatTensor(self.en_cols[index])
        return (descriptors, fingerprints,col_encoded, maccs,self.ris[index])

    def __len__(self):
        return len(self.ris)
    
def get_train_test_dataloaders(df_name:str,descriptors:np.array,fingerprints:np.array,formulas:np.array,maccs:np.array,batch_size:int)->tuple:
    """Prepares train and test dataloaders by splitting input data
    into respective MLP datasets.

    Parameters
    ----------
    df_name : str
        name of dataframe with smiles, column type and retention index
    descriptors: np.array
        Array of pre-generated molecular descriptros with length 1604
    fingerprints: np.array
        Array of pre-generated molecular fingerprints (ECFP4 with counts) with length 1024
    formulas: np.array or list
        List of unique formulas generated in preprocessing pipeline
    maccs: np.array
        Array of pre-generated MACCS keys with length of 167
    batch_size : int
        Batch size for the two dataloaders

    Returns
    -------
    tuple (DataLoader, DataLoader)
        tuple containing train and test DataLoaders respectively
    """
    df = pd.read_csv(df_name)
    df.columns = ["Formula", "RI", "ColType"]
    _,tst_unique=train_test_split(formulas.tolist(),test_size=0.2,random_state=42)
    trn_mask,tst_mask=[],[]
    tst_unique=set(tst_unique)
    for i,val in enumerate(df["Formula"].values):
        if val in tst_unique:
            tst_mask.append(i)
        else:
            trn_mask.append(i)
    trn_ds=MLP_Dataset(df,descriptors,fingerprints,formulas,maccs,trn_mask)
    tst_ds=MLP_Dataset(df,descriptors,fingerprints,formulas,maccs,tst_mask)
    return (DataLoader(trn_ds, batch_size,pin_memory=True,num_workers=4),
            DataLoader(tst_ds, batch_size,pin_memory=True,num_workers=4))

def get_val_dataloader(df_name:str,descriptors:np.array,fingerprints:np.array,formulas:np.array,maccs:np.array,batch_size:int)->DataLoader:
    """Prepares validation dataloader for MLP dataset.

    Parameters
    ----------
    df_name : str
        name of dataframe with smiles, column type and retention index
    descriptors: np.array
        Array of pre-generated molecular descriptros with length 1604
    fingerprints: np.array
        Array of pre-generated molecular fingerprints (ECFP4 with counts) with length 1024
    formulas: np.array or list
        List of unique formulas generated in preprocessing pipeline
    maccs: np.array
        Array of pre-generated MACCS keys with length of 167
    batch_size : int
        Batch size for the two dataloaders

    Returns
    -------
    DataLoader
        validation DataLoaders
    """
    df = pd.read_csv(df_name)
    df.columns = ["Formula", "RI", "ColType"]
    ds = MLP_Dataset(df,descriptors,fingerprints,formulas,maccs)
    return DataLoader(ds, batch_size,pin_memory=True,num_workers=4)