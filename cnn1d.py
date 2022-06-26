"""CNN1D class from the original paper"""
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection  import train_test_split


class CNN1D(nn.Module):
    """CNN1D class from the original paper implemented in Python with PyTorch"""
    def __init__(self):
        super(CNN1D, self).__init__()
        self.enc_0 = nn.Sequential(
            nn.Conv1d(33, 300, 6, 1),
            nn.ReLU(),
            nn.Conv1d(300, 300, 6, 1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(337, 600),
            nn.ReLU(),
            nn.Linear(600, 1),
            nn.Identity(),
        )

    def forward(self, x, col):
        x = self.enc_0(x)
        # dimensions of all training tensors are 33*256 so using 
        # sum instead of average pooling for each leayer makes
        # much more sense from performance standpoint without any 
        # meaningful sacrifices in overall architecture
        x = x.sum(dim=2,keepdim=True).squeeze()
        x = torch.cat([x, col], dim=1)
        return self.fc(x)

class CNN1D_Dataset(Dataset):
    """Dataset class for CNN1D neural network.
    It uses an OHE-encoded SMILES representation
    of a molecule along with encoding types of columns."""
    def __init__(self, dataframe,mask=None):
        def _split_formula(x: str):
            if "i" in x or "l" in x or "r" in x:
                tmp = list(x)
                NML = set(("Si", "Cl", "Br"))
                i = 0
                while i < len(tmp) - 1:
                    t_str = tmp[i] + tmp[i + 1]
                    if t_str in NML:
                        tmp[i] = t_str
                        tmp.pop(i + 1)
                    i += 1
                return np.array(tmp)
            else:
                return np.array(list(x))
        
        def _make_index(str_formulas, unique_formulas):
            d0=dict(zip(unique_formulas,range(len(unique_formulas))))
            d=np.zeros((len(str_formulas)))
            for i,val in enumerate(str_formulas):
                idx=d0[val]
                d[i]=idx
            return d.astype(int)
        
        def _encode_smiles(line):
            arr = np.zeros((len(self.SYMS), self.max_len))
            for i, sym in enumerate(line):
                arr[self.D_SYM.get(sym), i] = 1
            return arr

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
        self.SYMS = [
            '#', '(', ')', '+', '-', '1', '2', '3', '4', '5', '6', '7', '8',
            '9', '=', 'Br', 'B', 'Cl', 'C', 'F', 'H', 'I', 'N', 'O', 'P', 'Si',
            'S', '[', ']', 'c', 'n', 'o', 's'
        ]
        self.D_SYM = dict(zip(self.SYMS, range(len(self.SYMS))))
        unique_formulas=pd.unique(df["Formula"]).tolist()
        str_formulas=df["Formula"].values
        self.index=_make_index(str_formulas,unique_formulas)
        self.max_len = 256
        self.ris = df["RI"].values/1000
        self.cols = df["ColType"].values
        split_formulas=list(map(_split_formula,tqdm(unique_formulas)))
        self.enc_formulas=list(map(_encode_smiles,tqdm(split_formulas)))
        self.enc_columns=np.vstack(list(map(_encode_column,range(37))))
        

    def __getitem__(self, index):
        smiles_encoded = torch.FloatTensor(self.enc_formulas[self.index[index]])
        col_encoded = torch.FloatTensor(self.enc_columns[self.cols[index]])
        return (smiles_encoded, col_encoded, self.ris[index])

    def __len__(self):
        return len(self.ris)
    
def get_train_test_dataloaders(df_name,batch_size):
    """Prepares train and test dataloaders by splitting input data
    into respective CNN1D datasets.

    Parameters
    ----------
    df_name : str
        name of dataframe with smiles, column type and retention index
    batch_size : int
        Batch size for the two dataloaders

    Returns
    -------
    tuple (DataLoader, DataLoader)
        tuple containing train and test DataLoaders respectively
    """
    df = pd.read_csv(df_name)
    df.columns = ["Formula", "RI", "ColType"]
    _,tst_unique=train_test_split(pd.unique(df["Formula"]),test_size=0.2,random_state=42)
    trn_mask,tst_mask=[],[]
    tst_unique=set(tst_unique)
    for i,val in enumerate(df["Formula"].values):
        if val in tst_unique:
            tst_mask.append(i)
        else:
            trn_mask.append(i)
    trn_ds=CNN1D_Dataset(df,trn_mask)
    tst_ds=CNN1D_Dataset(df,tst_mask)
    return (DataLoader(trn_ds, batch_size,pin_memory=True,num_workers=4),
            DataLoader(tst_ds, batch_size,pin_memory=True,num_workers=4))

def get_val_dataloader(df_name, batch_size):
    """Prepares validation dataloader.

    Parameters
    ----------
    df_name : str
        name of dataframe with smiles, column type and retention index
    
    Returns
    -------
    DataLoader
        validation DataLoader
    """
    df = pd.read_csv(df_name)
    df.columns = ["Formula", "RI", "ColType"]
    ds = CNN1D_Dataset(df)
    return DataLoader(ds, batch_size,pin_memory=True,num_workers=4)