"""The whole prediction model and its respective Dataset and Dataloaders."""
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm

from cnn1d import CNN1D
from cnn2d import CNN2D
from mlp import MLP


class MultiModal(nn.Module):
    """The whole prediction model that contains three blocks: CNN1D, CNN2D, MLP
    and uses gradient boosting prediction results."""

    def __init__(self, cnn1d=CNN1D(), cnn2d=CNN2D(), mlp=MLP()):
        super(MultiModal, self).__init__()
        self.cnn1d = cnn1d
        self.cnn2d = cnn2d
        self.mlp = mlp
        self.fc = nn.Sequential(
            nn.Linear(4, 1),
            nn.ReLU(),
        )

    def forward(self, smiles, twod, md, maccs, fp, col, boost_pred):
        with torch.no_grad():
            # NN blocks are pre-trained, so they can be freezed
            x1 = self.cnn1d(smiles, col)
            x2 = self.cnn2d(twod, col)
            x3 = self.mlp(md, fp, col, maccs)
            # Neither CatBoost nor XGBoost can be called within nn.Module
            # call, so they should be incorporated in training loop:
            # def train():
            #   ...
            #   for smiles, twod, md, maccs, fp, col in train_dl:
            #       ...
            #       boost_data = torch.cat([md, maccs, col], dim=1).cpu().numpy()
            #       boost_pred = CatBoost.predict(boost_data)
            #       ...
            x4 = boost_pred
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.fc(x)
    # example usage:

    # device = torch.device("cuda")
    # cnn1d = CNN1D()
    # cnn1d.load_state_dict(torch.load("./cnn1d.pth"))
    # cnn2d = CNN2D()
    # cnn2d.load_state_dict(torch.load("./cnn2d.pth"))
    # mlp = MLP()
    # mlp.load_state_dict(torch.load("./mlp.pth"))
    # boost = CatBoostRegressor()
    # boost.load_model("./catboost.model")
    # mm = MultiModal(cnn1d, cnn2d, mlp)


class MM_Dataset(Dataset):
    """Dataset class for multimodal model.
    It uses parts of all other datasets."""

    def __init__(self, dataframe, en_2d, md, maccs, fp, mask=None):
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

        def _make_index(unique_formulas):
            d0 = dict(zip(unique_formulas, range(len(unique_formulas))))
            d = np.zeros((len(self.formulas)))
            for i, val in enumerate(self.formulas):
                idx = d0[val]
                d[i] = idx
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
            df = dataframe.iloc[mask].copy()
        else:
            df = dataframe

        self.SYMS = [
            '#', '(', ')', '+', '-', '1', '2', '3', '4', '5', '6', '7', '8',
            '9', '=', 'Br', 'B', 'Cl', 'C', 'F', 'H', 'I', 'N', 'O', 'P', 'Si',
            'S', '[', ']', 'c', 'n', 'o', 's'
        ]
        self.D_SYM = dict(zip(self.SYMS, range(len(self.SYMS))))
        self.formulas = df["Formula"].values
        unique_formulas = pd.unique(self.formulas)
        self.index = _make_index(unique_formulas)
        self.max_len = 256
        cols = df["ColType"].values
        split_formulas = list(map(_split_formula, tqdm(unique_formulas)))

        self.enc_formulas = list(map(_encode_smiles, tqdm(split_formulas)))
        self.en_2d = en_2d
        self.md = md
        self.maccs = maccs
        self.fp = fp
        self.en_cols = np.vstack(list(map(_encode_column, cols)))
        self.ris = df["RI"].values/1000

    def __getitem__(self, index):
        smiles = torch.FloatTensor(self.enc_formulas[self.index[index]])
        twod = torch.FloatTensor(self.en_2d[self.index[index]])
        descriptors = torch.FloatTensor(self.md[self.index[index]])
        maccs = torch.FloatTensor(self.maccs[self.index[index]])
        fingerprints = torch.FloatTensor(self.fp[self.index[index]])
        col_encoded = torch.FloatTensor(self.en_cols[index])
        return (smiles, twod, descriptors, maccs, fingerprints, col_encoded, self.ris[index])

    def __len__(self):
        return len(self.ris)


def get_train_test_dataloaders(df_name: str, en_2d: np.array, md: np.array, maccs: np.array, fp: np.array, batch_size: int):
    """Prepares train and test dataloaders by splitting input data
    into respective MM datasets.

    Parameters
    ----------
    df_name : str
        name of dataframe with smiles, column type and retention index
    en_2d : np.array
        Array with pre-encoded 2D images of molecules: two coordinate axes
        and channels as a third  OHE-encoded one

    md: np.array
        Array of pre-generated molecular descriptros with length 1604
    fp: np.array
        Array of pre-generated molecular fingerprints (ECFP4 with counts) with length 1024
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
    _, tst_unique = train_test_split(
        pd.unique(df["Formula"]), test_size=0.2, random_state=42)
    trn_mask, tst_mask = [], []
    tst_unique = set(tst_unique)
    for i, val in enumerate(df["Formula"].values):
        if val in tst_unique:
            tst_mask.append(i)
        else:
            trn_mask.append(i)
    trn_ds = MM_Dataset(df, en_2d, md, maccs, fp, trn_mask)
    tst_ds = MM_Dataset(df, en_2d, md, maccs, fp, tst_mask)
    return (DataLoader(trn_ds, batch_size, pin_memory=True, shuffle=True, num_workers=4),
            DataLoader(tst_ds, batch_size, pin_memory=True, shuffle=True, num_workers=4))


def get_val_dataloader(df_name: str, en_2d: np.array, md: np.array, maccs: np.array, fp: np.array, batch_size: int) -> DataLoader:
    """Prepares validation dataloader.

    Parameters
    ----------
    df_name : str
        name of dataframe with smiles, column type and retention index
    en_2d : np.array
        Array with pre-encoded 2D images of molecules: two coordinate axes
        and channels as a third  OHE-encoded one

    md: np.array
        Array of pre-generated molecular descriptros with length 1604
    fp: np.array
        Array of pre-generated molecular fingerprints (ECFP4 with counts) with length 1024
    maccs: np.array
        Array of pre-generated MACCS keys with length of 167
    batch_size : int
        Batch size for the two dataloaders

    Returns
    -------
    DataLoader
        validation DataLoader
    """
    df = pd.read_csv(df_name)
    df.columns = ["Formula", "RI", "ColType"]
    val_ds = MM_Dataset(df, en_2d, md, maccs, fp)
    return DataLoader(val_ds, batch_size, pin_memory=True, num_workers=4)
