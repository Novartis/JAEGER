"""
Copyright 2021 Novartis Institutes for BioMedical Research Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import pandas as pd
import janitor
import janitor.chemistry
# nn stuff
import torch
# chem stuff
import rdkit.Chem as Chem
from rdkit import DataStructs, RDLogger
from rdkit.Chem import AllChem
from torch.utils import data

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)



def normalize_morgans(morgans):
    anscombe = np.sqrt(morgans + (3.0 / 8.0)) * 2
    max_count = 30
    max_count = np.sqrt(max_count + (3.0 / 8.0)) * 2
    normalized = anscombe / max_count
    return normalized


class MorgansDataset(data.Dataset):
    def __init__(self, morgans: pd.DataFrame, targets: pd.DataFrame):
        """
        Create a Morgans dataset for use with PyTorch.

        targets and morgans must be pandas DataFrames respectively
        and they must be indexed by structure IDs.

        Assumes that the targets, and morgans are indexed identically.
        """
        # assert len(targets) == len(
        #    morgans
        # ), "morgans and targets must be of the same length."
        # assert ((targets.index == morgans.index).all());
        self.targets = targets
        self.morgans = morgans
        self.list_IDs = morgans.index
        # for real testing  the dataset is constructed without a target???  Right ...

    def __len__(self):
        """Return the total number of samples"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates one sample of data
        The index passed by the torch generator is an integer
        which we have to remap to our own internal index ...
        """
        # Load data and get target
        structure_id = self.list_IDs[index]
        normalized = normalize_morgans(self.morgans.loc[structure_id].values)
        X = torch.from_numpy(normalized).float()
        if self.targets is not None:
            y = torch.from_numpy(np.array(self.targets.loc[structure_id])).float()
        else:
            y = torch.tensor(-1).float()  # return a dummy variable

        return X, y



def preprocess_redux(assay_data,
                     binary_fp=False,
                     filter_mols=True,
                     n_atoms_filter=50,
                     convert_to_pac50=False,
                     drop_qualified = True,
                     convert_fps_to_numpy = False):
    toxdata = assay_data
    if drop_qualified:
        toxdata = toxdata.dropnotnull("qualifier")

    toxdata = toxdata.transform_column("val", np.log10)
    toxdata = toxdata.remove_columns(["qualifier"])
    toxdata = toxdata.replace([np.inf, -np.inf], np.nan).dropna(subset=["val"])

    if convert_to_pac50:
        toxdata["val"] = (toxdata["val"] - 6) * -1

    n_dropped = len(assay_data) - len(toxdata)
    print(n_dropped)
    morgans = {}
    mols = {}
    for idx in toxdata.index:
        smiles = toxdata.smiles.loc[idx]
        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            print(smiles)
            print("[ERROR] Could not create mol")
            continue
        
        try:
            if binary_fp:
                fp_array = AllChem.GetMorganFingerprint(mol, 2)
                # TODO SOMETIMES THIS IS NEEDED
                if convert_fps_to_numpy:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
                    fp_array = np.zeros((0,), dtype=np.int8)
                    DataStructs.ConvertToNumpyArray(fp, fp_array)
            else:
                fp = AllChem.GetHashedMorganFingerprint(
                    mol, radius=3, nBits=2048, useChirality=True
                )
                #fp = AllChem.GetHashedMorganFingerprint(
                #    mol, radius=2, nBits=1024, useChirality=True
                #)
                fp_array = np.zeros((0,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(fp, fp_array)
            morgans[idx] = fp_array
            mols[idx] = mol  # don't add the molecule if the FP cannot be computed
        except:
            print("[ERROR] Could not create fingerprint")
            continue

    morgans_df = pd.DataFrame.from_dict(morgans, orient="index")
    mols_df = pd.DataFrame.from_dict(mols, orient="index", columns=["mols"])

    print(len(morgans_df))
    print(len(mols_df))
    print(len(toxdata))

    toxdata = mols_df.join(toxdata)
    targets = toxdata.val
    toxdata_original = toxdata

    if filter_mols == True:
        n_entries = len(toxdata)
        mol_n_atoms = np.zeros((n_entries, 1))
        for i in range(n_entries):
            mol = toxdata.iloc[i].mols
            mol_n_atoms[i] = mol.GetNumAtoms()

        toxdata["n_atoms"] = mol_n_atoms
        idx = toxdata.n_atoms < n_atoms_filter
        toxdata = toxdata[idx]  # this drops about 200 molecules
        del toxdata["n_atoms"]

    print("preprocess_redux() :: toxdata len " + str(len(toxdata)))    
    morgans_df = morgans_df.loc[toxdata.index] # for consistency 
    targets = targets.loc[toxdata.index]
    return morgans_df, targets, toxdata
    

def modelling_data_from_csv(
    csv_file,
    binary_fp=False,
    filter_mols=True,
    n_atoms_filter=50,
    convert_to_pac50=False,
        drop_qualified = True,
        convert_fps_to_numpy = False,
):
    df = pd.read_csv(csv_file, engine="c", index_col=0)
    print(len(df))
    # --- select relevant data and rename columns
    assay_data = df.iloc[:, 0:3]
    assay_data.columns = ["smiles", "qualifier", "val"]
    print("------------S-------------")
    print(assay_data.head())
    print("------------E------------")
    assay_data.index.name = "structure_id"  
    print("modelling_data_from_csv() :: filter mols " + str(filter_mols))
    return preprocess_redux(assay_data,
                            binary_fp,
                            filter_mols,
                            n_atoms_filter,
                            convert_to_pac50,
                            drop_qualified,
                            convert_fps_to_numpy = convert_fps_to_numpy)
