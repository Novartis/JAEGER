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

NAME="JAEGER"
TITLE="**JAEGER**: JT-VAE Generative Modeling"
JAEGER_HOME="/path/to/models"
BASE_DIR=JAEGER_HOME+"/assays" 
TRAINING_DIR=JAEGER_HOME+"/training_data"
AVAIL_MODELS=JAEGER_HOME+"/jaeger_avail_models.csv"

### JAEGER
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os


# --- RDKIT imports
import rdkit.Chem as Chem
import rdkit

# --- TORCH imports
import torch


# --- JTVAE imports
from jtnn import *
from jtnn.jtprop_vae import JTPropVAE

# --- TOXSQUAD imports
from toxsquad.data import modelling_data_from_csv

import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# --- JAEGER imports
from jaeger.utils.jtvae_utils import compute_properties
from jaeger.utils.jtvae_utils import get_vocab
from jaeger.utils.jtvae_utils import get_neighbors_along_directions_tree_then_graph
from jaeger.utils.jtvae_utils import check_for_similarity
from jaeger.utils.jtvae_utils import check_for_similarity_to_collection_fp

# --- utils
import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")



### HERE I HAVE MOSTLY STREAMLIT CACHED FUNCTIONS
#try: 
import streamlit as st
@st.cache(
    hash_funcs={
        pd.DataFrame: lambda _: None,
    },
    suppress_st_warning=True,    
    persist=True,
)
def load_csv(csv_file, index_col):
    df = pd.read_csv(csv_file, index_col = index_col)
    return df

@st.cache(
    hash_funcs={
        rdkit.DataStructs.cDataStructs.UIntSparseIntVect: lambda _: None,
        rdkit.Chem.rdchem.Mol: lambda _: None,
        pd.DataFrame: lambda _: None,
    },
    suppress_st_warning=True,    
    persist=True,
)
def load_data(csv_file, filter_mols=True,
              drop_qualified=False,
              binary_fp=True, # because these morgans are not built for modelling, but for similarity checking
              pac50=True,
              convert_fps_to_numpy = False):
    # ok, from now on (2020 April 23)
    # we'll be using pAC50s
    # st.write("Cache miss ... " + csv_file)
    print("Cache miss ... " + csv_file)
    print("FILTER MOLS " + str(filter_mols))
    morgans_df, targets, toxdata = modelling_data_from_csv(csv_file,
                                                                binary_fp=binary_fp, 
                                                                filter_mols=filter_mols,
                                                                drop_qualified =drop_qualified,
                                                                convert_to_pac50 = True,
                                                                convert_fps_to_numpy = convert_fps_to_numpy)
    print("TOXDATA LEN " + str(len(toxdata)))
    return morgans_df, targets, toxdata

@st.cache(
    allow_output_mutation=True, hash_funcs={pd.DataFrame: lambda _: None},
)
def get_vocabulary(assay_dir, assay_id, toxdata):
    print("getting vocab")
    return get_vocab(assay_dir, assay_id, toxdata)

@st.cache(
    hash_funcs={    
        torch.nn.parameter.Parameter: lambda _: None,
        torch.Tensor: lambda _: None,
    },
    allow_output_mutation=True,
)
def get_model(vocab, model_params, device, infer_dir):
    torch.manual_seed(777)
    model = JTPropVAE(vocab, **model_params).to(device)
    model.load_state_dict(torch.load(infer_dir + "/model-ref.iter-35"))
    model = model.eval()
    return model


@st.cache(allow_output_mutation=True, persist=True,)
def get_embeddings(embeddings_csv_file):
    print("getting embeddings")
    latent = pd.read_csv(embeddings_csv_file, index_col=0, engine="c")
    return latent

@st.cache(allow_output_mutation=True, persist=True,)
def get_predictions(predictions_csv_file, convert_to_pac50):
    print("getting predictions")
    predictions = pd.read_csv(predictions_csv_file, index_col=0, engine="c")
    if convert_to_pac50:
        predictions = (predictions - 6) * -1 # also convert the ground truth?

    return predictions


#@st.cache
def load_avail_models():
    avail_models_file = AVAIL_MODELS
    available_models = pd.read_csv(avail_models_file, index_col='assay_id')
    return available_models


@st.cache
def compute_pca(embeddings):
    latent_size = embeddings.shape[1]
    reducer = PCA(n_components=latent_size)
    crds_pca = reducer.fit_transform(embeddings)
    var_explained = reducer.explained_variance_
    var_explained_ratios = reducer.explained_variance_ratio_
    var_ticks = np.arange(0, latent_size)
    var_coords = np.array(list(zip(var_ticks, np.cumsum(var_explained_ratios))))
    return reducer, crds_pca, var_coords, var_explained

#except:
#    e = sys.exc_info()[0]
#    print("Unexpected error")
#    print(e)



