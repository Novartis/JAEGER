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

import importlib
import sys

import numpy as np
import pandas as pd

from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import Chem

# --- disable rdkit warnings
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# --- JAEGER
import jaeger as jgr

# --- TOXSQUAD
from toxsquad.modelling import (
    shuffle_split
)


# --- parse params

import argparse

# --- JT-VAE
from jtnn import *
from jtnn.chemutils import *
from jtnn.jtprop_vae import JTPropVAE
from jaeger.utils.jtvae_utils import *

# --- utils
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(
        description=jgr.NAME + " xval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv_file", type=str, default="", help="CSV File",
    )
    parser.add_argument(
        "--use_qualified",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Use molecules with qualified values",
    )            
    parser.add_argument(
        "--assay_id", type=str, default="", help="Assay ID",
    )
    parser.add_argument(
        "--num_threads", type=int, default="", help="Number of workers",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight Decay",
    )
    parser.add_argument(
        "--seed", type=int, default=0,help="Seed for partitioning",
    )
    parser.add_argument(
        "--n_splits", type=int, default=3,help="Number of splits",
    )
    parser.add_argument(
        "--vis_host", type=str, default="",help="Visdom host",
    )

    

    
    print("[DEBUG] PARSING")
    args = parser.parse_args()

    xval(args.csv_file, args.assay_id, args.num_threads, args.use_qualified,weight_decay = args.weight_decay, seed = args.seed, n_splits = args.n_splits, vis_host = args.vis_host)





import os
from jaeger.utils.jtvae_utils import load_data
def xval(csv_file, assay_id, num_threads, use_qualified, n_splits=3, redo_partitions = True, weight_decay = 0.0, seed = 0, vis_host = ""):
    # --- LOAD DATA
    drop_qualified = not use_qualified
    morgans_df, targets, toxdata =load_data(csv_file, drop_qualified=drop_qualified) # so always using pAC50s now as transform
    # --- I/O
    assay_dir = jgr.BASE_DIR + "/" + str(assay_id)

    if not os.path.exists(assay_dir):
        os.mkdir(assay_dir)

    jtvae_dir = assay_dir + "/jtvae"
    if not os.path.exists(jtvae_dir):
        os.mkdir(jtvae_dir)
        
    # --- derive vocab
    vocab = get_vocab(assay_dir, assay_id, toxdata)

    # --- hardware settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # --- define model
    model_params = dict(hidden_size=420, latent_size=56, depth=7)

    model_name = (
        "jtvae-h-"
        + str(model_params["hidden_size"])
        + "-l-"
        + str(model_params["latent_size"])
        + "-d-"
        + str(model_params["depth"])
    )

    model_dir = assay_dir + "/jtvae/" + model_name
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if vis_host == "":
        vis_host = None

    CROSS_VALIDATE = True
    if CROSS_VALIDATE:
        xval_dir = model_dir + "/xval/"
        if not os.path.exists(xval_dir):
            os.mkdir(xval_dir)
        # --- partitions
        if redo_partitions:
            partitions = shuffle_split(n_splits, toxdata.index, seed = seed)
            # save them
            save_object(partitions, xval_dir + "/partitions.pkl")
        else:
            partitions = open_object(partitions, xval_dir + "/partitions.pkl")
        scores = cross_validate_jtvae(
            toxdata,
            partitions,
            xval_dir,
            vocab,
            model_params,
            device,
            model_name,
            base_lr=0.003,
            vis_host=vis_host,
            vis_port=8097,
            assay_name=assay_id,
            num_threads = num_threads,
            weight_decay = weight_decay
        )
        calculated_scores = []
        true_vs_predicted = []
        n_runs = len(scores)
        for i in range(n_runs):
            calculated_scores.append(scores[i][0])
            true_vs_predicted.append(scores[i][1])
        scores_df = pd.DataFrame(data=calculated_scores, columns=["mse", "corr"])
        scores_df.to_csv(xval_dir + "/scores.csv", index=False)
        true_vs_predicted_df = pd.DataFrame(
            data=np.concatenate(true_vs_predicted), columns=["true", "predicted"]
        )
        true_vs_predicted_df.to_csv(xval_dir + "/true_vs_predictions.csv", index=False)
        all_coords = np.concatenate(true_vs_predicted)

    
if __name__ == "__main__":
    main()
