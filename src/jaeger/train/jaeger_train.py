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

# --- parse params

import argparse

# --- JT-VAE
from jtnn import *
from jtnn.chemutils import *
from jtnn.jtprop_vae import JTPropVAE

# --- JAEGET
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
        description=jgr.NAME + " train",
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
        "--num_threads", type=int, default=12, help="Number of workers",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight Decay",
    )
    parser.add_argument(
        "--drop_larger_mols",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Drop molecules with 50 or more atoms",
    )            
    parser.add_argument(
        "--vis_host", type=str, default="",help="Visdom host",
    )

    args = parser.parse_args()
    train(args.csv_file, args.assay_id, args.num_threads, args.use_qualified, args.weight_decay, filter_mols = args.drop_larger_mols, vis_host=args.vis_host)





import os
from jaeger.utils.jtvae_utils import load_data
def train(csv_file, assay_id, num_threads, use_qualified, weight_decay, filter_mols = True, vis_host = ""):
    # --- LOAD DATA
    drop_qualified = not use_qualified
    _, _, toxdata =load_data(csv_file, drop_qualified=drop_qualified, filter_mols = filter_mols)

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

    # --- derive model for inference / molecule optimization
    infer_dir = model_dir + "/infer/"
    if not os.path.exists(infer_dir):
        os.mkdir(infer_dir)

    # --- visdom session
    from toxsquad.visualizations import Visualizations
    if vis_host == "":
        vis = None
    else:   
        vis = Visualizations(
            env_name="jtvae-train-"+str(assay_id), server=vis_host, port=8097
        )

    TRAIN_INFERENCE_MODEL = True
    if TRAIN_INFERENCE_MODEL:
        derive_inference_model(
            toxdata, vocab, infer_dir, model_params, vis, device, model_name, num_threads=num_threads, weight_decay=weight_decay
        )
        

if __name__ == "__main__":
    main()
