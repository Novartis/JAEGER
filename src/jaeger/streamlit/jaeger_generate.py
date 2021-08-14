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
import json
import os
import time
from itertools import compress

# --- JAEGER imports
import jaeger as jgr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- RDKIT imports
import rdkit
import rdkit.Chem as Chem
# --- TORCH imports
import torch
import torch.nn as nn

from jaeger.utils.jtvae_utils import (
    check_for_similarity, check_for_similarity_to_collection_fp,
    compute_properties, get_neighbor_along_direction_graph,
    get_neighbor_along_direction_tree,
    open_object,
    print_status)
# --- JTVAE imports
from jtnn import *
from jtnn.jtprop_vae import JTPropVAE
from rdkit import DataStructs
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
# --- TOXSQUAD imports
from toxsquad.data import modelling_data_from_csv

import streamlit as st

# --- STREAMLIT cached functions
from jaeger import (compute_pca, get_embeddings, get_model, get_predictions,
                    get_vocabulary, load_avail_models, load_data)

# --- STREAMLIT workarounds
from stutils import get_table_download_link, render_svg

# --- JAEGER


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

import rdkit.Chem as Chem
from rdkit.Chem import Draw
from matplotlib import pyplot as plt
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem import rdDepictor

#DrawingOptions.atomLabelFontSize = 55                                                                                                        
#DrawingOptions.dotsPerAngstrom = 100                                                                                                         
#DrawingOptions.bondLineWidth = 3.0                                                                                                           

DrawingOptions.atomLabelFontSize = 82.5
DrawingOptions.dotsPerAngstrom = 150
DrawingOptions.bondLineWidth = 4.5
def get_svg(in_smile, visdom=False, res=400):
    """                                                                                                                                       
    This function draws a molecule and returns an svg representation                                                                          
    of the drawing suitable for high-res rendering.                                                                                           
                                                                                                                                              
    :param in_smile: molecule to draw as SMILES                                                                                               
    :param visdom: if using visdom for showing the SVG, set to true                                                                           
    :returns SVG text                                                                                                                         
                                                                                                                                              
    """
    mol = Chem.MolFromSmiles(in_smile)
    rdDepictor.Compute2DCoords(mol)
    mc = Chem.Mol(mol.ToBinary())
    Chem.Kekulize(mc)
    drawer = Draw.MolDraw2DSVG(res, res)
    drawer.SetFontSize(0.7)
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    if visdom:
        svg = drawer.GetDrawingText().replace("svg:", "") # uncomment this line for visdom                                                    
    else:
        svg = drawer.GetDrawingText()
    return svg


import argparse


def decode(model, new_samples, show_st=False):
    n_samples = len(new_samples)
    new_smiles = []
    my_bar = None
    if show_st:
        try:
            import streamlit as st

            st.write("Decoding progress")
            my_bar = st.progress(0)
        except ImportError:
            pass

    for i in range(n_samples):
        if my_bar is not None:
            my_bar.progress((i + 1) / n_samples)

        print_status(i, n_samples)
        tree_vec, mol_vec = torch.chunk(new_samples[i].reshape(1,-1), 2, dim=1)
        more_smiles = model.decode(tree_vec, mol_vec, prob_decode=False)
        new_smiles.append(more_smiles)
    return new_smiles


from scipy.stats import norm


from toxsquad.data import normalize_morgans




# POTENCY ONLY
def filter_samples(reference_sample, samples, model, bypass=False, direction="Increase"):
    model = model.eval()
    n_samples = len(samples)
    device = next(model.parameters()).device
    predictions = []
    # compute reference pAC50
    smpl = torch.from_numpy(np.array(reference_sample)).to(device)
    ref_pac50 = (
        model.propNN(smpl).squeeze().cpu().detach().numpy()
    )  
    
    for i in range(n_samples):
        smpl = torch.from_numpy(np.array(samples[i]).reshape(1,-1)).to(device)
        prediction = (
            model.propNN(smpl).squeeze().cpu().detach().numpy()
        )  # compute the activity predictions
        predictions.append(prediction)

    predictions = np.array(predictions)
    if direction == "Increase":
        fit_samples_idx = predictions > ref_pac50
    else:
        fit_samples_idx = predictions < ref_pac50

    if bypass:
        fit_samples_idx: predictions > 0

        
    fit_samples = list(compress(samples, fit_samples_idx))
    fit_samples_pac50 = predictions[fit_samples_idx]

    return_stats_samples = {}
    return_stats_samples['pac50']        =  fit_samples_pac50

    return_stats_ref = {}
    return_stats_ref['pac50']                =  ref_pac50
    
    return fit_samples, return_stats_samples, return_stats_ref



# --- search function (streamlit agnostic)
def search_new(
    toxdata,
    morgans_df,
    model,
    cmpd,
    smiles,
    directions,
    std_explained_tree,
    std_scale,
    directions_graph,
    std_explained_graph,
    n_steps,
    sim_cutoff,
    cosine_cutoff,
    sel_direction,
    search_strategy,
    do_visual_debug,
    reducer_tree,
    surf_fig,
    filter_mols,
    direction,
    predict_in_chem_space
):
    # --- compute samples
    with st.spinner('Computing samples ...'):
        if search_strategy == "Deterministic":
            neighbors_vectors, my_vector_local = get_neighbors_along_directions_tree_then_graph_simple(
                model,
                smiles,
                # OK
                directions[0:sel_direction],
                std_explained_tree[0:sel_direction] * std_scale,
                directions_graph[0],  # set to zero
                std_explained_graph[0] * std_scale,
                n_steps,
                cosine_cutoff,
            )
        elif search_strategy == "DeterminsticFullGraph":
            neighbors_vectors, my_vector_local = get_neighbors_along_directions_tree_then_graph_complex(
                model,
                smiles,
                # OK
                directions[0:sel_direction],
                std_explained_tree[0:sel_direction] * std_scale,
                directions_graph[0:sel_direction-1], # -1 because no null direction in graph space  
                std_explained_graph[0:sel_direction-1] * std_scale, # -1 because no null direction in graph space  
                n_steps,
                cosine_cutoff,
            )            
            
    # --- assign my_vector
    my_vector = my_vector_local
    
    
    # --- filter samples
    with st.spinner('Filtering samples ...'):
        bypass = not filter_mols
        fit_samples, sample_stats, ref_stats = filter_samples(my_vector, neighbors_vectors, model, bypass=bypass, direction=direction)
           
    # --- decode samples
    torch_samples = torch.from_numpy(np.array(fit_samples)).float().to(next(model.parameters()).device)
    neighbors_smiles = decode(model, torch_samples, True)

    # --- rename some variables
    neighbors_vectors = fit_samples

    
    # --- results dataframe
    df = pd.DataFrame.from_dict(sample_stats) 
    
    if len(df) > 0:
        df["latent"] = neighbors_vectors
        df["smiles"] = neighbors_smiles
        
        # remove redundant smiles
        df.drop_duplicates(subset="smiles", inplace=True)
        # compute similarity to reference
        s, mols, fps = check_for_similarity(smiles, df["smiles"].values)
        df["sim"] = s
        df["mols"] = mols                

        # filter based on whether compounds are in training set
        in_training = check_for_similarity_to_collection_fp(
            fps, list(morgans_df[0].values)
        )
        in_training = np.array(in_training)
        df = df[~in_training]

        # filter those that are "identical" to input in terms of Tanimoto
        one_idx = df.sim < 1
        # filter those that don't fulfill the similarity cutoff
        if filter_mols:
            another_idx = df.sim > sim_cutoff
            idx = np.logical_and(one_idx, another_idx)
            df = df[idx]

        # sort according to fitness
        df.sort_values(inplace=True, by=["pac50"], ascending=False)
        df.reset_index(drop=True, inplace=True)

        # prepend original to table
        try:
            original_mol = toxdata.loc[cmpd].mols,
        except:
            original_mol = Chem.MolFromSmiles(smiles)

        original_row = pd.DataFrame.from_records(ref_stats, index=[cmpd])
        original_row['latent'] = my_vector
        original_row['smiles'] = smiles
        original_row['sim'] = 1
        original_row['mols'] = original_mol 

        df = original_row.append(df)
        # compute properties
        df = compute_properties(df)
        
        # set compound names
        new_names = []
        new_names.append(cmpd)
        for i in range(1, len(df)):
            new_name = cmpd + "-GC-" + str(i).zfill(3)
            new_names.append(new_name)
        df.index = new_names
    # can be an empty DF that is returned
    return df




def get_neighbors_along_directions_tree_then_graph_simple(
    model,
    smiles,
    directions,
    scale_factors,
    direction_graph,
    scale_factor_graph,
    n_neighbors=10,
    max_cosine_distance=1.6,        
):
    """
    In this function we iterate over the tree principal axes and a SINGLE principal graph axis.
    Returns NUMPY vectors (not Torch tensors)
    """
    sample_latent = model.embed(smiles)
    n_directions = len(directions)
    new_samples = []

    # tree steps (scaled in for loop)
    int_step_sizes = np.arange(-n_neighbors, n_neighbors + 1, 1) 
    idx = int_step_sizes == 0
    int_step_sizes = np.delete(int_step_sizes, np.where(idx)[0][0])
    actual_n_neighbors = len(int_step_sizes)

    # graph steps (scaled directly here)
    step_sizes_graph = np.arange(-n_neighbors, n_neighbors + 1, 1) 
    step_sizes_graph = step_sizes_graph * scale_factor_graph 

    actual_n_neighbors_graph = len(step_sizes_graph)
    cos = nn.CosineSimilarity(dim=1)
    for k in range(n_directions):  # iterate over axes
        step_sizes = int_step_sizes * scale_factors[k] 
        for i in range(actual_n_neighbors):  # iterate over steps along axis
            sample = get_neighbor_along_direction_tree(
                sample_latent, directions[k], step_sizes[i]
            )  # tree sample
            for j in range(actual_n_neighbors_graph):  # iterate along graph axis
                graph_sample = get_neighbor_along_direction_graph(
                    sample, direction_graph, step_sizes_graph[j] # uhh, here the graph variation is constant
                )
                # check cosine
                cdistance = 1 - cos(sample_latent, graph_sample)
                if cdistance.item() < max_cosine_distance:
                    new_samples.append(graph_sample.squeeze().cpu().detach().numpy())
                
    center_vector = sample_latent.squeeze().cpu().detach().numpy(),
    return new_samples, center_vector


def get_neighbors_along_directions_tree_then_graph_complex(
    model,
    smiles,
    directions,
    scale_factors,
    direction_graph,
    scale_factor_graph,
    n_neighbors=10,
    max_cosine_distance=1.6,        
):
    """
    In this function we iterate over the tree principal axes and a MULTIPLE principal graph axes.
    Compared to the function in the jtvae_utils file, this one really
    only computes the ellipsoidal samples. It filters with the 
    cosine distance only.
    Returns NUMPY vectors (not Torch tensors)
    """
    sample_latent = model.embed(smiles)
    n_directions = len(directions)
    new_samples = []

    # tree steps (scaled in for loop)
    int_step_sizes = np.arange(-n_neighbors, n_neighbors + 1, 1) 
    idx = int_step_sizes == 0
    int_step_sizes = np.delete(int_step_sizes, np.where(idx)[0][0])
    actual_n_neighbors = len(int_step_sizes)

    # graph steps (scaled directly here)  
    step_sizes_graph = np.arange(-n_neighbors, n_neighbors + 1, 1)
    actual_n_neighbors_graph = len(step_sizes_graph)    

    actual_n_neighbors_graph = len(step_sizes_graph)
    cos = nn.CosineSimilarity(dim=1)
    for k in range(n_directions):  # iterate over tree axes
        step_sizes = int_step_sizes * scale_factors[k] 
        for i in range(actual_n_neighbors):  # iterate over steps along tree axis
            sample = get_neighbor_along_direction_tree(
                sample_latent, directions[k], step_sizes[i]
            )  # tree sample

            for h in range(n_directions-1): # iterate over graph axes (no null direction)
                step_sizes_j = step_sizes_graph * scale_factor_graph[h]
                for j in range(actual_n_neighbors_graph):  # iterate along graph axis
                    graph_sample = get_neighbor_along_direction_graph(
                        sample, direction_graph[h], step_sizes_j[j]
                    )
                    # check cosine
                    cdistance = 1 - cos(sample_latent, graph_sample)
                    if cdistance.item() < max_cosine_distance:
                        new_samples.append(graph_sample.squeeze().cpu().detach().numpy())
                
    center_vector = sample_latent.squeeze().cpu().detach().numpy(),
    return new_samples, center_vector        

def enable_dropout(m, p= 0.1):
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.train()
            each_module.p = p

            

def predict_with_dropout(estimator, X, T=30, p=0.5):
    torch.manual_seed(777) # for reproducibility
    estimator.module_.eval()
    enable_dropout(estimator.module_,p=p)
    device = next(estimator.module_.parameters()).device
    n_samples = len(X)
    dropout_predictions = np.zeros((n_samples,T))
    for i in range(n_samples):            
        sample = torch.from_numpy(X[i,:].reshape(1,-1))
        samples = sample.repeat(T, 1).to(device)
        with torch.no_grad():
            dropout_predictions[i,:] = estimator.module_.forward(samples).squeeze().cpu().numpy()
    return dropout_predictions





def search_app(
        available_models,
        args,
):
    # -- Streamlit app title
    if args.run_streamlit:
        st.title(jgr.NAME + "+ Compound Generator")
        st.sidebar.header("Options")
        st.header("Assay Selection")
        assay_names  = available_models['assay_name']
        assay_names_selected = st.selectbox(
            "Available assays",
            assay_names,
            index=0
        )
        selected_index = available_models.index[available_models.assay_name == assay_names_selected].tolist()[0]
        assay_id = selected_index
    else:
        assay_id = args.assay_id

        
        
    csv_file = available_models.loc[assay_id].csv_file
    using_pac50 = available_models.loc[assay_id].pac50
    qualified = available_models.loc[assay_id].qualified
    drop_qualified = not qualified
    filter_mols = available_models.loc[assay_id].filter_mols
    
    model_name = 'jtvae-h-420-l-56-d-7' #TODO change this
    
    base_dir = jgr.BASE_DIR
    assay_dir = base_dir + "/" + str(assay_id)
    jtvae_dir = assay_dir + "/jtvae"
    model_dir = assay_dir + "/jtvae/" + model_name
    genchem_dir = model_dir + "/genchem"
    if not os.path.exists(genchem_dir): 
        os.mkdir(genchem_dir)
        os.chmod(genchem_dir, 0o777) #else Scriptr runs fail

    infer_dir = model_dir + "/infer/"
    embeddings_csv_file = model_dir + "/embeddings_training.csv" #THIS IS THE SUBSET, not the ADME-EXPANDED COMPOUND SET

    # --- load data
    morgans_df, targets,toxdata = load_data(csv_file, pac50=using_pac50, drop_qualified = drop_qualified, filter_mols = filter_mols)
    
    # --- load model
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_params = dict(hidden_size=420, latent_size=56, depth=7)
    vocab = get_vocabulary(assay_dir, assay_id, toxdata)
    model = get_model(vocab, model_params, device, infer_dir)

    
    # --- load predictive models
    FOUND_MODELS = True

    
    
    # --- load embeddings for training data (TREX ONLY)
    latent = get_embeddings(embeddings_csv_file)
    embeddings = latent.values
    subset_pca = False


    # --- Compute pca
    # joint pca
    reducer, crds_pca, _, var_explained = compute_pca(embeddings[:,:])
    std_explained = np.sqrt(var_explained)    
    components = reducer.components_
    
    # tree pca
    tree_dim = int(model_params["latent_size"] / 2)
    reducer_tree, crds_pca_tree, _, var_explained_tree = compute_pca(embeddings[:, 0:tree_dim])
    std_explained_tree = np.sqrt(var_explained_tree)
    components_tree = reducer_tree.components_

    # graph pca
    reducer_graph, _, _, var_explained_graph = compute_pca(embeddings[:, tree_dim:])
    std_explained_graph = np.sqrt(var_explained_graph)
    components_graph = reducer_graph.components_

    # --- set directions along PCs
    # directions are main components
    directions = torch.from_numpy(components_tree).to(device).float()
    directions_graph = torch.from_numpy(components_graph).to(device).float()
    directions_graph_plus = directions[0]  # actually, tree direction in graph space. Was a fluke but generates interesting things
    # space returned some interesting compounds!

    # null direction on tree space (so we explore neighbors starting at
    # compound in graph space, only)
    direction_null = torch.zeros(directions[0].shape).to(device).float()
    directions = torch.cat([direction_null.reshape(1, -1), directions], dim=0)
    std_explained_tree = np.insert(std_explained_tree, 0, 0)

    search_extents = dict(
        Sparse=(1, 2), Dense=(0.5, 5), SuperDense=(0.25, 10), MegaDense=(0.1, 25)
    )


    # --- visual debug
    DO_VISUAL_DEBUG = False
    if not args.run_streamlit:
        DO_VISUAL_DEBUG = False
    # actually, predict pac50s


    # --- Compound selection
    if args.run_streamlit:
        st.header("Compound Selection")
        use_cmpd_in_training = st.sidebar.radio("Use a compound included in assay?",('Yes','No'))
        ids = toxdata.index.values
        if use_cmpd_in_training == "Yes":
            cmpd = st.selectbox(
                "Select a seed compound",
                ids,
                index=0,
            )
            smiles = toxdata.loc[cmpd].smiles
        else:
            # free input
            cmpd = st.text_input("Compound name", value='cmpd-1', key=None, type='default')
            smiles = st.text_input("SMILES", value='CCC', key=None, type='default')
    else:
        cmpd = args.cmpd # now we require the SMILES to be passed, too
        smiles = args.smiles

    svg = get_svg(smiles)
    render_svg(svg)

    # SEARCH PARAMETERS
    if args.run_streamlit:
        sel_direction = st.sidebar.selectbox(
            "Select directions to sample", np.arange(1, tree_dim + 1, 1), index=3
        )
        sel_sampling_strategy = st.sidebar.selectbox(
            "Select sampling strategy", list(["Deterministic", "DeterministicFullGraph"]), index=0
        )            
        sel_sampling_density = st.sidebar.selectbox(
            "Select sampling density", list(search_extents.keys()), index=0
        )
        direction = st.sidebar.selectbox(
            "Optimization direction", ["Increase","Decrease"], index=0
        )        
        filter_mols = st.sidebar.checkbox(
            "Filter molecules", True
        )
        predict_in_chem_space = False

        sim_cutoff = st.sidebar.slider(
            "Chemical similarity cutoff",
            min_value=0.0,
            step=0.1,
            max_value=1.0,
            value=0.2,
        )
        if not filter_mols:
            sim_cutoff = 0


    else:
        sel_direction = args.sel_direction
        sel_sampling_density = args.sel_sampling_density
        sel_sampling_strategy = args.sel_sampling_strategy
        sim_cutoff = args.sim_cutoff
        filter_mols = args.filter_mols
        direction = args.opt_direction
        predict_in_chem_space = False


    strategy_params = search_extents[sel_sampling_density]
    std_scale = strategy_params[0]
    n_steps = strategy_params[1]

    cosine_cutoff = ((1 - sim_cutoff) * 2) + 0.2
    #st.sidebar.write(cosine_cutoff)        
    if cosine_cutoff > 2:
        cosine_cutoff = 2

    #st.sidebar.write(cosine_cutoff)        

    # VISUALIZE FITNESS LANDSCAPE!
    if DO_VISUAL_DEBUG:
        surf_fig = compute_surface(crds_pca_tree, toxdata)
        st.plotly_chart(surf_fig)
    else:
        surf_fig = None
        
        
    do_search = None
    
    if all((args.run_streamlit, FOUND_MODELS)):
        do_search = st.button("Generate")
        

    

    # --- SEARCH
    if all((any((do_search, not args.run_streamlit)), FOUND_MODELS)):
        if args.run_streamlit:
            st.header("Generated compounds")
        t0 = time.time()
        sel_direction = sel_direction + 1  # because of null tree vector        
        df = search_new(
            toxdata,
            morgans_df,
            model,
            cmpd,
            smiles,
            directions,
            std_explained_tree,
            std_scale,
            directions_graph,
            std_explained_graph,
            n_steps,
            sim_cutoff,
            cosine_cutoff,
            sel_direction,
            sel_sampling_strategy,
            DO_VISUAL_DEBUG,
            reducer_tree,
            surf_fig,
            filter_mols,
            direction,
            predict_in_chem_space
        )

        t1 = time.time()
        ellapsed = t1 - t0

        if len(df) > 1:
            actual_n_neighbors = len(df)
            if args.run_streamlit:
                st.write(
                    "Found "
                    + str(actual_n_neighbors - 1)
                    + " neighbors ("
                    + "{:.2f}".format(ellapsed)
                    + " seconds)"
                )
                # --- show smiles
                for i in range(actual_n_neighbors):
                    new_smiles = df["smiles"].iloc[i]
                    st.text(str(df.index[i]) + ": " + str(new_smiles))
                    smile_svg = get_svg(new_smiles, res=200)
                    render_svg(smile_svg)

            
            summary_df = df
            del df['latent']
            del df['mols']
            summary_df["source_cmpd"] = cmpd

            # surface (DISABLE!)
            if args.run_streamlit:
                SHOW_SURF = False
                if SHOW_SURF:
                    surf_fig = compute_surface(crds_pca, toxdata)  # surface is joint projection
                    
                    surf_fig = add_point_to_surface(
                        surf_fig,
                        reducer.transform(df["latent"].iloc[0].reshape(1, -1)),
                        "circle",
                        "start: " + str(df.index[0]) + ": " +  str(df["val"].iloc[0]),
                        color="cyan",
                    )

                    for i in range(1, actual_n_neighbors):
                        surf_fig = add_point_to_surface(
                            surf_fig,
                            reducer.transform(df["latent"].iloc[i].reshape(1, -1)),
                            "cross",
                            str(df.index[i]) + ": " + str(df["val"].iloc[i]),
                            color="magenta",
                        )

                    st.subheader("Neighbor Positions")
                    st.plotly_chart(surf_fig)
            
            if args.run_streamlit:
                st.subheader("Summary")
                st.table(summary_df)  # todo, save automatically somewhere
                st.markdown(
                    get_table_download_link(summary_df, "summary file", "jaeger-" + str(cmpd) + ".csv"),
                    unsafe_allow_html=True,
                )

                search_params = {
                    "cmpd": [cmpd],
                    "sel_direction": [int(sel_direction - 1)],
                    "search_strategy": [sel_sampling_strategy],
                    "search_extent": [sel_sampling_density],                    
                    "std_scale": [std_scale],
                    "n_steps": [int(n_steps)],
                    "sim_cutoff": [sim_cutoff],
                    "cosine_cutoff": [cosine_cutoff],
                    "filter_mols": [filter_mols],
                    "opt_direction": [direction]
                    
                }
                search_params_df = pd.DataFrame.from_dict(search_params)
                st.markdown(
                    get_table_download_link(search_params_df, "search parameters"),
                    unsafe_allow_html=True,
                )

            # SAVE FILES AND USER INPUT PARAMETERS
            cmpd_dir = genchem_dir + "/" + str(cmpd)            
            if not os.path.exists(cmpd_dir):
                os.mkdir(cmpd_dir)
                os.chmod(cmpd_dir, 0o777)

            run_name = (
                cmpd
                + "-dir-"
                + str(sel_direction - 1)
                + "-strategy-"
                + sel_sampling_strategy
                + "-extent-"
                + sel_sampling_density
                + "-cutoff-"
                + str(sim_cutoff)
                + "-filter-"
                + str(filter_mols)
                + "-opt_direction-"
                + str(direction)
                + "-subset_pca-"
                + str(subset_pca)                                
            )
            outfile = cmpd_dir + "/" + run_name + ".csv"
            summary_df.to_csv(outfile)
            os.chmod(outfile, 0o777)
            search_params = {
                "cmpd": cmpd,
                "sel_direction": int(sel_direction - 1),
                "search_strategy": sel_sampling_density,
                "std_scale": std_scale,
                "n_steps": int(n_steps),
                "sim_cutoff": sim_cutoff,
                "cosine_cutoff": cosine_cutoff,
            }
            search_params["file"] = outfile

        else:
            if args.run_streamlit:
                st.write("Did not find suitable neighbors")
            else:
                print("[DEBUG] Did not find suitable neighbors")





def main():
    # --- parse parameters
    parser = argparse.ArgumentParser(
        description=jgr.NAME + " Compound Generator ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )    
    parser.add_argument(
        "--run_streamlit",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Run streamlit app",
    )
    parser.add_argument(
        "--assay_id", type=str, default="Novartis_GNF", help="assay",
    )    
    parser.add_argument(
        "--cmpd", type=str, default="cpmd-1", help="Starting compound",
    )
    parser.add_argument(
        "--smiles", type=str, default="CCC", help="Structure",
    )    
    parser.add_argument(
        "--sel_direction", type=int, default=4, help="Number of PC directions to use",
    )
    parser.add_argument(
        "--sel_sampling_strategy", type=str,
            nargs="?",
            default="Deterministic",
            help="Options are Deterministic or DeterministicFullGraph",
    )
    parser.add_argument(
        "--sel_sampling_density",
        type=str,
        default="SuperDense",
        help="Options are Sparse, Dense, SuperDense, and MegaDense",
    )
    parser.add_argument(
        "--sim_cutoff", type=float, default=0.2, help="Tanimoto similarity cutoff ",
    )
    parser.add_argument(
        "--filter_mols", type=str2bool, default=True, help="Filter molecules ",
    )
    parser.add_argument(
        "--opt_direction",
        type=str,
        default="Increase",
        help="Options are Increase or Decrease",
    )
    
    
    args = parser.parse_args()
   

    
    # --- load assay data
    available_models = load_avail_models();

    search_app(
        available_models,
        args,
    )


if __name__ == "__main__":
    main()
