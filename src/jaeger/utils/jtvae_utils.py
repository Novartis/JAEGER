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
# ---
import os
import random
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import rdkit.Chem as Chem
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# --- JT-VAE
from jtnn import *  # not cool, but this is how they do it ...
from jtnn.datautils import ToxPropDataset
# --- disable rdkit warnings
from rdkit import RDLogger
from torch.utils import data
from toxsquad.data import *
from toxsquad.losses import *
from toxsquad.modelling import *
from toxsquad.visualizations import Visualizations

# --- toxsquad






lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


import os
import pickle

# ------------ PRE-PROCESSING ROUTINES ------------
from mol_tree import *


def save_object(obj, filename):
    with open(filename, "wb") as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def open_object(filename):
    with open(filename, "rb") as input:
        reopened = pickle.load(input)
    return reopened


def get_vocab(assay_dir, assay_id, toxdata):
    filename = assay_dir + "/jtvae/" + str(assay_id) + "-vocab.pkl"
    if os.path.isfile(filename):
        print("Re-opening vocabulary file")
        vocab = open_object(filename)
    else:
        print("Deriving vocabulary")
        vocab = set()
        for (
            smiles
        ) in toxdata.smiles:  # I guess here we should only use the training data??
            mol = MolTree(smiles)
            for c in mol.nodes:
                vocab.add(c.smiles)
        vocab = Vocab(list(vocab))
        save_object(vocab, filename)
    return vocab


# ------------ MODEL OPTIMIZATION ROUTINES ------------
def derive_inference_model(
    toxdata,
    vocab,
    infer_dir,
    model_params,
    vis,
    device,
    model_name,
    base_lr=0.003,
    beta=0.005,
    num_threads = 24,
    weight_decay = 0.000
        
):
    from jtnn.jtprop_vae import JTPropVAE

    smiles = toxdata.smiles
    props = toxdata.val
    dataset = ToxPropDataset(smiles, props)
    batch_size = 8
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_threads,
        collate_fn=lambda x: x,
        drop_last=True,
    )
    from jtnn.jtprop_vae import JTPropVAE

    model = JTPropVAE(vocab, **model_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
    scheduler.step()
    # --- pre-train AE
    total_step_count = 0
    total_step_count = pre_train_jtvae(
        model,
        optimizer,
        scheduler,
        dataloader,
        device,
        infer_dir,
        vis,
        total_step_count,
        model_name,
        MAX_EPOCH=36,
        PRINT_ITER=5,
    )
    # train (set a smaller initial LR, beta to  0.005)
    optimizer = optim.Adam(model.parameters(), lr=0.0003,weight_decay=weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
    scheduler.step()
    print("[DEBUG] TRAINING")
    total_step_count = train_jtvae(
        model,
        optimizer,
        scheduler,
        dataloader,
        device,
        infer_dir,
        vis,
        total_step_count,
        beta=0.005,
        model_name=model_name,
        MAX_EPOCH=36,
        PRINT_ITER=5,
    )

    # --- fine tune AE
    # optimizer = optim.Adam(model.parameters(), lr=0.0003)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
    # scheduler.step()
    # total_step_count = train_jtvae(model, optimizer, scheduler, dataloader, device, infer_dir, vis, total_step_count, 0.005, model_name, MAX_EPOCH=36, PRINT_ITER=5)






def cross_validate_jtvae(
        toxdata,
        partitions,
        xval_dir,
        vocab,
        model_params,
        device,
        model_name,
        base_lr=0.003,
        vis_host=None,
        vis_port=8097,
        assay_name="",
        num_threads = 24,
        weight_decay = 0.0000
):
    """
    :todo ensure same training parameters are used for inference and cross-val models
    """
    MAX_EPOCH = 36
    PRINT_ITER = 5
    run = 0
    scores = []
    for partition in partitions:
        # I/O
        save_dir = xval_dir + "/run-" + str(run)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        # vis
        if vis_host is not None:
            vis = Visualizations(
                env_name="jtvae-xval-" + str(assay_name) + "-run-" + str(run), server=vis_host, port=vis_port
            )
        else:
            vis = None
        # data

        smiles = toxdata.smiles.loc[partition["train"]]
        props = toxdata.val.loc[partition["train"]]
        dataset = ToxPropDataset(smiles, props)
        batch_size = 8
        dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_threads,
            collate_fn=lambda x: x,
            drop_last=True,
        )
        # model
        from jtnn.jtprop_vae import JTPropVAE

        model = JTPropVAE(vocab, **model_params).to(device)
        optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
        scheduler.step()


        # pretrain
        print("[DEBUG] PRETRAINING")
        total_step_count = pre_train_jtvae(
            model,
            optimizer,
            scheduler,
            dataloader,
            device,
            save_dir,
            vis,
            0,
            model_name,
            MAX_EPOCH=36,
            PRINT_ITER=5,
        )
        # train (set a smaller initial LR, beta to  0.005)
        optimizer = optim.Adam(model.parameters(), lr=0.0003,weight_decay=weight_decay)
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
        scheduler.step()
        print("[DEBUG] TRAINING")
        total_step_count = train_jtvae(
            model,
            optimizer,
            scheduler,
            dataloader,
            device,
            save_dir,
            vis,
            total_step_count,
            beta=0.005,
            model_name=model_name,
            MAX_EPOCH=36,
            PRINT_ITER=5,
        )
        # evaluate (only property prediction accuracy for now)
        scores.append(
            evaluate_predictions_model(
                model,
                toxdata.smiles.loc[partition["test"]],
                toxdata.val.loc[partition["test"]],
                vis,
            )
        )
        # memory management
        del model        
        del optimizer
        torch.cuda.empty_cache()
        run = run + 1
    return scores


def pre_train_jtvae(
    model,
    optimizer,
    scheduler,
    dataloader,
    device,
    model_dir,
    vis,
    total_step_count,
    model_name,
    MAX_EPOCH=36,
    PRINT_ITER=5,
):
    my_log = open(model_dir + "/loss-pre.txt", "w")
    for epoch in range(MAX_EPOCH):
        print("pre epoch: " + str(epoch))
        word_acc, topo_acc, assm_acc, steo_acc, prop_acc = 0, 0, 0, 0, 0
        for it, batch in enumerate(dataloader):
            for mol_tree, _ in batch:
                for node in mol_tree.nodes:
                    if node.label not in node.cands:
                        node.cands.append(node.label)
                        node.cand_mols.append(node.label_mol)
            model.zero_grad()
            torch.cuda.empty_cache()
            loss, kl_div, wacc, tacc, sacc, dacc, pacc = model(batch, beta=0)
            loss.backward()
            optimizer.step()
            word_acc += wacc
            topo_acc += tacc
            assm_acc += sacc
            steo_acc += dacc
            prop_acc += pacc
            if (it + 1) % PRINT_ITER == 0:
                word_acc = word_acc / PRINT_ITER * 100
                topo_acc = topo_acc / PRINT_ITER * 100
                assm_acc = assm_acc / PRINT_ITER * 100
                steo_acc = steo_acc / PRINT_ITER * 100
                prop_acc = prop_acc / PRINT_ITER
                if vis is not None:
                    vis.plot_loss(word_acc, total_step_count, 1, model_name, "word-acc")
                    vis.plot_loss(prop_acc, total_step_count, 1, model_name, "mse")
                print(
                    "Epoch: %d, Step: %d, KL: %.1f, Word: %.2f, Topo: %.2f, Assm: %.2f, Steo: %.2f, Prop: %.4f"
                    % (
                        epoch,
                        it + 1,
                        kl_div,
                        word_acc,
                        topo_acc,
                        assm_acc,
                        steo_acc,
                        prop_acc,
                    ),
                    file=my_log,
                    flush=True,
                )
                word_acc, topo_acc, assm_acc, steo_acc, prop_acc = 0, 0, 0, 0, 0
            del loss
            del kl_div
            total_step_count = total_step_count + 1
            torch.cuda.empty_cache()
        scheduler.step()
        print("learning rate: %.6f" % scheduler.get_lr()[0])
        torch.save(
            model.cpu().state_dict(), model_dir + "/model-pre.iter-" + str(epoch)
        )
        torch.cuda.empty_cache()
        model = model.to(device)
    my_log.close()
    return total_step_count


def train_jtvae(
    model,
    optimizer,
    scheduler,
    dataloader,
    device,
    model_dir,
    vis,
    total_step_count,
    beta,
    model_name,
    MAX_EPOCH=36,
    PRINT_ITER=5,
):
    my_log = open(model_dir + "/loss-ref.txt", "w")
    for epoch in range(MAX_EPOCH):
        print("epoch: " + str(epoch))
        word_acc, topo_acc, assm_acc, steo_acc, prop_acc = 0, 0, 0, 0, 0
        for it, batch in enumerate(dataloader):
            for mol_tree, _ in batch:
                for node in mol_tree.nodes:
                    if node.label not in node.cands:
                        node.cands.append(node.label)
                        node.cand_mols.append(node.label_mol)
            model.zero_grad()
            torch.cuda.empty_cache()
            loss, kl_div, wacc, tacc, sacc, dacc, pacc = model(batch, beta)
            loss.backward()
            optimizer.step()
            word_acc += wacc
            topo_acc += tacc
            assm_acc += sacc
            steo_acc += dacc
            prop_acc += pacc
            if (it + 1) % PRINT_ITER == 0:
                word_acc = word_acc / PRINT_ITER * 100
                topo_acc = topo_acc / PRINT_ITER * 100
                assm_acc = assm_acc / PRINT_ITER * 100
                steo_acc = steo_acc / PRINT_ITER * 100
                prop_acc /= PRINT_ITER
                if vis is not None:
                    vis.plot_loss(word_acc, total_step_count, 1, model_name, "word-acc")
                    vis.plot_loss(prop_acc, total_step_count, 1, model_name, "mse")
                print(
                    "Epoch: %d, Step: %d, KL: %.1f, Word: %.2f, Topo: %.2f, Assm: %.2f, Steo: %.2f, Prop: %.4f"
                    % (
                        epoch,
                        it + 1,
                        kl_div,
                        word_acc,
                        topo_acc,
                        assm_acc,
                        steo_acc,
                        prop_acc,
                    ),
                    file=my_log,
                    flush=True,
                )
                word_acc, topo_acc, assm_acc, steo_acc, prop_acc = 0, 0, 0, 0, 0
            # if (it + 1) % 1500 == 0:  # Fast annealing
            #    # does this make sense? With the smaller datasets
            #    # we don't get to 1500? Why is this happening?
            #    # I don't quite trust it
            #    # But here, since we call model.cpu()
            #    # we need to move the model to the device again
            #    # else we ran onto that weird issue!
            #    scheduler.step()
            #    print("learning rate: %.6f" % scheduler.get_lr()[0])
            #    #torch.save(
            #    #    model.cpu().state_dict(),
            #    #    model_dir + "/model-ref.iter-%d-%d" % (epoch, it + 1),
            #    #)
            #    model.to(device)
            del loss
            del kl_div
            total_step_count = total_step_count + 1
        scheduler.step()
        print("learning rate: %.6f" % scheduler.get_lr()[0])
        torch.save(
            model.cpu().state_dict(), model_dir + "/model-ref.iter-" + str(epoch)
        )  # is this the expensive part?
        model = model.to(device)
    my_log.close()
    return total_step_count


# ------------ MODEL EVALUATION ROUTINES ------------
def evaluate_predictions_model(model, smiles, props, vis):
    """
    Return evaluation objects for JT-VAE model.

    This function will return a list of [mse, r2] for the smiles passed in,
    and also return a 2-col matrix for plotting predicted vs. actual.

    vis object allows us to use Visdom to directly update
    a live performance plot view.

    :param model: JT-VAE model
    :param smiles: Pandas series with SMILES as entries
        We usually pass toxdata.smiles
    :param props: Pandas series with molecular activity or property to predict
    :param vis: Visualization object from toxsquad.visualizations

    :returns: Scores, coords
        - Scores is a list of mean squared error and correlation coefficient
          (for entire smiles batch). This is of length 2.
        - coords are x, y coordinates for the "performance plot"
          (where x=actual and y=predicted).
    """
    predictions = dict()
    n_molecules = len(smiles)
    coords = np.zeros((n_molecules, 2))
    # k = 0;
    model = model.eval()
    for k, idx in enumerate(smiles.index):
        print_status(k, n_molecules)
        sml = smiles.loc[idx]
        prop = props.loc[idx]
        # model.predict(sml) returns a torch tensor
        # on which we need to call .item()
        # to get the actual floating point value out.
        predictions[idx] = model.predict(sml).item()
        coords[k, 0] = prop.item()
        coords[k, 1] = predictions[idx]
        # k = k + 1;
    model = model.train()
    mse = np.mean((coords[:, 1] - coords[:, 0]) ** 2)
    corr = np.corrcoef(coords[:, 1], coords[:, 0])[0, 1]
    print("MSE: " + str(mse))
    print("Corr: " + str(corr))
    scores = []
    scores.append(mse)
    scores.append(corr)

    # TODO do reconstruction test

    if vis is not None:
        vis.plot_scatter_gt_predictions(
            coords, f"{mse:.2f}" + "-r: " + f"{corr:.2f}", ""
        )

    return scores, coords




# ------------ LATENT SPACE ROUTINES ------------
from numpy.random import choice
from rdkit import DataStructs
from rdkit.Chem import AllChem


def get_neighbor_along_direction_tree(sample_latent, direction, step_size):
    """
    Direction should be normalized
    Direction is in tree space
    """
    tree_vec, mol_vec = torch.chunk(sample_latent, 2, dim=1)
    new_tree_vec = tree_vec + (direction * step_size)
    new_sample = torch.cat([new_tree_vec, mol_vec], dim=1)
    return new_sample


def get_neighbor_along_direction_graph(sample_latent, direction, step_size):
    """
    Direction should be normalized
    """

    tree_vec, mol_vec = torch.chunk(sample_latent, 2, dim=1)
    # update graph
    new_mol_vec = mol_vec + (
        direction * step_size
    )  # maybe the step size will have to be different?
    new_sample = torch.cat([tree_vec, new_mol_vec], dim=1)
    return new_sample


def get_neighbors_along_directions_tree_then_graph(
    model,
    smiles,
    directions,
    scale_factors,
    direction_graph,
    scale_factor_graph,
    n_neighbors=10,
    val_to_beat=-2,
    max_cosine_distance=1.6,
    direction_graph_plus=None,
    convert_to_pac50=False,
):
    sample_latent = model.embed(smiles)

    n_directions = len(directions)
    new_samples = []

    int_step_sizes = np.arange(-n_neighbors, n_neighbors + 1, 1)
    idx = int_step_sizes == 0
    int_step_sizes = np.delete(int_step_sizes, np.where(idx)[0][0])
    actual_n_neighbors = len(int_step_sizes)

    # dynamic range (this adds a loot of additional samples ... just takes longer)
    step_sizes_graph = np.arange(-n_neighbors, n_neighbors + 1, 1)
    step_sizes_graph = step_sizes_graph * scale_factor_graph
    # fixed range (original implementation)
    step_sizes_graph_original = np.arange(-1, 2, 1)
    step_sizes_graph_original = (
        step_sizes_graph_original * 0.5
    )  # so here the step size is also fixed!

    step_sizes_graph = np.concatenate(
        (step_sizes_graph, step_sizes_graph_original), axis=None
    )

    actual_n_neighbors_graph = len(step_sizes_graph)

    # this is pretty quick, as it's just arimethic operations in latent space
    # todo: since cosine similarity in latent space correlates to an extent with
    # chemical similarity, we could further reduce the number of evaluations based on that
    cos = nn.CosineSimilarity(dim=1)
    for k in range(n_directions):  # iterate over axes
        step_sizes = int_step_sizes * scale_factors[k]
        for i in range(actual_n_neighbors):  # iterate over steps along axis
            sample = get_neighbor_along_direction_tree(
                sample_latent, directions[k], step_sizes[i]
            )  # tree sample
            for j in range(actual_n_neighbors_graph):  # iterate along graph axis
                graph_sample = get_neighbor_along_direction_graph(
                    sample, direction_graph, step_sizes_graph[j]
                )
                # check cosine
                cdistance = 1 - cos(sample_latent, graph_sample)
                if cdistance.item() < max_cosine_distance:
                    new_samples.append(graph_sample)
                # additional direction
                if direction_graph_plus is not None:
                    graph_sample = get_neighbor_along_direction_graph(
                        sample, direction_graph_plus, step_sizes_graph[j]
                    )
                    # check cosine
                    cdistance = 1 - cos(sample_latent, graph_sample)
                    if cdistance.item() < max_cosine_distance:
                        new_samples.append(graph_sample)

    # predict activity and decode samples (probably should be another function, also because this happens ALL the time)
    new_smiles, new_activities, new_samples = predict_and_decode_strict(
        model, new_samples, val_to_beat, convert_to_pac50
    )

    return (
        new_samples,
        new_smiles,
        new_activities,
        sample_latent.squeeze().cpu().detach().numpy(),
    )


# I guess the min val should be informed also relative to the MSE of the model
#
def predict_and_decode_strict(model, new_samples, min_val, convert_to_pac50=False):
    n_samples = len(new_samples)
    new_smiles = []
    new_activities = []
    my_bar = None
    filtered_samples = []
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
        prediction = (
            
            model.propNN(new_samples[i]).squeeze().cpu().detach().numpy()
        )  # compute the activity predictions

        if convert_to_pac50: 
            prediction = (prediction - 6) * -1

        # HIGHER IS BETTER    
        prediction_condition = prediction > min_val
            
        if prediction_condition:
            new_activities.append(prediction)
            tree_vec, mol_vec = torch.chunk(new_samples[i], 2, dim=1)
            more_smiles = model.decode(tree_vec, mol_vec, prob_decode=False)
            new_smiles.append(more_smiles)
            new_samples[i] = new_samples[i].squeeze().cpu().detach().numpy()
            filtered_samples.append(new_samples[i])
    return new_smiles, new_activities, filtered_samples


def predict_and_decode(model, new_samples, show_st=False):
    n_samples = len(new_samples)
    new_smiles = []
    new_activities = []
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
        prediction = (
            model.propNN(new_samples[i]).squeeze().cpu().detach().numpy()
        )  # compute the activity predictions
        new_activities.append(prediction)
        tree_vec, mol_vec = torch.chunk(new_samples[i], 2, dim=1)
        more_smiles = model.decode(tree_vec, mol_vec, prob_decode=False)
        new_smiles.append(more_smiles)
        new_samples[i] = new_samples[i].squeeze().cpu().detach().numpy()
    return new_smiles, new_activities




def sample_gaussian(mean, sigma, n_samples):
    center = mean
    covariance = sigma
    m = torch.distributions.MultivariateNormal(center, covariance)
    samples = []
    for i in range(n_samples):
        samples.append(m.sample())
    samples = torch.stack(samples)
    return samples

def sample_gaussian_and_predict(model, n_samples, mean, sigma):
    dim = int(model.latent_size)
    center = mean
    covariance = sigma
    m = torch.distributions.MultivariateNormal(center, covariance)
    samples = []
    for i in range(n_samples):
        samples.append(m.sample())
    samples = torch.stack(samples)
    cur_vec = create_var(samples.data, False)
    predictions = model.propNN(cur_vec).squeeze()
    vectors = cur_vec.cpu().detach().numpy()
    predictions = predictions.cpu().detach().numpy()

    return vectors, predictions


def get_embeddings(model, toxdata):
    k = 0
    n_molecules = len(toxdata)
    vectors = {}
    for idx in toxdata.smiles.index:
        print_status(k, n_molecules)
        sml = toxdata.smiles.loc[idx]
        vectors[idx] = model.embed(sml).cpu().detach().numpy().ravel()
        k = k + 1
    return vectors


from rdkit import DataStructs
from rdkit.Chem import AllChem
from sklearn.metrics.pairwise import cosine_similarity


def sample_latent_space(model, latent, n_samples=2000, decode=False):
    mu = torch.from_numpy(np.mean(latent).values).float()
    sigma = torch.from_numpy(np.cov(latent.values.transpose())).float()
    return sample_latent_space_pass_normal(model, mu, sigma, n_samples, decode)


def sample_latent_space_pass_normal(model, mu, sigma, n_samples=2000, decode=False):
    samples, samples_predictions = model.sample_gaussian_and_predict(
        n_samples, mu, sigma
    )  # this is fast
    samples = samples.astype("float64")
    samples_predictions = samples_predictions.astype("float64")
    # dim = int(model_params["latent_size"] / 2)
    dim = int(model.latent_size / 2)
    tree_vec = create_var(torch.from_numpy(samples[:, 0:dim]).float())
    mol_vec = create_var(torch.from_numpy(samples[:, dim : dim * 2]).float())
    samples_decoded = []
    if decode:
        for i in range(n_samples):
            print_status(i, n_samples)
            samples_decoded.append(
                model.decode(
                    tree_vec[i, :].reshape(1, -1),
                    mol_vec[i, :].reshape(1, -1),
                    prob_decode=False,
                )
            )  # this is slow

        samples_decoded_df = pd.DataFrame(data=samples_decoded)
        samples_decoded_df.columns = ["smiles"]
    else:
        samples_decoded_df = None

    return samples, samples_predictions, samples_decoded_df


# ------------ MISC ROUTINES ------------
def print_status(i, maxSteps):
    percent = "0.00"
    percentage = (float(i) / float(maxSteps)) * 100
    divisor = 5
    if i % divisor == 0:
        sys.stdout.write("Progress: %d%%   \r" % (percentage))
        sys.stdout.flush()


# ------------ DISTANCES ROUTINES ------------
def normalize_morgans(morgans):
    morgans_normalized = {}
    for key in morgans.keys():
        fp = morgans[key]
        fp_array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        morgans_normalized[key] = normalize_to_unity(fp_array)
    return morgans_normalized


def normalize_to_unity(fp):
    if np.sum(fp) == 0:
        print("invalid fp")
        return fp
    else:
        return fp / np.sum(fp)




import cadd.sascorer as sascorer
import networkx as nx
# ------------ CHEMISTRY ROUTINES ------------
from rdkit.Chem import Descriptors, rdmolops
from rdkit.Chem.Descriptors import ExactMolWt


def get_cycle_score(mol):
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    current_cycle_score = cycle_length
    return current_cycle_score


import cadd.sascorer as sascorer
# toxdata should include a mols value
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Descriptors import ExactMolWt

NumHDonors = lambda x: rdMolDescriptors.CalcNumHBD(x)
NumHAcceptors = lambda x: rdMolDescriptors.CalcNumHBA(x)
from rdkit.Chem import Descriptors

TPSA = lambda x: Descriptors.TPSA(x)




def compute_properties(toxdata):
    n_molecules = len(toxdata)
    k = 0
    mw = {}
    na = {}
    log_p = {}
    sas = {}
    cycle_scores = {}
    # more properties
    nhdon= {}
    nhacc = {}
    tpsa = {}
    
    
    for idx in toxdata.index:
        print_status(k, n_molecules)
        mol = toxdata.loc[idx].mols
        try:
            mw[idx] = ExactMolWt(mol)
            log_p[idx] = Descriptors.MolLogP(mol)
            sas[idx] = sascorer.calculateScore(mol)
            cycle_scores[idx] = get_cycle_score(mol)
            na[idx] = mol.GetNumAtoms()
            nhdon[idx] = NumHDonors(mol)
            nhacc[idx] = NumHAcceptors(mol)
            tpsa[idx] = TPSA(mol)
            
        except:
            print("[DEBUG] Error computing properties")
            mw[idx] = np.nan
            log_p[idx] = np.nan
            sas[idx] = np.nan
            cycle_scores[idx] = np.nan
            na[idx] = np.nan
            nhdon[idx] = np.nan
            nhacc[idx] = np.nan
            tpsa[idx]  = np.nan                                    
            continue
        k = k + 1

    props = [
        pd.DataFrame.from_dict(mw, orient="index"),
        pd.DataFrame.from_dict(log_p, orient="index"),
        pd.DataFrame.from_dict(sas, orient="index"),
        pd.DataFrame.from_dict(cycle_scores, orient="index"),
        pd.DataFrame.from_dict(na, orient="index"),
        pd.DataFrame.from_dict(nhdon, orient="index"),
        pd.DataFrame.from_dict(nhacc, orient="index"),
        pd.DataFrame.from_dict(tpsa, orient="index"),                
        
    ]

    props_df = pd.concat(props, axis=1)
    props_df.columns = ["mw", "log_p", "sas", "cycle_scores", "n_atoms","HBD",
                        "HBA",
                        "TPSA"]



    toxdata_props = pd.merge(toxdata, props_df, left_index=True, right_index=True)
    return toxdata_props


def check_for_similarity(ref_smiles, test_smiles, do_tanimoto=True):
    ref_mol = Chem.MolFromSmiles(ref_smiles)
    if do_tanimoto:
        ref_fp = AllChem.GetMorganFingerprint(ref_mol, 2)

    n_test = len(test_smiles)
    test_mols = []
    test_fps = []
    for i in range(n_test):
        try:
            test_mol = Chem.MolFromSmiles(test_smiles[i])
            test_mols.append(test_mol)
            test_fps.append(AllChem.GetMorganFingerprint(test_mol, 2))
        except:
            test_fps.append(
                ref_fp
            )  # this smiles will be dropped because we drop sim == 1
            test_mols.append(ref_mol)
    s = DataStructs.BulkTanimotoSimilarity(ref_fp, test_fps)
    return s, test_mols, test_fps


def check_for_similarity_to_collection_fp(test_fps, ref_fps, do_tanimoto=True):
    similarities = []
    n_test = len(test_fps)
    for i in range(n_test):
        dists = 1 - np.array(DataStructs.BulkTanimotoSimilarity(test_fps[i], ref_fps))
        similar = False
        if any(dists < 0.0001):
            similar = True
        similarities.append(similar)

    return similarities



from toxsquad.data import modelling_data_from_csv


def load_data(csv_file, filter_mols=True,
              drop_qualified=False,
              pac50=True,
              binary_fp = False):
    # ok, from now on (2020 April 23)
    # we'll be using pAC50s
    morgans_df, targets, toxdata = modelling_data_from_csv(csv_file,
                                                                filter_mols=filter_mols,
                                                                drop_qualified =drop_qualified,
                                                                convert_to_pac50 = pac50,
                                                                binary_fp = binary_fp)
    return morgans_df, targets, toxdata


