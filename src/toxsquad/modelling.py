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

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from joblib import parallel_backend
from sklearn.model_selection import ShuffleSplit




# ---nn stuff
def shuffle_split(n_splits: int, structure_ids: pd.Series, fraction: float = 0.9, seed: int = 0):
    """
    Return train/test partitions for each shuffle split iteration.

    :param n_splits: Number of splits to make on the data.
    :param structure_ids: A list of structure IDs. Used downstream to index
        into a pandas DataFrame that contains the structure IDs as the index.
        Each structure ID maps onto a unique compound identifier. 
    :returns: A list of dictionaries. Each dictionary contains two keys, one
        for 'train' and one for 'test', and the values are structure IDs that
        belong in the train or test set respectively for that split. There are
        ``n_splits`` elements in the list of dictionaries.
    """
    ss = ShuffleSplit(n_splits=n_splits, random_state=seed, train_size = fraction)
    partitions = []
    for train_index, test_index in ss.split(structure_ids):
        training = structure_ids[train_index]
        test = structure_ids[test_index]
        assert len(training) + len(test) == len(
            structure_ids
        ), "Some samples are  missing in the training or test set."
        partition = {"train": training.values, "test": test.values}
        partitions.append(partition)
    return partitions



def cross_val_score(
    estimator,
    morgans,
    targets,
    partitions,
    scoring_funcs: dict,
    scoring_funcs_kwargs: dict = {},
    callbacks=[],
    return_predictions=False,
    use_dask=True,
):
    """
    Custom cross-validation function for a generic model.

    The intent behind this function is to be model-agnostic. We want to be able
    to handle both PyTorch, sklearn, Chainer, and jax models.

    Inspired by William's original custom implementation for RF.
    We will generalize the ML model interface
    to make it compatible with sklearn.

    :param estimator: A ML model object. Should implement .fit()
        and .predict(), in accordance with the sklearn API.
    :param morgans: Pandas dataframe of morgan fingerprints.
    :param targets: Pandas series of values to predict.
    :param partitions: Partitions of data.
    :param scoring_funcs: A dictionary where a string keys a scoring function
        that has the signature ``func(y_pred, y_true)``.
    :param model_kwargs: A dictionary of keyword arguments that are relevant
        to the model.
    """
    scores = defaultdict(list)
    cv_preds = list()
    cv_idxs = list()
    for partition in partitions:
        USING_DICT=False
        try:
            train_morgans = morgans.loc[partition["train"]]
            test_morgans = morgans.loc[partition["test"]]
        except:
            # try dictionary
            train_morgans = np.array([morgans.get(key) for key in partition["train"]]).astype(np.float32)
            test_morgans = np.array([morgans.get(key) for key in partition["test"]]).astype(np.float32)
            USING_DICT =True
            
        train_targets = targets.loc[partition["train"]]
        test_targets = targets.loc[partition["test"]]


        cv_idxs.extend(partition["test"])
        if (
            use_dask
        ):  # not pretty but works ... I don't always have a dask instance running
            with parallel_backend("dask"):
                estimator.fit(train_morgans, train_targets)
        else:
            try:
                estimator.fit(train_morgans, train_targets)
            except:
                # SKORCH FAILS WITH 1D TARGETS
                if len(train_targets.values.shape) == 1:
                    if not USING_DICT:
                        estimator.fit(train_morgans.values.astype(np.float32), train_targets.values.reshape(-1,1).astype(np.float32))
                    else:
                        estimator.fit(train_morgans, train_targets.values.reshape(-1,1).astype(np.float32))

        try:
            preds = estimator.predict(test_morgans)
        except:
            # SKORCH IS NOT CONVERTING THE DATA PROPERLY THIS TIME AROUND
            # DID I MISS SOMETHING
            if not USING_DICT:
                preds = estimator.predict(test_morgans.values.astype(np.float32))
            else:
                preds = estimator.predict(test_morgans)

        cv_preds.extend(preds)
        for scorer, func in scoring_funcs.items():
            kwargs = scoring_funcs_kwargs.get(scorer, {})
            print(preds.squeeze().shape)
            print(test_targets.squeeze().shape)
            scores[scorer].append(
                func(preds.squeeze(), test_targets.squeeze(), **kwargs)
            )
            print(scores[scorer])
    if callbacks:
        for callback in callbacks:
            callback()

    if return_predictions:
        return scores, cv_preds, cv_idxs
    else:
        return scores

