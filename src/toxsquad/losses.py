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


def variance_explained_(y_pred, y_true):
    return 1 - np.var(residual(y_pred, y_true)) / np.var(y_true)


from sklearn.metrics import r2_score
def r2_score_(y_pred, y_true):
    return r2_score(y_true, y_pred)


def mse_(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def residual(y_pred, y_true):
    return y_pred - y_true



def corr_(y_pred: np.ndarray, y_true: np.ndarray):
    """
    """
    x = y_pred
    y = y_true
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    corr = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
    return corr






import scipy
def rsquared(y, x): 
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2



