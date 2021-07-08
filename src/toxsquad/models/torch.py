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

import torch
import torch.nn as nn
import torch.nn.functional as F

from skorch.regressor import NeuralNetRegressor

import toxsquad.modelling


class ToxNetMorgansMkIb(nn.Module):
    def getName(self):
        return self.name

    def __init__(self, n_units, drop_input=0, drop=0, n_bits=2048, n_out=1):
        super(ToxNetMorgansMkIb, self).__init__()
        # n_bits=2048 # morgan_bits TODO: parametrize
        # n_units=256;
        self.name = "mkib" + "-" + str(n_units) + "-" + str(drop)
        self.dropIn = nn.Dropout(p=drop_input)
        self.ip0 = nn.Linear(n_bits, n_units)
        # strong compression already ...
        self.bn0 = nn.BatchNorm1d(n_units)
        self.drop0 = nn.Dropout(p=drop)  #

        self.ip1 = nn.Linear(n_units, n_units)
        self.bn1 = nn.BatchNorm1d(n_units)
        self.drop1 = nn.Dropout(p=drop)

        self.ip2 = nn.Linear(n_units, n_units)
        self.bn2 = nn.BatchNorm1d(n_units)
        self.drop2 = nn.Dropout(p=drop)

        self.ip3 = nn.Linear(n_units, n_units)
        self.bn3 = nn.BatchNorm1d(n_units)
        self.drop3 = nn.Dropout(p=drop)

        self.ip4 = nn.Linear(n_units, n_units)
        self.bn4 = nn.BatchNorm1d(n_units)
        self.drop4 = nn.Dropout(p=drop)

        self.ip5 = nn.Linear(n_units, n_out)

        self.initialize_weights()

    def forward(self, x):
        # whole thing
        x = self.dropIn(x)
        x = self.drop0(F.leaky_relu(self.bn0(self.ip0(x)), 0.01))
        x = self.drop1(F.leaky_relu(self.bn1(self.ip1(x)), 0.01))
        x = self.drop2(F.leaky_relu(self.bn2(self.ip2(x)), 0.01))
        x = self.drop3(F.leaky_relu(self.bn3(self.ip3(x)), 0.01))
        x = self.drop4(F.leaky_relu(self.bn4(self.ip4(x)), 0.01))
        x = self.ip5(x)
        return x

    def initialize_weights(self):
        # weight initializations
        torch.nn.init.xavier_uniform_(self.ip0.weight)
        torch.nn.init.xavier_uniform_(self.ip1.weight)
        torch.nn.init.xavier_uniform_(self.ip2.weight)
        torch.nn.init.xavier_uniform_(self.ip3.weight)
        torch.nn.init.xavier_uniform_(self.ip4.weight)
        torch.nn.init.xavier_uniform_(self.ip5.weight)

        # bias initialization
        torch.nn.init.constant_(self.ip0.bias, 0)
        torch.nn.init.constant_(self.ip1.bias, 0)
        torch.nn.init.constant_(self.ip2.bias, 0)
        torch.nn.init.constant_(self.ip3.bias, 0)
        torch.nn.init.constant_(self.ip4.bias, 0)
        torch.nn.init.constant_(self.ip5.bias, 0)

        # batch params
        self.bn0.reset_parameters()
        self.bn1.reset_parameters()
        self.bn2.reset_parameters()
        self.bn3.reset_parameters()
        self.bn4.reset_parameters()


class ResidualPredictor(nn.Module):
    def __init__(self, input_size, layer_size, output_size, drop=0.1):
        super(ResidualPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, layer_size),
            nn.LeakyReLU(0.01),
            ResidualBlock(layer_size,drop = drop),
            ResidualBlock(layer_size,drop = drop),
            ResidualBlock(layer_size,drop = drop), # works with 512, better with 2k units
            ResidualBlock(layer_size,drop = drop),
            ResidualBlock(layer_size,drop = drop),
            ResidualBlock(layer_size,drop = drop),
            ResidualBlock(layer_size,drop = drop),
            nn.Linear(layer_size, output_size)
        )

    def forward(self, x):
        x = self.net(x) 
        return x

        

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size=None, drop=0):
        super(ResidualBlock, self).__init__()
        if output_size is None:
            output_size = input_size
        self.input_norm = nn.BatchNorm1d(input_size)
        self.ip0 = nn.Linear(input_size, output_size, bias=False)
        self.transform_norm = nn.BatchNorm1d(output_size)
        self.drop_pre_ip1 = nn.Dropout(p=drop)
        self.ip1 = nn.Linear(output_size, output_size, bias=False)
        if input_size != output_size:
            self.ipPrime = nn.Linear(input_size, output_size, bias=True)
        else:
            self.ipPrime = None

    def forward(self, x):
        residual = F.relu(self.input_norm(x))
        residual = self.ip0(residual)
        residual = self.drop_pre_ip1(F.relu(self.transform_norm(residual)))
        residual = self.ip1(residual)
        if self.ipPrime is None:
            x = x + residual
        else:
            x = self.ipPrime(x) + residual
        return x



from skorch.callbacks.lr_scheduler import LRScheduler
from skorch.regressor import NeuralNetRegressor
import torch.optim as optim
from toxsquad.data import MorgansDataset
def get_default_skorch_regressor_params(device="cpu",drop=0.1):
    # the skorch class
    gamma = 0.1
    scheduler = LRScheduler("StepLR", step_size=9, gamma=gamma)
    estimator_params = dict(
        module=toxsquad.models.ToxNetMorgansMkIb,
        module__n_units=2304,
        module__drop_input=0,
        module__drop=drop,
        module__n_bits=2048,
        criterion=nn.MSELoss,
        optimizer=optim.Adam,
        optimizer__weight_decay=0.0005,
        iterator_train__shuffle=True,
        iterator_train__num_workers=8,
        iterator_train__drop_last=True,
        lr=0.0003,
        max_epochs=27,
        batch_size=8,
        device=device,
        train_split=None,
        callbacks=[("scheduler", scheduler)],
    )
    return estimator_params

    
