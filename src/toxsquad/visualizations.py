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

from datetime import datetime

try:
    import visdom
except ImportError:
    print("visdom not found.")
import numpy as np


class Visualizations:
    def __init__(
        self, env_name=None, server="http://server.company.net", port=8097,
    ):
        if env_name is None:
            env_name = str(datetime.now().strftime("%d-%m %Hh%M")) + "-torch"
        self.env_name = env_name
        self.vis = visdom.Visdom(
            env=self.env_name, server=server, port=port, use_incoming_socket=False,
        )
        self.loss_win = None
        self.scatter_win = None

    def get_viz(self):
        return self.vis

    def plot_loss(self, loss, step, nSteps, titel, name):
        self.loss_win = self.vis.line(
            [loss],
            [step],
            win=self.loss_win,
            name=name,
            update="append" if self.loss_win else None,
            opts=dict(xlabel="Step", ylabel="Loss", title=titel),
        )

    def plot_scatter_gt_predictions(self, coords, mse, titel, new_win=False):
        if new_win:
            self.scatter_win = None
        self.scatter_win = self.vis.scatter(
            coords,
            win=self.scatter_win,
            name="scatter",
            opts=dict(
                xlabel="Ground truth", ylabel="Prediction", title="MSE: " + mse + titel,
            ),
        )
        self.scatter_win = self.vis.line(
            X=[coords[:, 0].min(), coords[:, 0].max()],
            Y=[coords[:, 0].min(), coords[:, 0].max()],
            win=self.scatter_win,
            name="identity",
            update="insert",
            opts=dict(linecolor=np.array([[255, 0, 0]])),  # RGB
        )

