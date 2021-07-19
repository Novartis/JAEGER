# JAEGER

**J**T-V**AE** **G**en**er**ative Modeling (**JAEGER**) is a deep
generative approach for small-molecule design. JAEGER is based on the
Junction-Tree Variational Auto-Encoder (JT-VAE) method [1], which
ensures chemical validity for the generated molecules.

JAEGER is trained on existing molecules associated with activity
values measured in a given assay. During training, JAEGER learns how
to map each molecule onto a (high-dimensional) coordinate space, often
referred to as the **latent space**. JAEGER also learns how to
**decode** a coordinate position in the latent space back to a
molecule.

To generate new molecules, JAEGER defines numerical search strategies
to efficiently and effectively explore that latent space. JAEGER
couples the exploration of that latent space together with activity
predictive models to discover and optimize novel active molecules.

[1] Wengong Jin, Regina Barzilay, Tommi S. Jaakkola: Junction Tree
Variational Autoencoder for Molecular Graph Generation. ICML 2018:
2328-2337

## Installation

* Install torch, skorch, pyjanitor, rdkit, and streamlit in your python environment.

* Get a copy of the JT-VAE JAEGER branch 
  [here](https://github.com/PsiGamma/icml18-jtnn/tree/jaeger)
  Include the `icml18-jtnn` directory in your python path.

* Get a copy of the JAEGER repo (this repo). Include the `src`
  directory in your python path

## Training a model

See TRAINING.md

## Generating molecules

See GENCHEM.md

## License

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


See LICENSE.txt

# Contact

william_jose.godinez_navarro@novartis.com
