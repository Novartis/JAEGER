# JT-VAE Generative Modeling (JAEGER)

Once you have trained a model, you can use the Streamlit interface in
`src/jaeger/streamlit/` to generate molecules.

# Create model registry

In the `$JAEGER_HOME` directory (cf. `src/jaeger/jaeger.py`, line 19), create a CSV file called `jaeger_avail_models.csv` with the following columns:

* `assay_id`: String with assay ID used to create a directory where the model was saved (cf. TRAINING.md)
* `csv_file`: String with the file path of the CSV file containing structures and AC50s used for training
* `pac50`: Boolean set to True if using pAC50 values for the activity models (default to True)
* `qualified`: Boolean to determine whether molecules with qualified AC50 values should be used (default to True)
* `assay_name`: String with name of assay
* `assay_name_short`: String with a short version of the assay name
* `unit`: Concentration units
* `filter_mols`: Flag to determine whether molecules with more than 50 atoms were left out (default to True)

Add one entry with the respective information per trained model
(cf. TRAINING.md).

A sample registry file is included in `models/jaeger_avail_models.csv`

# Start streamlit interface

```sh
cd src/jaeger/streamlit
streamlit run jaeger_generate.py
```

Open app in browser with the URL provided by Streamlit.

# Generate compounds

Select your model for given assay and seed compound.

## Options

### Use a compound included in assay?

Select "No" if the seed compound is not in the assay.
You can then input an arbitrary SMILES string as seed point.

### Select directions to sample

The directions to sample are the number of principal axes to explore in tree space.

### Select sampling strategy

Sampling strategy. Use `Deterministic` to use the sampling
strategy based on principal axes. In this strategy, only a single axis
is sampled in graph space. If more are desired, select
`DeterministicFullGraph`. Then an equal number of principal axes is
sampled in both tree and graph space.

### Select sampling density

This dictates how densely we'll sample the exploration neighborhood in
latent space. Sparser strategies run more quickly.

### Optimization direction

This option dictates whether we should increase or decrease the
desired activity.

### Filter molecules

Check this box if JAEGER should only decode promising molecules. Else all
sampled data points in latent space will be decoded (this is slower).

### Chemical similarity cutoff

Tanimoto value for minimum similarity to seed compound. Compounds with
similarities to the seed molecule below this value are discarded.

## Demo use


1. Open the streamlit app in your browser.
2. Select compound `GNF-Pf-1042` as the starting point.
3. Click on the `Generate` button at the bottom of the page.
4. Wait a couple of seconds (~100 seconds depending on your hardware) while the molecules are generated.
5. You should get 13 new virtual molecules with the default settings.
6. Scroll down to see the structures as well as a table with computed
   properties for the molecules.
7. At the bottom there are links to download the table with structures and properties.

The interace by default saves automatically the resulting molecules to
`$JAEGER_HOME/assays/<assay_id>/jtvae/jtvae-h-420-l-56-d-7/genchem/<seed_compound>`,
where `<seed_compound>` is the name of the seed compound.

For the generation process outlined above, you'll find a file called
`$JAEGER_HOME/assays/Novartis_GNF/jtvae/jtvae-h-420-l-56-d-7/genchem/GNF-Pf-1042/GNF-Pf-1042-dir-4-strategy-Deterministic-extent-Sparse-cutoff-0.2-filter-True-opt_direction-Increase-subset_pca-False.csv`
with the resulting molecules.

## Command-line interface

The script `jaeger_generate.py` can be likewise invoked from the
command-line. This is helpful for running more thorough searches
(e.g., with a larger number of directions as well as denser
exploration strategies).

Usage instructions can be displayed by typing the following:

```sh
cd src/jaeger/streamlit
python jaeger_generate.py --help
```

To reproduce the same generation process that we did in the streamlit app as outlined above, we would type:

```sh
python jaeger_generate.py --run_streamlit False --assay_id Novartis_GNF --cmpd GNF-Pf-1042 --smiles "COC1=CC2=C(C=C1OC)C(=O)N(NC(=O)C3=CC=CC(F)=C3)C(=N2)C4CCC4" --sel_direction 4 --sel_sampling_strategy Deterministic --sel_sampling_density Sparse --sim_cutoff 0.2 --filter_mols True --opt_direction Increase
```

Consistent with the streamlit interface, the script by default saves automatically the resulting molecules to `$JAEGER_HOME/assays/Novartis_GNF/jtvae/jtvae-h-420-l-56-d-7/genchem/GNF-Pf-1042/GNF-Pf-1042-dir-4-strategy-Deterministic-extent-Sparse-cutoff-0.2-filter-True-opt_direction-Increase-subset_pca-False.csv`.
