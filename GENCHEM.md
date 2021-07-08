# JT-VAE Generative Modeling (JAEGER)

Once trained, you can use the Streamlit interface in `src/jaeger/streamlit/` to generate molecules.

# Create model registry

In the `$JAEGER_HOME` directory (cf. `src/jaeger/jaeger.py`, line 19), create a CSV file with the following columns:

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

### Select directions to explore

The directions to explore are the number of principal axes to explore in tree space.

### Select exploration strategy

Exploration strategy. Use `deterministic` to use the exploration
strategy based on principal axes. In this strategy, only a single axis
is explored in graph space. If more are desired, select
`DeterministicFullGraph`. Then an equal number of principal axes is
explored in both tree and graph space.

### Select exploration density

This dictates how densely we'll sample the exploration regions in
latent space. More sparse strategies run more quickly.

### Optimization direction

This option dictates whether we should increase or decrease the
desired activity.

### Filter molecules

Check this box if we should only decode promising molecules. Else all
sampled data points in latent space will be decoded (this is slower).

### Chemical similarity cutoff

Tanimoto value for minimum similarity to seed compound. Compounds with
similarities to the seed molecule below this value are discarded.



