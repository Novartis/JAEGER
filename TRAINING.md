# Training JAEGER models

## Overview

Currently a JAEGER model can be trained on compounds with AC50
measurements in a single assay. 

## Training Data

The training data should be provided in a CSV file with the following four columns:

* `Id`: Structure ID
* `Structure`: SMILES compound representation
* `<AC50 qualifier>`: AC50 Qualifier (e.g., '>', '<', '=')
* `<AC50 measurement>`: Qualified AC50 value

The order of the columns is to be respected.
The column names for the first two columns must be `Id` and `Structure`, respectively.
The column names for the latter two columns can be arbitrary.

An example CSV file with training data is included in `./models/training_data/Novartis_GNF.csv`.


## Set JAEGER_HOME

In `src/jaeger/jaeger.py`, line 19, please adjust the variable
`JAEGER_HOME` to any directory where you want JAEGER to write out the
models. This path is referred as `$JAEGER_HOME` in the following text.

## Training Script

For training a model from scratch, JAEGER includes a training script
at `src/jaeger/train/jaeger_train.py`. It takes the following
parameters:

* `--csv_file`: String with the file path of the CSV file containing structures and AC50s as described above
* `--use_qualified`: Boolean to determine whether molecules with qualified AC50 values should be used (default to True)
* `--assay_id`: String with assay ID which will be used to create a directory where the model will be saved
* `--num_threads`: Integer indicating how many threads to use while training (default to 12)
* `--weight_decay`: Float with weight decay coefficient used during training (default to 0.0)

**N.B.:** The model will automatically convert micromolar concentration values to
  pAC50s (negative base 10 logarithm of molar concentration).
  
**N.B.:** The model filters out all molecules with more than 50 atoms.
 

The training process is split into two phases: model pre-training and
 model fine-tuning. In pre-training, the Gaussian prior on the
latent representations is disabled. During fine-tuning the Gaussian
prior is turned back on. Pre-training and fine-tuning are each run
over 36 epochs. The script will run both phases consecutively. 

The script will create a directory (including parents) at
`$JAEGER_HOME/assays/<assay_id>/jtvae/jtvae-h-420-l-56-d-7/infer`
where it will store training loss values and model checkpoints,
including the final checkpoint. Model checkpoints computed during
pre-training are saved at files starting with
`model-pre.iter<epoch_number>`. Model checkpoints computed during
fine-tuning are saved at files starting with
`model-ref.iter<epoch_number>`.

The script will use the GPU assigned through the
`$CUDA_VISIBLE_DEVICES` environment variable for training.

As an example with the demo data (please adjust the path to the CSV file accordingly):

```sh
python jaeger_train.py --csv_file /path/to/models/training_data/Novartis_GNF.csv  --assay_id Novartis_GNF --num_threads 24 --use_qualified True
```

On the demo dataset it will take about 24 hours for the training process to complete.

## Validation of inference model

JAEGER includes a script `src/jaeger/train/jaeger_validate.py` for simple validation of
the model. For all training molecules, the script will:

* compare the fitted pAC50 predictions against the true pAC50 measurement 
* compute the latent representation for all training data.
	
It takes the following parameters:

* `--csv_file`: String with the file path of the CSV file containing structures and AC50s
* `--use_qualified`: Boolean to determine whether molecules with qualified AC50 values should be used (default to True)
* `--assay_id`: String with assay ID used as model name

The script will search in
`$JAEGER_HOME/assays/<assay_id>/jtvae/jtvae-h-420-l-56-d-7/infer` for
the final model and load it to calculate the predictions and embeddings.

The script will save the pAC50 predictions and embeddings, respectively, onto
* `$JAEGER_HOME/assays/<assay_id>/jtvae/jtvae-h-420-l-56-d-7/infer/predictions_training.csv`
* `$JAEGER_HOME/assays/<assay_id>/jtvae/jtvae-h-420-l-56-d-7/infer/embeddings_training.csv`

As an example:

```sh
python jaeger_validate.py /path/to/models/training_data/Novartis_GNF.csv --assay_id Novartis_GNF
```

## Cross-validation

Finally, JAEGER includes a cross-validation script `src/jaeger/train/jaeger_xval.py`
for checking the performance of the pAC50 predictor built in latent
space.

Cross-validation is performed by a random 90/10 train/test split of
the compounds.  The split is performed three times. The model is
thus trained from scratch three times.

The script takes the following parameters:

* `--csv_file`: String with the file path of the CSV file containing structures and AC50s as described above
* `--use_qualified`: Boolean to determine whether molecules with qualified AC50 values should be used (default to True)
* `--assay_id`: String with assay ID which will be used to create a directory where the model will be saved
* `--num_threads`: Integer indicating how many threads to use while training (default to 12)
* `--weight_decay`: Float with weight decay coefficient used during training (default to 0.0)

That script will create a directory at
`$JAEGER_HOME/assays/<assay_id>/jtvae/jtvae-h-420-l-56-d-7/xval` where
it will store, per cross-validation run, training loss values and
model checkpoints. After doing all cross-validation runs, the script
saves performance scores (mean-squared-error, correlation) per run at
`$JAEGER_HOME/assays/<assay_id>/jtvae/jtvae-h-420-l-56-d-7/xval/scores.csv`. The
predictions for the test data over all runs is also saved at
`$JAEGER_HOME/assays/<assay_id>/jtvae/jtvae-h-420-l-56-d-7/xval/true_vs_predictions.csv`

As an example:

```sh
python jaeger_xval.py --csv_file /path/to/models/training_data/Novartis_GNF.csv  --assay_id Novartis_GNF --num_threads 24 --use_qualified True
```
