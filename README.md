# Burgers FNO

Predict the solution of the periodic viscous Burgers equation at **t=1**
from its initial profile at **t=0**:

\[
u_t + u\,u_x = 0.01\,u_{xx}, \qquad x \in [0,1).
\]


## Run everything

Use Python 3.10 or newer. Extract the files into a folder, open a terminal
there, and run:

```bash
python run_comparison.py
```

It prepares data if needed, trains both models for 128 epochs, and generates
three graphs in `comparison_results/`. On Windows it opens that folder.
If matching comparison weights already exist, it reuses them.

The dataset has 640 examples: 512 training, 64 validation, and 64 test.
Inputs and targets have shape `(batch, spatial_points, 1)` on a 128-point grid.
Reference targets are computed on a 512-point grid.

## Run the original FNO separately

```bash
python -m pip install -r requirements.txt
python prepare_burgers_data.py
python train_burgers.py --epochs 128
python predict_burgers.py
```

To predict from your own initial values:

```bash
python predict_burgers.py --input initial.npy --output my_prediction
```

The input must be real, finite, and unnormalized, with shape `(128,)`,
`(128,1)`, or `(B,128,1)`. Sample the periodic interval without repeating x=1.

## Generated outputs

These folders are created when the scripts run:

| Folder | Contents |
| --- | --- |
| `data/` | Paired initial/final profiles and dataset metadata. |
| `results/` | FNO weights, training history, metrics, and a sample prediction. |
| `predictions/` | Predicted values, CSV output, and a prediction plot. |
| `comparison_results/` | Comparison weights, measured losses and timings, metrics, and the three figures. |

Both comparison models have 19,873 real scalar weights, counting each complex
weight twice. Both use the same initial weights, data splits, batch order,
optimizer, and learning-rate schedule. Their best weights are selected using
validation error before evaluating the test set.

The loss comparison uses log-log axes for training MSE, validation MSE, and
validation mean relative L2. MSE is still the training objective; neither the
training data nor the measured losses are transformed. The original FNO
training graph also shows MSE on log-log axes.

Two FNO trends `C * epoch^(-gamma)` are fitted separately to the early and late
halves of epochs 10 onward. For 128 epochs, the windows are 10-69 and 70-128.
Each legend reports its fitted exponent and log-space R2. These are descriptive
fits, not two proven laws; a flat linear baseline is not labeled as a power law.
The fitted coefficients and windows are saved in `comparison.json` and a table
is saved in `SUMMARY.md`. Short runs need at least three points per fit.

To regenerate the graphs from existing comparison weights without retraining:

```bash
python compare_models.py --skip-training
```

Euler time-step refinement is performed on a fixed spatial grid; spatial
discretization error remains. Timing excludes training and reference-target
generation.

Existing training and prediction outputs are protected from overwriting.
Choose a new `--output` folder for another run of those scripts.
