# Feature / Training / Inference Pipelines

Architecture based on:

<https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines>

## The system

```
  New candidates ─┐                                                   ┌─▶  Streamlit
  (unlabelled)    │                                                   │      demo
                  │   ┌───────────┐   ┌───────────┐   ┌───────────┐   │
                  ├──▶│  FEATURE  │   │ TRAINING  │   │ INFERENCE │───┘
                  │   │ PIPELINE  │   │ PIPELINE  │   │ PIPELINE  │
  Historical  ────┘   │  [batch]  │   │[on-demand]│   │  [batch]  │
  (labelled)          └─────┬─────┘   └──┬─────▲──┘   └──┬─────▲──┘
                            │            │     │         │     │
                       features &      model   │      predic-  │  features
                         labels          │  features   tions   │  & model
                            ▼            ▼     │         ▼     │
 ┌────────────────────────────────────────────────────────────────────────────┐
 │                                                                            │
 │   FEATURE STORE                          MODEL REGISTRY                    │
 │   data/04_feature/                       src/model/                        │
 │     admisiones_features.parquet            modelo-seleccion-*.joblib       │
 │     candidatos_sin_etiqueta.parquet        metricas_entrenamiento.json     │
 │                                            predicciones.csv                │
 └────────────────────────────────────────────────────────────────────────────┘
```

Every stage reads from the store and writes back to it. No stage hands data to the next
one directly, which is what lets each of them run on its own schedule.

## What each pipeline does

### Feature pipeline — `pipelines/feature_pipeline/`

Two doors into the store, one per kind of record:

| Door | Input | Output |
| --- | --- | --- |
| historical | `data/01_raw/Admission_Predict.csv`, decided applicants | `admisiones_features.parquet` |
| candidates | any CSV of undecided applicants, via `--candidates-path` | `candidatos_sin_etiqueta.parquet` |

Both apply the same **model-independent** typing and the same validation contract, minus
the two rules that need a label: there is no admission chance to bound, and records cannot
contradict each other on an answer nobody has given yet.

Only the candidate door reads the research flag however it was written — `yes`, `sí`, `1`,
`true` — and refuses what it cannot read. The historical source is a file this project
controls and its flag is already numeric, so that door demands `0` or `1`. Candidate
batches arrive from outside, written by whoever filled them in.

They differ in one invariant, on purpose. The historical door deduplicates, so one record
never counts twice during training. The candidate door does not: two applicants with
identical scores are two people, and each needs their own prediction.

```bash
python src/pipelines/feature_pipeline/feature_pipeline.py
python src/pipelines/feature_pipeline/feature_pipeline.py --candidates-path new.csv
```

### Training pipeline — `pipelines/training_pipeline/`

Reads the labelled table, holds out a test set stratified by quantiles of the target,
checks that split for leakage and drift, fits the model, cross-validates it and diagnoses
its fit before storing anything.

This is where the **model-dependent** transformations live. Imputation, scaling and
encoding are parameterized by the training data, so they are fitted here, on the training
split alone, and travel inside the serialized artifact.

```bash
python src/pipelines/training_pipeline/train_pipeline.py
```

### Inference pipeline — `pipelines/inference_pipeline/`

Reads the undecided candidates and the stored model, scores them, and writes the
predictions back to the store.

It prepares nothing and validates nothing, and that is the point of the architecture. The
model-independent transformations already happened at the feature store, and the
model-dependent ones come inside the artifact with the values they learned. The only thing
checked here is that every column the model was fitted on is present in the stored batch,
which is then handed over in the model's own order. Nothing about the values themselves is
re-examined: the feature store already did that, and doing it twice would give those rules
two owners.

```bash
python src/pipelines/inference_pipeline/inference_pipeline.py
```

## Where each transformation belongs

Following the ML transformation taxonomy:

| Kind | Lives in | Examples |
| --- | --- | --- |
| **Model-independent** | feature pipeline | typing, deduplication, reading the research flag |
| **Model-dependent** | inside the `.joblib` | imputation, scaling, encoding |
| **On-demand** | inference / Streamlit | clipping to a valid probability, extrapolation warnings |

Putting a model-dependent transformation in the feature pipeline would fit it on the whole
dataset, before the split: that is data leakage. Keeping it inside the artifact is also
what spares the inference pipeline from reimplementing anything.

## Where the system refuses to continue

```
feature pipeline    invalid data          →  no feature table is written
training pipeline   leakage in the split  →  no model is written
inference pipeline  store ≠ model         →  no predictions are written
```

No stage writes its artifact unless it can guarantee what that artifact promises. A file
that exists is a file that can be trusted.

## Folder structure

The code lives under `src/`:

- `src/`
    - `pipelines/`
        - `feature_pipeline/` — raw records into reusable features
        - `training_pipeline/` — features and labels into a model
        - `inference_pipeline/` — features and a model into predictions
    - `model/` — the model registry: trained artifact, metrics, predictions
    - `data/`, `inference/` — reserved by the project template, still empty

The data lives at the repository root, following the layered data-engineering convention
described in `data/README.md`:

- `data/`
    - `01_raw/` — the immutable source, never written to
    - `04_feature/` — the feature store both other pipelines read from

Those two are production inputs and outputs, not scratch space: the pipelines read and
write them on every run.

The proof of concept is `notebooks/`, which explored the problem, and the Streamlit demo
it produced under `notebooks/7-deploy/`. `src/` holds the productionized implementation of
what those notebooks found, and it does not import from them.
