# Colab-Free Fall Detection from Frozen Human Pose Estimation

A reproducible, compute-aware research pipeline for binary fall/non-fall video classification using frozen YOLO11n-Pose features, lightweight spatiotemporal modeling, deterministic split policies, persistent caching, validation-only model selection, and external cross-dataset evaluation.

<!--
Source basis: implementation details are derived from the supplied Python source, including the declared experimental objective, default datasets, pose representation, model family, evaluation policy, and measured-runtime safeguards. :contentReference[oaicite:0]{index=0}
-->

## Abstract

This project investigates whether a lightweight classifier can detect human falls from pose-estimation sequences without training a large RGB video model. The implementation extracts COCO 17-keypoint poses with a frozen `YOLO11n-Pose` model, converts each annotated video segment into a fixed-length 32-frame sequence, derives motion features, and feeds those features to compact neural architectures.

The primary model combines two complementary branches:

1. a spatial graph branch based on a normalized 17-joint human-body graph;
2. a temporal convolutional branch operating on flattened per-frame joint features.

The branches are fused by a learned gate, optionally followed by two-head temporal self-attention. Two small baselines, a flattened MLP and a GRU, are available for comparison.

The experiment is designed around a constrained Google Colab Free/T4 workflow. The code uses persistent Google Drive caches, resumable downloads, checkpointed pose extraction, measured runtime accounting, an operational GPU-memory ceiling, and validation-only model selection. The implementation deliberately avoids unsupported claims of state-of-the-art performance and treats MCFD primarily as a view-shift evaluation rather than a subject-independent generalization benchmark.

## Research Objective

The central question is:

> Can a compact, reproducible pose-only pipeline provide scientifically interpretable evidence for fall detection while remaining practical under a strict Colab Free/T4 compute budget?

The project is therefore designed around experimental discipline rather than model scale. The frozen pose estimator acts as a feature extractor; the trainable component focuses on skeleton geometry and motion. This separation reduces the amount of trainable visual computation and makes the downstream experiment easier to reproduce.

The implementation explicitly avoids treating architecture names such as ST-GCN, TCN, or temporal attention as novelty claims. Their role here is to form a compact experimental model whose behavior can be measured under a controlled protocol.

<!-- Architecture reference: :contentReference[oaicite:1]{index=1} -->

## Default Experimental Design

| Component                                      | Implemented choice                                                                          |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Training data                                  | MCFD + GMDCSA24 v2.1                                                                        |
| External test                                  | CAUCAFall v5                                                                                |
| Pose estimator                                 | `YOLO11n-Pose`                                                                              |
| Pose keypoints                                 | COCO 17 joints                                                                              |
| Frames per segment                             | 32 uniformly sampled frames                                                                 |
| Raw feature channels                           | 8                                                                                           |
| Raw feature order                              | `x`, `y`, confidence, `velocity_x`, `velocity_y`, `acceleration_x`, `acceleration_y`, speed |
| Default model                                  | `ProposedFallNet`                                                                           |
| Spatial branch                                 | Graph convolution + temporal convolutions                                                   |
| Temporal branch                                | Linear projection + dilated 1-D temporal convolutions                                       |
| Attention                                      | 2-head temporal self-attention                                                              |
| Baselines                                      | Tiny MLP, Tiny GRU                                                                          |
| Default learning rate                          | `3e-4`                                                                                      |
| Optimizer                                      | AdamW                                                                                       |
| Loss                                           | Binary cross-entropy with logits, unweighted by default                                     |
| Validation threshold grid                      | 0.20 to 0.80 in increments of 0.025                                                         |
| Primary selection metric                       | Validation Accuracy after validation-only threshold calibration                             |
| Runtime target                                 | 3 hours                                                                                     |
| Hard engineering ceiling in runtime controller | 3.83 hours; final verification also records a 4-hour ceiling                                |
| Operational peak allocated VRAM limit          | 13.5 GiB                                                                                    |
| Random seed                                    | `20260903`                                                                                  |

## Scientific Background

### 1. Pose-based fall detection

A pose estimator maps an RGB frame to human detections and body-joint coordinates. Instead of learning directly from pixels, this project uses the estimated skeleton as a compact representation of human motion.

For each frame $t$ and joint $v$, the implementation stores a 2-D joint location and a confidence score. Consecutive frames are then used to estimate velocity and acceleration. This makes the downstream classifier sensitive to dynamic changes such as rapid vertical motion, posture transitions, and changes in body configuration.

### 2. Human-body graph

The 17 COCO joints form a fixed graph whose edges follow the body connectivity encoded in the source. Self-loops are added before degree normalization.

Let $A$ denote the resulting adjacency matrix and let

$$
d_i = \sum_j A_{ij}.
$$

The implemented normalized adjacency is

$$
\hat{A}_{ij} = \frac{A_{ij}}{\sqrt{d_i d_j}}.
$$

This is the element-wise form of symmetric degree normalization used by the implemented graph convolution. The code stores this matrix as a non-trainable PyTorch buffer.

### 3. Temporal motion features

After body-centered normalization, the implementation computes numerical derivatives with respect to the sampled frame times.

For a normalized coordinate sequence $x_t$, the conceptual velocity is

$$
v_t \approx \frac{\partial x_t}{\partial t},
$$

and acceleration is obtained from the temporal derivative of velocity,

$$
a_t \approx \frac{\partial v_t}{\partial t}.
$$

The implementation computes these quantities with `numpy.gradient`, using the sampled timestamps and replacing extremely small time steps with a `1/30` second fallback.

The per-joint feature vector is therefore

$$
f_{t,v} =
\left[
x_{t,v},
y_{t,v},
c_{t,v},
v^x_{t,v},
v^y_{t,v},
a^x_{t,v},
a^y_{t,v},
\|v_{t,v}\|_2
\right].
$$

The resulting tensor has shape `(T, V, C) = (32, 17, 8)`.

### 4. Body-centered normalization

The pose coordinates are not used in the original image coordinate system.

For each frame, the center is selected in the following order:

1. midpoint of the left and right hips, when reliable;
2. midpoint of the shoulders, when the hips are unavailable;
3. mean of all valid joints;
4. zero center when no valid center exists.

The scale is based on shoulder width and, when available, the hip-to-shoulder torso distance. Coordinates are transformed as

$$
\tilde{\mathbf{p}}_{t,v}
=
\frac{\mathbf{p}_{t,v}-\mathbf{c}_t}
{\max(s_t,\epsilon)}.
$$

Joints whose confidence is below the implementation threshold of `0.20` are treated as missing before normalization. Missing coordinates are later converted to zeros after motion features are assembled.

The implementation explicitly avoids treating low-confidence `(0, 0)` coordinates as genuine observations.

<!-- Pose-preprocessing reference: :contentReference[oaicite:2]{index=2} -->

## Datasets and Evaluation Protocol

### MCFD

The Multiple Cameras Fall Dataset is used primarily for cross-view evaluation. The implementation expects a versioned repack and resolves camera streams through a canonical `chute<N>/cam<M>` identity rather than relying on literal whole-path equality.

The published cross-view split encoded in the project is:

| Split      | Manifest rows |
| ---------- | ------------: |
| Train      |           169 |
| Validation |           169 |
| Test       |          1014 |
| Total      |          1352 |

Eight camera IDs are expected. The code labels this protocol as **cross-view / view-shift** and deliberately does not describe it as subject-independent generalization.

### GMDCSA24 v2.1

GMDCSA24 is mapped through subject-aware canonical identities based on subject, `ADL`/`Fall` partition, and filename stem. The implementation expects 160 physical MP4 clips distributed across four subjects.

The split is deterministic:

| Subject   | Role       |
| --------- | ---------- |
| Subject 1 | Train      |
| Subject 2 | Train      |
| Subject 3 | Validation |
| Subject 4 | Test       |

This gives a subject-disjoint held-out test protocol within GMDCSA24.

### CAUCAFall v5

CAUCAFall is reserved for external evaluation. It is never added to training and is not used for threshold tuning or model selection.

The implementation resolves temporal annotations through the OmniFall metadata schema and expects 10 subjects, a single camera ID, approximately 100 logical activity videos, and 258 usable temporal segments in the verified annotation configuration.

### Binary label construction

The project consumes a staged 16-class activity vocabulary and converts it to binary fall detection. The positive class is defined exactly as:

* source label `1` → `fall`
* source label `2` → `fallen`

All other source activities become non-fall examples.

This design intentionally preserves non-fall activities such as walking, sitting, lying, kneeling, squatting, crawling, and jumping as potential hard negatives.

### Data leakage controls

The split boundary is established before sequence sampling and pose extraction.

The project never performs a frame-level random split. Instead:

* GMDCSA24 uses subject-disjoint train/validation/test groups.
* MCFD uses the versioned cross-view split.
* CAUCAFall is isolated as an external test set.
* Test and external pose features are extracted only after final model selection.
* Threshold calibration is performed only on validation predictions.
* The final threshold is then frozen before held-out evaluation.

This separation is essential because overlapping temporal windows or repeated observations of the same subject/event can otherwise create optimistic estimates.

## End-to-End Methodology

The complete pipeline is:

```text
Dataset archives
      |
      v
Verified download + MD5 validation
      |
      v
Temporary safe extraction
      |
      v
OmniFall metadata resolution
      |
      v
Manifest construction
      |
      +------------------------------+
      |                              |
      v                              v
Train / validation            Held-out / external
segments                      segments
      |                              |
      v                              |
Uniform 32-frame sampling            |
      |                              |
      v                              |
Frozen YOLO11n-Pose                  |
      |                              |
      v                              |
Primary-person tracking heuristic    |
      |                              |
      v                              |
Body-centered pose normalization     |
      |                              |
      v                              |
Velocity / acceleration / speed      |
      |                              |
      v                              |
Persistent compressed NPZ cache      |
      |                              |
      v                              |
Train-only standardization            |
      |                              |
      v                              |
ProposedFallNet / baselines          |
      |                              |
      v                              |
Validation-only threshold selection  |
      |                              |
      v                              |
Final model freeze ------------------+
      |
      +--------------------------+
      |                          |
      v                          v
MCFD cross-view test       GMDCSA24 subject-held-out
      |                          |
      +------------+-------------+
                   |
                   v
            CAUCAFall external test
                   |
                   v
          Metrics + hard negatives
                   |
                   v
        Reproducibility metadata
```

## Pose Extraction

### Temporal sampling

For an annotated interval $[t_{\mathrm{start}}, t_{\mathrm{end}}]$, the implementation converts the interval to frame indices and samples exactly 32 target positions using a uniform `linspace` strategy.

If an interval contains frames $i_{\mathrm{start}}$ through $i_{\mathrm{end}}$, the target indices are obtained conceptually as

$$
i_k =
\operatorname{round}
\left(
i_{\mathrm{start}}
+
\frac{k}{T-1}
(i_{\mathrm{end}}-i_{\mathrm{start}})
\right),
\qquad
k=0,\ldots,T-1,
$$

with $T=32$.

The reader includes a sequential fallback for video codecs where direct frame seeking is unreliable.

### Primary-person selection

For each frame, the pose estimator may produce multiple detections. The project chooses one person using a deterministic score combining:

* detector confidence;
* mean keypoint confidence;
* normalized bounding-box area;
* temporal center continuity.

The implemented score is

$$
S =
0.45\,s_{\mathrm{box}}
+
0.30\,s_{\mathrm{kpt}}
+
0.15\,s_{\mathrm{area}}
+
0.10\,s_{\mathrm{continuity}}.
$$

Temporal continuity is measured from the distance between the current and previous detection centers, normalized by the image diagonal and clipped to the interval $[0,1]$.

This is a heuristic primary-person association mechanism, not a full multi-object tracking algorithm.

### Frozen YOLO11n-Pose configuration

The source configures pose inference with:

* image size: `640`;
* inference batch: `32` by default;
* FP16 on CUDA;
* detection confidence threshold: `0.15`;
* IoU threshold: `0.70`;
* maximum detections per frame: `10`.

The pose network itself is not fine-tuned by this project.

## Proposed Model

The trainable classifier is `ProposedFallNet`.

### Architecture

For an input tensor

$$
X \in \mathbb{R}^{B \times T \times V \times C},
$$

the network contains:

1. **ST-GCN branch** with hidden width `48`;
2. **TCN branch** with hidden width `48`;
3. **learned fusion gate**;
4. **2-head temporal self-attention**;
5. **mean temporal pooling**;
6. **LayerNorm + dropout + linear binary head**.

### ST-GCN branch

The graph branch first permutes the input to `(B, C, T, V)` and applies graph propagation through the fixed normalized adjacency.

At a high level, a graph-convolution block is

$$
H' =
\operatorname{ReLU}
\left(
\operatorname{BN}
\left(
W_{1\times1}(H\hat{A})
\right)
\right),
$$

where $W_{1\times1}$ is a learnable pointwise channel projection.

The implementation uses two graph-convolution operations interleaved with temporal 2-D convolutions using temporal kernel sizes `5` and `3`. The graph output is then mean-pooled over joints to produce a sequence representation of shape `(B, T, H)`.

### TCN branch

The temporal branch reshapes each frame from `(V, C)` into a vector of length $17C$ and applies a linear projection to the hidden width.

Two 1-D convolutions follow:

* kernel `3`, dilation `1`;
* kernel `3`, dilation `2`.

Each convolution is followed by GELU and dropout where configured.

The second convolution therefore provides a larger temporal receptive field without requiring a deep temporal stack.

### Confidence-conditioned fusion gate

The network computes two confidence statistics for every time step:

$$
\mu_t =
\frac{1}{V}
\sum_{v=1}^{V} c_{t,v},
$$

and

$$
\sigma_t =
\operatorname{std}_{v=1}^{V}(c_{t,v}) + 10^{-5}.
$$

The gate receives the concatenated spatial feature, temporal feature, and these two statistics:

$$
g_t =
\operatorname{sigmoid}
\left(
\operatorname{MLP}
\left[
s_t;\,
\tau_t;\,
\mu_t;\,
\sigma_t
\right]
\right).
$$

The two branches are then fused as

$$
h_t =
g_t \odot s_t
+
(1-g_t) \odot \tau_t.
$$

### Important implementation nuance

Although the architecture contains explicit confidence statistics, the default training mask is

```text
[1, 1, 0, 1, 1, 1, 1, 0]
```

which corresponds to:

```text
[x, y, confidence, velocity_x, velocity_y, acceleration_x, acceleration_y, speed]
```

Therefore, the default `E_full_proposed` experiment disables the confidence and speed channels before they reach the model. The gate remains structurally present, but in the default masked run its confidence inputs are effectively zeroed.

Accordingly, the scientifically precise description of the default experiment is:

> a lightweight dual-branch pose-motion classifier with a structurally confidence-conditioned fusion mechanism, evaluated with a motion-focused default input mask.

It should not be described as a fully confidence-driven input model unless the mask is changed and that variant is actually evaluated.

### Temporal attention

When enabled, the fused sequence is passed through PyTorch multi-head self-attention with:

* embedding dimension: `48`;
* number of heads: `2`;
* dropout: `0.1`;
* residual connection;
* LayerNorm.

The implementation requests `need_weights=False`, so attention matrices are not stored as an output artifact.

### Classification head

The temporally pooled representation is passed through:

```text
LayerNorm
   -> Dropout(0.2)
   -> Linear(hidden, 1)
```

The final scalar is a logit, not a probability. Evaluation converts it to a probability with the sigmoid function.

## Baselines and Ablations

The source provides two compact baselines.

### Tiny MLP

The MLP flattens the entire `(32, 17, C)` tensor and maps it through:

```text
Flatten
-> Linear(32*17*C, 64)
-> GELU
-> Dropout(0.2)
-> Linear(64, 1)
```

### Tiny GRU

The GRU baseline first maps each frame's `17*C` features to a 48-dimensional vector and then processes the sequence with a single GRU. The classifier uses the final temporal state.

### Channel ablations

When the optional ablation stage is enabled, the implementation evaluates:

| Variant                                       | Active channels                                                          |
| --------------------------------------------- | ------------------------------------------------------------------------ |
| `A_position_only`                             | x, y                                                                     |
| `B_position_velocity`                         | x, y, velocity_x, velocity_y                                             |
| `D_position_velocity_acceleration_confidence` | x, y, confidence, velocity_x, velocity_y, acceleration_x, acceleration_y |
| `E_full_proposed`                             | x, y, velocity_x, velocity_y, acceleration_x, acceleration_y             |

The source also contains structural ablations that remove ST-GCN or temporal attention while keeping the remaining lightweight architecture.

Optional experiments may be skipped by the runtime controller once the target budget is exhausted. A skipped experiment is recorded as a budget decision rather than interpreted as a negative scientific result.

## Training Procedure

### Data augmentation

Training sequences can receive two inexpensive skeleton-level augmentations:

* small Gaussian coordinate noise applied with probability `0.5`;
* temporal sequence reversal applied with probability `0.25`.

The label is unchanged by either operation.

### Standardization

All channels except confidence are standardized using statistics computed from the training split only.

For a standardized channel $c$:

$$
z =
\frac{x-\mu_{\mathrm{train}}}
{\max(\sigma_{\mathrm{train}},10^{-4})}.
$$

The confidence channel is deliberately excluded from z-scoring so its natural `[0, 1]` interpretation is preserved.

### Optimizer and loss

The default optimizer is AdamW with:

```text
learning rate = 3e-4
weight decay  = 1e-4
gradient clip = 1.0
epochs        = 18
early stopping patience = 4
batch size    = 64
```

The default loss is unweighted `BCEWithLogitsLoss`.

The binary cross-entropy objective can be written as

$$
\mathcal{L}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\left[
y_i \log \sigma(z_i)
+
(1-y_i)\log(1-\sigma(z_i))
\right],
$$

where $z_i$ is the model logit, $y_i \in {0,1}$ is the target label, and $\sigma(\cdot)$ is the sigmoid function.

The implementation explicitly avoids combining weighted sampling and positive-class loss weighting in the default configuration.

### Mixed precision and memory control

On CUDA, the training/evaluation paths use FP16 autocasting and gradient scaling. Peak allocated CUDA memory is checked against the project's operational limit of `13.5 GiB`.

This limit is an engineering guardrail, not a hardware specification.

## Model Selection and Threshold Calibration

Model selection is performed exclusively on the validation split.

For a validation probability vector $p$ and a threshold $\theta$, the predicted label is

$$
\hat{y}_i = \mathbf{1}[p_i \ge \theta],
$$

where $\mathbf{1}[\cdot]$ is the indicator function.

The implementation searches a coarse validation-only grid from `0.20` through `0.80` with a step of `0.025`.

Threshold ranking is deterministic:

1. higher validation Accuracy;
2. higher validation F1;
3. lower validation false-positive rate;
4. threshold closer to `0.5`.

After the best operating threshold is found on validation data, it is frozen.

The final model is then selected using a deterministic tolerance rule:

1. identify the highest validation calibrated Accuracy;
2. keep models within `0.002` Accuracy points of that maximum;
3. among those candidates, prefer higher validation F1;
4. then lower validation false-positive rate;
5. finally fewer parameters.

No held-out test or external data are used during this selection stage. The selection and freezing policy is implemented explicitly in the source.

<!-- Model-selection reference: :contentReference[oaicite:3]{index=3} -->

## Evaluation Metrics

The implementation computes:

* Accuracy;
* Precision;
* Recall / Sensitivity;
* F1;
* Specificity;
* False-positive rate;
* False-negative rate;
* ROC-AUC;
* PR-AUC;
* confusion-matrix counts (`TN`, `FP`, `FN`, `TP`);
* evaluation runtime and examples per second.

The false-positive rate is

$$
\mathrm{FPR}
=
\frac{FP}{FP+TN}.
$$

The false-negative rate is

$$
\mathrm{FNR}
=
\frac{FN}{FN+TP}.
$$

F1 is the harmonic mean of precision and recall,

$$
F_1 =
\frac{2PR}{P+R}.
$$

ROC-AUC and PR-AUC are calculated only when both classes are present in the evaluated labels.

Although validation Accuracy is the primary model-selection metric in this implementation, the source explicitly recommends interpreting it together with error rates, hard negatives, and PR-AUC.

<!-- Interpretation reference: :contentReference[oaicite:4]{index=4} -->

## Hard-Negative Analysis

A dedicated analysis groups non-fall examples into recognizable activities and reports:

| Field                             | Meaning                                     |
| --------------------------------- | ------------------------------------------- |
| `number_of_examples`              | Number of non-fall examples in the category |
| `false_alarms`                    | Examples predicted as fall                  |
| `false_alarm_rate`                | False alarms divided by category size       |
| `mean_predicted_fall_probability` | Mean predicted fall probability             |

This analysis is performed on CAUCAFall when available and on the GMDCSA24 subject-held-out test as a secondary diagnostic.

The purpose is to identify clinically or operationally important failure modes that can be hidden by aggregate Accuracy.

## Data Integrity and Reproducibility

The pipeline includes several safeguards that are useful for research-grade execution.

### Versioned dataset metadata

The default source table records version or record identifiers, expected filenames, expected MD5 checksums, source provenance, and licensing notes.

| Dataset   | Implementation source        | Version / record                                  | Expected MD5                       |
| --------- | ---------------------------- | ------------------------------------------------- | ---------------------------------- |
| MCFD      | Zenodo repack                | `17170592`                                        | `b365b5a4b16691fce3fc09ccd554df79` |
| GMDCSA24  | Official Zenodo release      | v2.1, DOI `10.5281/zenodo.13354453`               | `3d36f2c5c1a666b99639e4e9fd843efb` |
| CAUCAFall | Zenodo repack of Mendeley v5 | `17170592` / original DOI `10.17632/7w7fccy7ky.5` | `5a1899fe0022c9c4c0d37ce64ac0c986` |

Dataset provenance should always be cited separately from any convenience repack. The code itself makes this distinction and does not treat a repack as a new license grant.

<!-- Provenance reference: :contentReference[oaicite:5]{index=5} -->

### Resumable downloads

Large archives are downloaded in chunks with:

* real byte-based progress reporting;
* HTTP `Range` resume support;
* detection of servers that ignore resume requests;
* partial-file preservation;
* MD5 validation after completion;
* verified-cache reuse.

### Safe archive extraction

ZIP members are checked with path resolution before extraction to prevent path traversal / Zip Slip behavior.

### Persistent pose cache

Every extracted sequence is written as a compressed `.npz` file.

A cache is considered valid only when:

* the file contains the expected feature tensor;
* shape is exactly `(32, 17, 8)`;
* the stored label matches the manifest;
* the cache version matches the current pose configuration.

The cache is written through a temporary path and atomically renamed after successful creation.

### Experiment signatures

Training checkpoints contain an experiment signature that incorporates the model name, random seed, configuration, channel mask, cache version, and train/validation data signatures.

A stale or incompatible checkpoint is therefore not silently reused.

### Runtime accounting

The persistent runtime controller tracks cumulative execution time across sessions. It distinguishes:

* a 3-hour target;
* a hard engineering limit near 4 hours;
* essential stages;
* optional stages.

Optional work can be skipped after the target is exceeded, while essential work is allowed to continue until the hard guard is reached.

The project records stage timings and budget decisions so that a later run can be audited rather than relying on informal wall-clock estimates.

## Google Colab and Hardware Assumptions

The notebook is explicitly designed around Google Colab Free and a T4-class CUDA environment.

Important operational characteristics include:

* Google Drive persistence under `/content/drive/MyDrive/fall_detection_project`;
* temporary working storage under `/content/fall_detection_work`;
* FP16 pose inference on CUDA;
* default pose batch size `32`;
* model training batch size `64`;
* `2` data-loader workers;
* peak allocated VRAM guard at `13.5 GiB`;
* runtime target of `180` minutes;
* hard runtime control before expensive stages.

The code does not claim that every Colab session will satisfy a fixed wall-clock runtime. Measured runtime is written to metadata, and the project guardrails distinguish estimates from observations.

## Installation

### Recommended environment: Google Colab

Open the notebook in Google Colab and run it from the first cell downward.

The source intentionally avoids force-reinstalling NumPy, pandas, PyTorch, torchvision, OpenCV, and other core Colab packages. Ultralytics is installed or repaired only when needed, and the implementation targets:

```text
ultralytics==8.3.176
huggingface_hub>=0.24,<1
```

The notebook imports the remaining scientific stack from the active Colab environment.

The source includes the required imports for:

```text
numpy
pandas
matplotlib
tqdm
opencv-python
requests
psutil
torch
scikit-learn
ultralytics
huggingface_hub
```

A local environment can be used only after providing equivalents for the Colab-specific Google Drive and download behavior. The supplied Python file is a generated notebook-oriented source file and is not a drop-in replacement for a standalone local application.

## Usage

### 1. Open the notebook

Use the supplied `.ipynb` file as the primary execution interface.

### 2. Connect a CUDA runtime

A CUDA-enabled runtime is strongly preferred because the project was engineered around GPU-based pose inference and lightweight training.

### 3. Run cells from top to bottom

The notebook initializes the environment, mounts Google Drive, creates the persistent directory structure, downloads and verifies datasets, constructs manifests, estimates the pose workload, extracts cached pose sequences, standardizes features, trains models, freezes the final model, and evaluates the held-out datasets.

### 4. Inspect the generated artifacts

The persistent root is:

```text
/content/drive/MyDrive/fall_detection_project/
```

with the following structure:

```text
fall_detection_project/
├── datasets_zips/
├── pose_cache/
├── metadata/
├── results/
├── models/
├── logs/
└── fall_detection_results_bundle.zip
```

The pipeline is designed so a later Colab session can reuse verified archives, pose caches, checkpoints, manifests, and metadata instead of repeating completed work.

## Output Artifacts

The project expects the following core files after a successful run:

```text
results/
├── validation_results.csv
├── cross_domain_results.csv
├── ablation_results.csv
├── all_metrics.csv
├── hard_negative_activity_breakdown.csv
├── validation_comparison.png
├── confusion_matrix_caucafall.png
└── hard_negative_false_alarm.png

metadata/
├── full_manifest.csv
├── pose_manifest.csv
├── test_pose_manifest.csv
├── run_metadata.json
└── final_output_verification.json

fall_detection_results_bundle.zip
```

The two activity-level plots are optional and are generated only when the corresponding data are available.

The final bundle intentionally excludes the raw dataset archives and raw extracted datasets. It contains results, metadata, models, and logs intended for inspection and reproducibility.

The final packaging stage verifies required outputs before marking the run complete.

<!-- Packaging reference: :contentReference[oaicite:6]{index=6} -->

## Results

### What is and is not reported here

No numerical performance result is fabricated in this README.

The supplied source code contains the machinery for producing validation, cross-view, subject-held-out, external, ablation, and hard-negative results, but it does not provide a completed set of run metrics that can be independently quoted as a scientific result from the source alone.

Accordingly, this repository description should not claim a specific Accuracy, F1, ROC-AUC, PR-AUC, runtime, or VRAM measurement unless the corresponding generated artifact is inspected.

The authoritative runtime and experiment metadata are written to:

```text
metadata/run_metadata.json
```

The authoritative tabular outputs are written to:

```text
results/validation_results.csv
results/cross_domain_results.csv
results/ablation_results.csv
results/all_metrics.csv
results/hard_negative_activity_breakdown.csv
```

## Limitations

### Dataset diversity

The MCFD evaluation is informative for view shift but is not sufficient for a strong subject-independent generalization claim. The project therefore labels it as cross-view evaluation.

### Frozen pose quality

The classifier is downstream of a frozen pose estimator. Errors in person detection, keypoint localization, occlusion handling, or pose association can propagate into the final classifier.

### Primary-person heuristic

The primary-person selection method is a deterministic scoring heuristic. It is not equivalent to a dedicated multi-object tracker, identity-preserving tracker, or full temporal data-association system.

### Annotation dependence

The experimental unit is a temporal segment derived from dataset metadata. The validity of the results therefore depends on the quality and semantics of the upstream annotations.

### Coarse threshold calibration

The decision threshold is selected from a discrete grid with step size `0.025`. This is deliberate for simplicity and budget control, but it is not a continuous optimization of the operating point.

### Accuracy-centered selection

The primary model-selection metric is calibrated validation Accuracy. In imbalanced or deployment-sensitive settings, other objectives such as recall at a fixed false-positive rate, balanced accuracy, cost-sensitive utility, or calibrated PR-AUC may be more appropriate.

### No uncertainty quantification

The current pipeline does not estimate confidence intervals, bootstrap uncertainty, calibration curves, or statistical significance across repeated independent runs.

### Default confidence-mask inconsistency

The architecture exposes a confidence-conditioned fusion mechanism, but the default `E_full_proposed` channel mask zeros the confidence channel. This reduces the practical influence of the confidence signal in the default experiment and should be resolved or explicitly justified before making strong claims about confidence-aware modeling.

### Colab dependence

The project is engineered for a constrained interactive environment rather than production deployment. Disk persistence, GPU availability, codec behavior, network bandwidth, and Colab session limits can all affect execution.

## Future Work

A scientifically useful next stage would focus on stronger validation rather than simply increasing model size.

### Cross-subject and cross-dataset generalization

Expand subject diversity and evaluate repeated subject-disjoint splits where the datasets permit it. Preserve CAUCAFall as a completely external dataset and add additional out-of-distribution datasets only when they answer a clearly defined research question.

### Confidence-aware modeling

Either keep the confidence channel active in the default model or redesign the fusion block so that uncertainty enters explicitly and measurably. Then perform an ablation that isolates the value of confidence from the value of motion derivatives.

### Calibration and operating-point analysis

Add reliability diagrams, expected calibration error, threshold-free curves, and deployment-oriented operating points such as recall at a target false-positive rate.

### Statistical robustness

Repeat the full training procedure across multiple seeds and report mean, standard deviation, and confidence intervals rather than a single run.

### Better temporal modeling

Investigate temporal attention variants, causal temporal convolutions, multi-scale temporal receptive fields, or compact transformer alternatives while preserving the same split and evaluation discipline.

### Person association

Replace the heuristic primary-person chooser with a dedicated pose tracking or identity association mechanism for crowded scenes and multi-person videos.

### Deployment analysis

Measure latency, memory, and throughput for an end-to-end inference pipeline rather than only the downstream pose-sequence classifier.



