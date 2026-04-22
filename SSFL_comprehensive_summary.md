# SSFL — Comprehensive Project & Paper Summary

**Paper:** Zhao, R., Wang, Y., Xue, Z., Ohtsuki, T., Adebisi, B., & Gui, G. (2023).
*Semisupervised Federated-Learning-Based Intrusion Detection Method for Internet of Things.*
**IEEE Internet of Things Journal, Vol. 10, No. 10, pp. 8645–8657, 15 May 2023.**
DOI: 10.1109/JIOT.2022.3175918.

**Local companion artefacts in this capstone:**

- `data/` — raw N-BaIoT CSVs (89 files, ~7 GB)
- `prepared_data/` — CNN-ready SSFL dataset (flat + 2D tensors, 3 scenarios)
- `prepared_data_ml/` — same splits without the 2D tensors (for tabular baselines)
- `Semisupervised_Federated-Learning-Based_Intrusion_Detection_Method_for_Internet_of_Things.txt` — full paper text
- `implementation_plan.md`, `dataset_guide.md`, `DATASET_README.md`, `ssfl_concept_guide.md`, `opus_report.md` — supporting documentation

---

## 1. Problem Statement and Motivation

The paper tackles **intrusion detection in IoT networks**, where three practical constraints collide:

1. **Security / privacy.** Centralised deep learning (CDL) requires traffic to be uploaded to a server, which exposes user-private raw data. Traditional federated learning (FL) avoids uploading raw data, but **gradient leakage attacks** (Zhu & Han, "Deep Leakage from Gradients") show that uploaded model parameters can be inverted to reconstruct the raw training samples. So vanilla FL is not truly private either.
2. **Accuracy under non-IID data.** Real IoT clients hold very heterogeneous traffic. Each client typically represents one device and only sees a subset of the 11 traffic classes. Both FedAvg-style FL and distillation-based FL (FD) degrade sharply when the data is non-IID, because a client's locally-trained model cannot produce reliable predictions for unfamiliar classes.
3. **Communication efficiency.** IoT deployments involve large numbers of resource-constrained edge devices. Uploading full model parameters every round is prohibitively expensive. Earlier works used gradient sparsification, top-k compression, or binarised nets, but communication still scales with model size.

The authors propose **SSFL** (Semi-Supervised Federated Learning) — an FL framework that exchanges **hard label predictions on a shared unlabeled "open" dataset** instead of model parameters. It uses knowledge distillation to transfer information between clients, a CNN backbone for feature extraction, and a per-client **discriminator** network that flags unfamiliar inputs, so noisy predictions are filtered out before aggregation. A **majority-voting** mechanism on the server turns per-client hard labels into global labels that each client then distills from.

**Design goals, restated from the paper:**

- **Security:** private data must not be reconstructible → never upload parameters or gradients.
- **Accuracy under non-IID:** each client's contribution must be filtered so its weakness on unseen classes does not poison the global model.
- **Efficiency:** communication cost must not scale with model size, so IoT nodes can participate.

---

## 2. Dataset: N-BaIoT (source data)

### 2.1 Provenance

- **Origin:** N-BaIoT dataset by Meidan et al., *"N-BaIoT: Network-based detection of IoT botnet attacks using deep autoencoders"* (IEEE Pervasive Computing 2018), hosted on the UCI Machine Learning Repository.
- **Purpose:** contains benign and botnet-infected network-traffic statistics collected from **9 commercial IoT devices**.
- **Botnets covered:** **Mirai** and **BASHLITE (Gafgyt)** — the two most-cited IoT botnet families. Each produces multiple attack sub-types (scan, TCP flood, UDP flood, syn flood, ack flood, combo, junk, udpplain).
- **Capture pipeline:** traffic was recorded per device and processed into statistical features by the **Kitsune** framework (Mirsky et al., NDSS 2018).

### 2.2 Devices (9 total)

| Device ID | Device |
|---|---|
| 1 | Danmini Doorbell |
| 2 | Ecobee Thermostat |
| 3 | Ennio Doorbell |
| 4 | Philips B120N10 Baby Monitor |
| 5 | Provision PT-737E Security Camera |
| 6 | Provision PT-838 Security Camera |
| 7 | Samsung SNH-1011N Webcam |
| 8 | SimpleHome XCS7-1002-WHT Security Camera |
| 9 | SimpleHome XCS7-1003-WHT Security Camera |

Devices are primarily IP cameras, doorbells, a thermostat, and a baby monitor — a realistic IoT cross-section.

### 2.3 Traffic classes (11 total)

| Label | Class | Type | Coverage |
|---|---|---|---|
| 0 | `benign` | normal traffic | all 9 devices |
| 1 | `gafgyt.combo` | attack (combo) | all 9 devices |
| 2 | `gafgyt.junk` | attack | all 9 devices |
| 3 | `gafgyt.scan` | attack | all 9 devices |
| 4 | `gafgyt.tcp` | attack (TCP flood) | all 9 devices |
| 5 | `gafgyt.udp` | attack (UDP flood) | all 9 devices |
| 6 | `mirai.ack` | attack (ACK flood) | 7 devices (not 3, 7) |
| 7 | `mirai.scan` | attack | 7 devices (not 3, 7) |
| 8 | `mirai.syn` | attack (SYN flood) | 7 devices (not 3, 7) |
| 9 | `mirai.udp` | attack (UDP flood) | 7 devices (not 3, 7) |
| 10 | `mirai.udpplain` | attack (plain UDP) | 7 devices (not 3, 7) |

Devices 3 (Ennio Doorbell) and 7 (Samsung Webcam) were never infected with Mirai, so they only contain the 6 "benign + gafgyt.*" classes. This is a real source of non-IID-ness — two of the nine devices are missing 5 attack categories entirely.

Storage convention on disk: one CSV per `(device_id, traffic_class)` pair (`{device}.{class}.csv`) — 89 files total, ~7.06 million raw traffic records, ~7 GB.

### 2.4 Feature structure — 115 features per sample

Each record is a vector of **115 statistical features** computed by Kitsune's feature extractor. The features are organised as **5 time-decay windows × 23 statistics**:

- **4 stream aggregation types**:
  - `H`: recent traffic from this packet's source host
  - `HH`: source host → destination host
  - `HH_jit`: packet-jitter for the source → destination stream
  - `HpHp`: source host+port → destination host+port
- **5 time-decay windows**: approximately **100 ms, 500 ms, 1.5 s, 10 s, and 1 min** (decay factors λ ∈ {5, 3, 1, 0.1, 0.01} on an exponentially-weighted moving aggregator).
- **Per-window statistics**: `weight`, `mean`, `variance`, plus pairwise quantities (`radius`, `magnitude`, `covariance`, `pcc`) for the host-to-host streams. Stacked across the four stream types, this yields 23 statistics per window (115 / 5 = 23).

The features are designed to be computable in a streaming fashion on resource-constrained IoT devices, i.e. real-time traffic analysis is feasible.

---

## 3. mini-N-BaIoT — how the raw data is compressed into a workable subset

The authors argue that training on the full N-BaIoT is computationally unnecessary, so they construct a balanced subset they call **mini-N-BaIoT**.

### 3.1 Construction rule

For every `(device_d_i, traffic_category_l)` pair that actually exists in the raw dataset, they randomly sample **1000 records**. Because devices 3 and 7 only have 6 classes and the other 7 devices have 11:

- 7 devices × 11 classes × 1000 = **77,000** samples
- 2 devices × 6 classes × 1000 = **12,000** samples
- **Total mini-N-BaIoT = 89,000 samples**

This is the number used everywhere in the paper and reproduced exactly in the local `prepared_data/` directory.

### 3.2 Random sampling is seeded (reproducibility)

- Mini-N-BaIoT sampling and the private/open/test partition use `seed = 42`.
- Each scenario uses `seed = 42 + scenario_id` when assigning clients, so the three scenarios are independent but deterministic.

### 3.3 Split ratios

After constructing mini-N-BaIoT, the 89,000 samples are partitioned **stratified by device and class** (so every device and every class appears proportionally in every split) into three disjoint sets:

| Split | Proportion | Samples | Purpose |
|---|---|---|---|
| **Private (Dᵖ)** | 70% | **62,300** | distributed to FL clients for local supervised training |
| **Open (Dᵒ)** | 10% | **8,900** | shared unlabeled corpus used for distillation across all clients |
| **Test (Dᵗᵉˢᵗ)** | 20% | **17,800** | held out for global evaluation |

The three sets are **pairwise disjoint** by construction (no sample leakage). The open set's labels exist in the file (for evaluation sanity) but are **deliberately never used during SSFL training**.

### 3.4 Preprocessing pipeline (exactly as implemented in the codebase)

The paper describes three preprocessing steps; the code follows them to the letter.

**Step A — Data partition.** The 89,000 samples are split 70/10/20 (private/open/test) with stratification over (device, class).

**Step B — Min–max normalisation.** All 115 features are scaled into `[0, 1]`. Crucially, the min/max statistics are computed **only on the private set** (to avoid test-set leakage) and then applied to all three splits. These are stored on disk as `feat_min.npy` and `feat_max.npy` (shape `(115,)`, dtype float32).

**Step C — 2-D reshaping ("dimensionalization").** Each 115-d feature vector `xᵢ` is reshaped into a **23×5 matrix**:

```
        window0  window1  window2  window3  window4
stat0  [ x_0      x_23     x_46     x_69     x_92  ]
stat1  [ x_1      x_24     x_47     x_70     x_93  ]
 ...
stat22 [ x_22     x_45     x_68     x_91     x_114 ]
```

i.e. the 23 per-window statistics occupy the row axis, and the 5 time windows occupy the column axis. The samples are stored as 4-D tensors of shape `(N, 1, 23, 5)` (explicit channel axis for PyTorch's `Conv2d`). This is the canonical spatial layout that lets a 2-D CNN convolve across **both** adjacent statistics (rows) **and** adjacent time scales (columns) — this interaction is exactly what the paper claims motivates using a CNN.

Concretely (Eq. 19 in the paper), the reshape rule is `x[r, c] = x_flat[r + 23·c]`.

### 3.5 Artefacts produced on disk

Under `prepared_data/` (and its tabular twin `prepared_data_ml/` without the 2-D tensors):

```
prepared_data/
├── label_map.json          # {class_name: int_label, ...} — 11 entries
├── feat_min.npy            # (115,) float32
├── feat_max.npy            # (115,) float32
├── private/   X.npy (62300,115)  X_2d.npy (62300,1,23,5)  y.npy (62300,)  device_ids.npy
├── open/      X.npy  (8900,115)  X_2d.npy  (8900,1,23,5)  y.npy  (8900,)  device_ids.npy  (y unused at training time)
├── test/      X.npy (17800,115)  X_2d.npy (17800,1,23,5)  y.npy (17800,)  device_ids.npy
├── scenario_1/  summary.json + client_{0..26}/{X,X_2d,y}.npy   # 27 clients
├── scenario_2/  summary.json + client_{0..88}/{X,X_2d,y}.npy   # 89 clients
└── scenario_3/  summary.json + client_{0..88}/{X,X_2d,y}.npy   # 89 clients
```

`device_ids.npy` tracks the origin device of every sample so a FL client can be wired to the correct device's traffic. A per-scenario `summary.json` lists `{client_id, device_id, num_samples, labels_present}` per client, which is also used by the client_app to configure its local training loop.

The two on-disk datasets differ only in whether `X_2d.npy` is present:

| Property | `prepared_data` | `prepared_data_ml` |
|---|---|---|
| Shape of `X_2d.npy` | `(N, 1, 23, 5)` float32 | absent |
| Designed for | CNN (Conv2D) – the SSFL reproduction | Tabular models (sklearn, XGBoost, LR, SVM, MLP) |
| Disk size | ~257 MB (633 files) | ~130 MB (425 files) |
| Sample-level content | Identical to the ml version | Identical to the non-ml version |

Both use the same `seed = 42` and the same deterministic pipeline, so a sample in `scenario_1/client_5/` is literally the same sample between the two datasets — only the CNN-ready tensor is missing in the ml variant.

---

## 4. The three non-IID client scenarios

Each scenario takes the 62,300 private samples (70% split) and hands them out to a fixed population of clients under a **different heterogeneity rule**. In all three, **each client's data comes from exactly one device** — consistent with a realistic "one IoT node = one client" deployment.

Let `D_{d_i}` be device `d_i`'s subset of the private data, and `L_{d_i}` be the number of classes that device has (11 for most, 6 for devices 3 and 7).

### 4.1 Scenario 1 — shard-based with K = 3 per device

- Sort each device's data by label.
- Cut it into `2 · K_{d_i} = 6` equal shards.
- Assign 2 contiguous shards to each of 3 clients per device.
- Total clients: `3 × 9 = 27`.
- Average labels per client: ~4.5 out of 11.
- Average samples per client: `62,300 / 27 ≈ 2,307`.

Each client sees roughly half the label alphabet but fairly uniform sample counts. This is the "easier" non-IID setting and follows the pioneer DS-FL study by Itahara et al. (2021).

### 4.2 Scenario 2 — shard-based with K = L per device

- Same shard cutting strategy, but `K_{d_i} = L_{d_i}`.
- So devices with 11 classes generate 11 clients (22 shards) and devices with 6 classes generate 6 clients (12 shards).
- Total clients: `7·11 + 2·6 = 77 + 12 = 89`.
- Each client sees **only ~2 classes** (maximum label exclusivity).
- Average samples per client: `62,300 / 89 = 700`.

This is much harsher — most classes are completely absent from most clients, which is exactly the regime where FD and DS-FL baselines collapse and SSFL's discriminator shines.

### 4.3 Scenario 3 — Dirichlet(α=0.1) distribution

- 89 clients (same count as Scenario 2, `K_{d_i} = L_{d_i}`).
- For each device, class proportions are drawn from a Dirichlet distribution with concentration `α = 0.1`.
- A small α (0.1) produces sparse, highly-skewed per-client label proportions: most of the probability mass collapses onto 1–2 labels per client, with the rest appearing in small amounts.
- Sample counts per client vary wildly (empirically: min = 9, max = 1974).

This is the most adversarial setting — extreme class imbalance and extreme sample-count imbalance together. All three scenarios are "non-IID" but only Scenario 3 is extreme enough to be genuinely adversarial for distillation methods.

The three scenarios share the same private / open / test splits, so head-to-head numerical comparisons are fair.

---

## 5. The CNN architecture

### 5.1 Why a CNN for packet features?

The authors explicitly justify the choice of a convolutional backbone with two arguments:

1. **Strong feature-extraction ability** — CNNs can learn hierarchical abstractions from the input statistics, more expressive than hand-engineered tabular features.
2. **Spatial locality aligns with the time-window layout.** The 23×5 reshape places adjacent per-window statistics next to each other on the row axis and adjacent time scales (100 ms → 500 ms → 1.5 s → 10 s → 1 min) next to each other on the column axis. Convolutions that slide over this matrix therefore exploit **cross-feature** and **cross-window (i.e. cross-time-scale) interactions** simultaneously — a 2-D prior that a plain MLP cannot encode.

Empirically, the paper also shows in Table II and the convergence plots that the CNN beats the MLP and LSTM baselines trained under the same SSFL harness, especially in Scenarios 2 and 3. So the choice is not only a priori reasonable but also empirically justified in the ablations.

### 5.2 Full architecture (from Table I of the paper)

The classifier and discriminator **share the same backbone** and differ only in the output dimension of the final fully-connected layer. This is deliberate — it means they can reuse the same feature extractor and swap heads.

**Input.** `(batch, 1, 23, 5)` — one channel, 23 rows (stats), 5 columns (windows). Note: the paper's Table I also states "the input channels of the first convolutional layer are 23", which is an alternate reading where the 5 windows are the spatial axis and the 23 statistics are channels. Either encoding is functionally equivalent; the reference implementation plan in this repo uses `(N, 1, 23, 5)` with 1 input channel (a normal image-like layout).

**Feature-extraction block (8 conv layers).**

| Layer | Filters | Kernel size | Notes |
|---|---|---|---|
| conv1 | 64 | 3 | ReLU |
| conv2 | 64 | 3 | ReLU |
| conv3 | 64 | 3 | ReLU |
| conv4 | 64 | 3 | ReLU |
| conv5 | 128 | 3 | ReLU |
| conv6 | 128 | 3 | ReLU |
| conv7 | 128 | 3 | ReLU |
| conv8 | 128 | 3 | ReLU |

**Classification head — a 3-layer MLP on the flattened feature map:**

- FC1: hidden units (typically 256)
- FC2: hidden units (typically 128)
- FC3: **11 neurons** in the classifier (one per traffic class) or **2 neurons** in the discriminator (familiar vs. unfamiliar), followed by softmax.

**Key points highlighted by the paper:**

- The final layer of the classifier has **11 neurons**; the discriminator has **2 neurons**. Everything else is identical.
- The paper says the feature map is flattened and then passed through "a multilayer perceptron (MLP) with three fully connected layers" for classification.
- A crucial design statement from the paper: *"our federated training scheme does not aggregate model parameters, which means that it can work even if clients adopt different model structures."* So architectural heterogeneity across clients is explicitly allowed — the only requirement is that each client can emit a hard label on the shared open dataset.

---

## 6. The SSFL algorithm in detail

### 6.1 Actors and data

- **K clients**, each with:
  - a private labeled dataset `D^{k,c} = {(x_i^{k,c}, y_i^{k,c})}` (on-device traffic with labels),
  - a copy of the shared **unlabeled open dataset** `D^o = {x_j^o}` (downloaded once from the server),
  - a **classifier network** `w^{k,c}` (CNN, 11-way output),
  - a **discriminator network** `w^{k,d}` (same CNN backbone, 2-way output).
- **1 central server**, which:
  - stores the open set and distributes it at the start,
  - aggregates per-client hard-label predictions via majority voting,
  - broadcasts the aggregated **global hard labels** `P^s` back to all clients,
  - optionally maintains a global model `w^s` for centralised evaluation.

### 6.2 One training round (T rounds total) — Algorithm 1

For every communication round, each client performs **five stages**:

#### Step 1 — Train the classifier on private labeled data (Eq. 11)

Each client `k` trains `w^{k,c}` on `(X^{k,c}, Y^{k,c})` with cross-entropy and gradient descent:

```
w^{k,c} ← w^{k,c} − γ · ∇ψ( F(X^{k,c}; w^{k,c}), Y^{k,c} )
```

Then each client runs the classifier on every sample in the shared open set to produce a softmax vector `\hat p_j^k = F(x_j^o; w^{k,c})` and a **confidence score** `c_j^{k,o} = max(\hat p_j^k)` (Eq. 12).

#### Step 2 — Train the discriminator (Eqs. 13–14)

The discriminator's job is to learn "does this open-set sample look like something I have seen locally?" Training data is built as follows:

- For every open sample, if its confidence score `c_j^{k,o} < θ_c`, label it `[0, 1]` → **unfamiliar**.
- For every private sample, label it `[1, 0]` → **familiar** (by definition).
- The union of these is `D^{k,d}`.

Then train `w^{k,d}` on `D^{k,d}`. The threshold `θ_c` is **not fixed** — for each client it is set to the **median** of that client's predicted confidence scores on the open set. So θ is client-adaptive. The ablation study confirms the median is better than any fixed cutoff (0.7 / 0.8 / 0.9), especially in the harder scenarios.

#### Step 3 — Filter and upload (Eqs. 15–16)

Each client re-scores every open sample with its *discriminator*:

- `d_j^{k,o} = argmax(F(x_j^o; w^{k,d}))` — 0 if "familiar", 1 if "unfamiliar".
- Combined filter (Eq. 16):
  - if `d_j^{k,o} = 0`: `\hat p_j^k = argmax(F(x_j^o; w^{k,c}))` — the classifier's hard label.
  - if `d_j^{k,o} = 1`: `\hat p_j^k = −1` — an explicit "abstain" marker.

The client uploads only the 1-D vector of integer hard labels (length `N_o = 8900`). If a sample looked unfamiliar, it's uploaded as `−1` so the server can ignore it.

#### Step 4 — Vote and broadcast (Eq. 17) — on the server

For each open sample `j` the server collects all clients' hard labels (skipping `−1` values) into `L` voting sets `V_{j,0}, V_{j,1}, …, V_{j,L-1}` and takes the majority vote:

```
\hat p_j^s = argmax( |V_{j,0}|, |V_{j,1}|, …, |V_{j,L-1}| )
```

The concatenation `P^s = (\hat p_1^s, …, \hat p_{No}^s)` is broadcast to every client. The server itself can also train a global model `w^s` on `(X^o, P^s)` for centralised evaluation.

#### Step 5 — Distillation (Eq. 18) — on the client

Each client uses `(X^o, P^s)` as a labeled dataset and does a distillation pass:

```
w^{k,c} ← w^{k,c} − γ · ∇ψ( F(X^o; w^{k,c}), P^s )
```

This is the step that lets knowledge cross the (non-IID) client boundary — a client whose private data lacks class 6 can still learn class 6 because the open-set samples that are actually class 6 received class-6 votes from clients that do have class 6 data.

These 5 steps repeat for `T` rounds (paper uses T = 200, tracking Top-1 accuracy over rounds).

### 6.3 Why the discriminator + voting + hard-label combination matters

Each design choice maps onto one of the paper's three goals:

- **Discriminator.** Prevents a client from voting on classes it has never seen. This directly addresses the failure mode of vanilla DS-FL under non-IID data, where a client's wrong predictions poison the aggregated teacher signal.
- **Voting (majority, not averaging).** Turns a noisy per-client hard label into a robust global hard label that only exists when a critical mass of clients agree. If only unfamiliar clients vote, the sample is effectively dropped.
- **Hard labels (vs. soft logits).** Cuts communication cost dramatically. A soft label is a length-11 `float64` vector (~704 bits); a hard label is a single `int` (typically 8 bits). The paper's Fig. 6 shows that rounding soft labels to 8/6/4/2 decimals gives essentially the same accuracy but progressively smaller payloads, and hard labels (the limit) give the smallest payload while keeping accuracy.

### 6.4 Side benefits — privacy and model heterogeneity

- **No parameters or gradients ever cross the wire**, so gradient-leakage attacks (Zhu & Han) are inapplicable.
- Each client can **have a different model architecture** — the protocol only requires hard labels on `X^o`. (This is a practical advantage for heterogeneous IoT fleets and is called out explicitly in Section IV-A of the paper.)

---

## 7. Experimental environment and hyperparameters

### 7.1 Hardware & software stack (reported in Section V-C)

| Item | Value |
|---|---|
| Language | Python 3.7 |
| Deep-learning framework | PyTorch 1.9.0 |
| CPU | Intel Core i9-11900K @ 3.50 GHz |
| RAM | 64 GB |
| GPU | NVIDIA GeForce RTX 3090 |

This is a single-workstation setup. The federated training is simulated on one machine (which is the usual practice for FL research at this scale — a real 89-client deployment is impractical to reproduce).

### 7.2 Hyperparameters (Section V-C2)

| Hyperparameter | Value | Notes |
|---|---|---|
| Optimizer | Adam | |
| Learning rate `γ` | 1e-4 | applied in Eqs. 11 and 18 |
| Batch size | 100 | |
| Local epochs per round | 5 | |
| Rounds `T` | 200 (tracked) | table-III convergence analysis |
| Confidence threshold `θ_c` | **median of each client's own confidences** | per-client, not global |
| Distillation temperature `T` | only relevant for DS-FL baseline (used to sharpen soft logits); SSFL itself exchanges hard labels and does not use it | |

The local repository's `conf/config.yaml` mirrors exactly these settings and adds a `scenario` switch (1 / 2 / 3) and a `seed = 42` for reproducibility.

### 7.3 Evaluation metrics

The paper reports **Accuracy, F₁, and Precision**, all computed on the held-out test set of 17,800 samples. Definitions (Eqs. 20–22):

```
Recall     = T_p / (T_p + F_n)
Precision  = T_p / (T_p + F_p)
F_1        = 2 · Precision · Recall / (Precision + Recall)
```

Confusion matrices are also shown (Fig. 3) to inspect per-class behaviour.

### 7.4 Baselines compared against

| Baseline | Description |
|---|---|
| **FL (FedAvg)** | Classic parameter-averaging FL. |
| **FD** | Federated Distillation (Jeong et al., 2018). Clients upload per-class average logits. |
| **DS-FL** | Distillation-based Semi-supervised FL (Itahara et al., 2021). Clients upload soft logits over an open dataset; server averages them. |
| **MLP under SSFL** | Same SSFL pipeline but with an MLP backbone — isolates the CNN's contribution. |
| **LSTM under SSFL** | Same SSFL pipeline but with an LSTM backbone — isolates the CNN's contribution. |
| **SSFL (proposed)** | Full method: CNN backbone + discriminator + voting + hard labels. |

All baselines were evaluated on the same private/open/test splits and the same three scenarios to keep the comparison fair.

---

## 8. Key experimental results

### 8.1 Detection performance (RQ1 — Table II)

SSFL consistently beats every baseline on all three scenarios:

- **FL:** good accuracy but prohibitive communication overhead and gradient-leakage risk.
- **FD:** collapses under non-IID. Roughly 50% accuracy in Scenarios 1 and 3, **only ~22% in Scenario 2** (where each client has only ~2 classes, so per-class average logits carry almost no signal). In fact, the paper notes that FD's best accuracy is obtained by a single client's local model — i.e. federation is actively unhelpful under these conditions.
- **DS-FL:** improves over FD in Scenarios 1 and 2 because the open set acts as a shared vocabulary, but falls apart in Scenario 3's Dirichlet heterogeneity — too many clients produce wrong soft logits, which poisons the aggregated teacher.
- **MLP / LSTM under SSFL:** both converge more slowly than CNN under SSFL, confirming that the CNN backbone is doing real work (it's not only the training harness that drives SSFL's gains).
- **SSFL:** highest Accuracy / F₁ / Precision in all three scenarios. The repo's reproduction targets (from `implementation_plan.md`) are approximately: Scenario 1 ~92% accuracy, Scenario 2 ~89%, Scenario 3 ~85%.

Confusion matrix patterns (Fig. 3): benign traffic is detected with **>99% accuracy in all three scenarios**. The only non-trivial confusion is between `gafgyt.tcp` and `gafgyt.udp` — both are DoS-style floods with very similar statistical signatures, which is a dataset-level difficulty rather than an algorithmic one.

### 8.2 Communication efficiency and convergence (RQ2 — Tables III & IV)

- **Convergence speed.** SSFL reaches high accuracy within the first ~10 rounds and converges around ~150 rounds. FD and DS-FL plateau early and low. FL's Top-1 at 200 rounds is still well below its own eventual maximum, reflecting slow parameter-averaging convergence.
- **Communication cost.** FL's per-round cost scales with the number of model parameters (very large for an 8-layer CNN). FD / DS-FL / SSFL scale with the output dimension of the model, not its depth. SSFL is further reduced by exchanging hard labels (one integer per open sample) instead of soft logit vectors.
- **Net result.** SSFL achieves simultaneously the highest accuracy and the lowest communication overhead.

### 8.3 Ablation study (RQ3 — Figures 4–6)

The authors ablate three components:

- **No voting.** Direct aggregation (averaging per-class votes) makes things worse because a majority-vote mechanism implicitly *requires* a reasonable quality of each client's prediction. Without the discriminator, voting amplifies bad predictions.
- **No discriminator.** Every client predicts on every open sample, including ones it has no expertise in. Performance drops sharply in every scenario. The discriminator is **by far the largest single contributor** to SSFL's advantage over DS-FL.
- **Simple filtering** (a hand-tuned confidence threshold instead of a learned discriminator): partially works in Scenarios 1 and 2, but **fails completely in Scenario 3** because a single static threshold cannot track the per-client confidence distribution when the data is that heterogeneous.
- **Confidence threshold θ_c.** Comparing fixed θ_c ∈ {0.7, 0.8, 0.9} vs. median: the median is always best, and its advantage grows as the client's data gets smaller / more skewed.
- **Soft vs. hard labels.** Rounding soft labels to 8, 6, 4, or 2 decimal places gives essentially identical accuracy but progressively smaller payloads. Hard labels (1 int each) give essentially the same accuracy as full-precision soft labels while being the cheapest to transmit — so the hard-label strategy is "free" for accuracy and enormous for bandwidth.

---

## 9. How the paper maps to the code in this workspace

### 9.1 Dataset-preparation pipeline — *already done*

| Paper step | Code artefact |
|---|---|
| Construct mini-N-BaIoT (1000 samples per `(device, class)`) | `prepare_data_ml.ipynb` + the resulting `prepared_data/` and `prepared_data_ml/` directories |
| 70/10/20 stratified split | `private/`, `open/`, `test/` subdirectories in both |
| Min-max normalisation on private only | `feat_min.npy`, `feat_max.npy` at the top level |
| 23×5 reshape for CNN input | `X_2d.npy` files (shape `(N, 1, 23, 5)`, dtype float32) |
| Three non-IID scenarios | `scenario_1/` (27 clients), `scenario_2/` and `scenario_3/` (89 clients each) |
| Client metadata | `scenario_*/summary.json` — `{client_id, device_id, num_samples, labels_present}` per client |

### 9.2 Flower training pipeline — *to be built* (the capstone's own scope)

| Paper step | Planned code location | Flower role |
|---|---|---|
| Step 1 — train classifier on private labeled data | `src/train.py::train_classifier` | called inside `ClientApp.fit()` |
| Step 1 cont. — confidence scores on open set | `src/train.py::compute_confidence_scores` | inside `ClientApp.fit()` |
| Step 2 — build `D^{k,d}` and train discriminator | `src/train.py::build_discriminator_dataset` + `train_discriminator` | inside `ClientApp.fit()` |
| Step 3 — filter + upload hard labels | `src/train.py::filter_and_predict` → `ArrayRecord` | client → server |
| Step 4 — vote & broadcast global hard labels | `SSFLStrategy.aggregate_fit()` | custom Flower strategy (cannot use FedAvg) |
| Step 5 — distillation on `(X^o, P^s)` | `src/train.py::distillation_train` | called inside `ClientApp.fit()` (next round) |
| CNN architecture (classifier + discriminator) | `src/model.py` — `CNNBackbone`, `Classifier`, `Discriminator` | shared backbone |
| Data loaders | `src/data.py` — `load_client_data`, `load_open_data`, `load_test_data` | |
| Hyperparameters from Section V-C | `conf/config.yaml` | lr=1e-4, batch=100, local_epochs=5, rounds=200, θ=median, seed=42 |
| End-to-end entry point | `run_simulation.py` | Flower simulation engine; runs K virtual clients on one GPU |

### 9.3 Reproducibility guarantees

The same `seed = 42` for sampling/splitting and `seed = 42 + scenario_id` for client assignment ensures that re-running `prepare_dataset.py` gives bitwise identical outputs. Client 5 in Scenario 1 is always the same 2,568 rows, from the same device, with the same label set, across runs and across the `prepared_data` / `prepared_data_ml` variants.

---

## 10. Cheat-sheet summary

- **Task:** 11-class IoT traffic classification (1 benign + 10 botnet attack sub-types).
- **Raw dataset:** N-BaIoT, 9 IoT devices, ~7 M samples, 115 statistical features per sample (5 time windows × 23 stats).
- **Working subset:** mini-N-BaIoT — 89,000 balanced samples (1000 per `(device, class)`).
- **Splits:** 70% private (62,300) / 10% open, unlabeled (8,900) / 20% test (17,800) — disjoint, stratified.
- **Preprocessing:** min-max normalisation fit on private only; 23×5 reshape to make the features 2-D.
- **Three non-IID scenarios:** (1) 27 clients shard-based K=3, (2) 89 clients shard-based K=L, (3) 89 clients Dirichlet(α=0.1).
- **Model:** shared CNN backbone — 8 conv layers (64/64/64/64/128/128/128/128 filters, kernel=3) → flatten → 3-layer MLP. Classifier head has 11 outputs; discriminator head has 2 outputs.
- **Why CNN:** captures cross-feature + cross-time-scale interactions implicit in the 23×5 layout; empirically beats MLP/LSTM under the same SSFL harness.
- **Algorithm:** 5-step round — train classifier → train discriminator → filter & upload hard labels → server majority-vote → distill on global hard labels — repeated for T=200 rounds.
- **What's on the wire:** one `int` per open-set sample per client (plus `-1` abstain markers). No gradients, no parameters — no gradient-leakage risk.
- **Hyperparameters:** Adam, lr=1e-4, batch=100, 5 local epochs / round, θ = per-client median confidence, seed=42.
- **Environment:** Python 3.7, PyTorch 1.9.0, i9-11900K, 64 GB RAM, RTX 3090.
- **Metrics:** Accuracy, F₁, Precision, Recall; Top-1 curves over rounds; confusion matrices; communication-overhead bytes-per-round.
- **Key results:** SSFL beats FL, FD, DS-FL on all three scenarios; communication cost is orders of magnitude lower than FL; benign detection ≥99% everywhere; main residual confusion is gafgyt.tcp ↔ gafgyt.udp (both are DoS floods with similar signatures).
- **Ablation headline:** the discriminator is the biggest single contributor; voting amplifies it; hard labels preserve accuracy while being the cheapest wire format.
