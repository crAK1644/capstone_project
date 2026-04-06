# SSFL Capstone Project — Comprehensive Report

> **Paper:** Zhao et al., *"Semisupervised Federated-Learning-Based Intrusion Detection Method for Internet of Things,"* IEEE Internet of Things Journal, Vol. 10, No. 10, May 2023.

---

## Table of Contents

1. [Project Summary (TL;DR)](#1-project-summary-tldr)
2. [What Problem Are We Solving?](#2-what-problem-are-we-solving)
3. [The Dataset — N-BaIoT in Detail](#3-the-dataset--n-baiot-in-detail)
4. [How the Data Flows Through the System](#4-how-the-data-flows-through-the-system)
5. [The SSFL Algorithm — Step by Step](#5-the-ssfl-algorithm--step-by-step)
6. [The CNN Architecture — Why a CNN?](#6-the-cnn-architecture--why-a-cnn)
7. [Flower Infrastructure — How It All Connects](#7-flower-infrastructure--how-it-all-connects)
8. [Current Project Status — What's Done, What's Left](#8-current-project-status--whats-done-whats-left)
9. [Anticipated Questions & Answers](#9-anticipated-questions--answers)
10. [Glossary](#10-glossary)

---

## 1. Project Summary (TL;DR)

We are reproducing a research paper that proposes **SSFL** (Semisupervised Federated Learning) — a method for detecting cyberattacks on IoT devices (smart cameras, doorbells, thermostats) **without** any single server collecting everyone's private traffic data.

**In one sentence:** Multiple IoT devices collaboratively train intrusion detection models by exchanging only prediction labels on shared unlabeled data — never model weights, never raw data — achieving both privacy and communication efficiency.

**What makes this different from normal federated learning:**

| Aspect | Traditional FL (FedAvg) | Our Method (SSFL) |
|---|---|---|
| What gets uploaded | Full model weights (millions of floats) | Hard labels (one integer per open sample) |
| Privacy risk | Weights can be reverse-engineered to recover data | Labels cannot recover original data |
| Communication cost | ~3.8 MB per client per round | ~8.9 KB per client per round |
| Handles non-IID data? | Poorly | Yes, via discriminator + voting |

---

## 2. What Problem Are We Solving?

### The Real-World Scenario

Imagine a smart home with a Philips baby monitor, an Ecobee thermostat, and a Samsung webcam. Each device generates network traffic. Some of this traffic is normal (benign), but some could be from botnets like **Mirai** or **Gafgyt** that hijack IoT devices for DDoS attacks.

We want to detect these attacks. But:

1. **Privacy:** We can't collect everyone's traffic data on a central server — that's a huge privacy violation (and illegal under GDPR).
2. **Data heterogeneity:** Each device sees different types of traffic. A doorbell camera might see `mirai.scan` attacks, while a thermostat might see `gafgyt.tcp`. This is the **non-IID** (non Independent and Identically Distributed) problem.
3. **Communication:** IoT devices have limited bandwidth. We can't send large neural network models back and forth every training round.

### Our Solution: SSFL

The paper's solution:
- Each device trains its own local model on its own private data
- All devices share a small pool of **unlabeled** open traffic data (no privacy concern since it's unlabeled and already public)
- Instead of sharing model parameters, devices share their **predictions** (just integer labels) on this open data
- A server aggregates predictions by **voting** and sends the consensus labels back
- Each device then learns from these consensus labels (knowledge distillation)

This cycle repeats for T rounds (typically 200) until the models converge.

---

## 3. The Dataset — N-BaIoT in Detail

### 3.1 Source

The **N-BaIoT dataset** comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT). It was originally created by researchers at Ben-Gurion University who infected real IoT devices with Mirai and Gafgyt malware and recorded the network traffic.

### 3.2 The 9 Devices

| Device ID | Device Name | Traffic Categories |
|---|---|---|
| 1 | Danmini Doorbell | 11 (all categories) |
| 2 | Ecobee Thermostat | 11 |
| 3 | Ennio Doorbell | **6** (benign + 5 gafgyt only, no mirai) |
| 4 | Philips B120N10 Baby Monitor | 11 |
| 5 | Provision PT 737E Security Camera | 11 |
| 6 | Provision PT 838 Security Camera | 11 |
| 7 | Samsung SNH 1011 N Webcam | **6** (benign + 5 gafgyt only, no mirai) |
| 8 | SimpleHome XCS7 1002 WHT Security Camera | 11 |
| 9 | SimpleHome XCS7 1003 WHT Security Camera | 11 |

> **Important for teachers:** Devices 3 and 7 only have 6 categories. This is not a bug — these real devices were only infected with Gafgyt malware, not Mirai. This makes the problem harder because some clients will never see Mirai attacks during training.

### 3.3 The 11 Traffic Categories

| Label | Category | Type | Description |
|---|---|---|---|
| 0 | benign | Normal | Legitimate device traffic |
| 1 | gafgyt.combo | Attack | Gafgyt botnet — combination flood |
| 2 | gafgyt.junk | Attack | Gafgyt — junk traffic flood |
| 3 | gafgyt.scan | Attack | Gafgyt — network scanning |
| 4 | gafgyt.tcp | Attack | Gafgyt — TCP flood (DoS) |
| 5 | gafgyt.udp | Attack | Gafgyt — UDP flood (DoS) |
| 6 | mirai.ack | Attack | Mirai botnet — ACK flood |
| 7 | mirai.scan | Attack | Mirai — network scanning |
| 8 | mirai.syn | Attack | Mirai — SYN flood |
| 9 | mirai.udp | Attack | Mirai — UDP flood |
| 10 | mirai.udpplain | Attack | Mirai — plain UDP flood |

### 3.4 The 115 Features

Each traffic sample has **115 numerical features**. These come from **5 time windows** × **23 statistical features**:

**Time windows:** 100ms, 500ms, 1.5s, 10s, 1 minute

**For each time window, 23 statistics are computed over recent packets:**
- Weight, Mean, Std (3 features)
- Magnitude, Radius, Covariance, Pearson coefficient (4 features)
- Repeated for source IP, source-destination pair, etc.

> **Why 115 features matter:** These statistics capture the *behavior pattern* of traffic over time. A DDoS attack looks very different from normal traffic when you look at packet rate statistics over 100ms vs 1 minute.

### 3.5 mini-N-BaIoT — Our Processed Dataset

The raw N-BaIoT has millions of records (some classes have 100K+ samples). Following the paper, we create a balanced subset:

- **1,000 samples** randomly drawn from each (device, traffic-category) pair
- Total: 9 devices × varying classes = **89,000 samples**
- Random seed: `42` for reproducibility

### 3.6 Data Splits

The 89,000 samples are split into three **disjoint** sets:

| Split | % | Samples | Purpose |
|---|---|---|---|
| **Private** | 70% | 62,300 | Distributed to clients for local supervised training |
| **Open** | 10% | 8,900 | Shared unlabeled data for prediction exchange & distillation |
| **Test** | 20% | 17,800 | Held out — never seen during training — for final evaluation |

> **Critical:** The open set has labels in our files (for verification), but **labels are never used during training**. The whole point of SSFL is that the open data is unlabeled.

### 3.7 Normalisation

All 115 features are scaled to [0, 1] using **min-max normalisation**:

```
X_normalized = (X - X_min) / (X_max - X_min)
```

- `feat_min.npy` and `feat_max.npy` are fitted **only on the private set** (to avoid data leakage from test/open)
- The same min/max values are applied to open and test sets

### 3.8 2D Reshaping for CNN Input

The 115 features are reshaped into a **23 × 5 matrix** (23 features per time window, 5 time windows), stored as tensors with shape `(N, 1, 23, 5)` — the "1" is the single channel (like a grayscale image).

```
Original: [x₀, x₁, ..., x₁₁₄]  (115-d vector)
                    ↓ reshape
Reshaped:  ┌─────────────────────────┐
           │  100ms  500ms  1.5s  10s  1min │
     feat₁ │  x₀     x₂₃    x₄₆   x₆₉  x₉₂  │
     feat₂ │  x₁     x₂₄    x₄₇   x₇₀  x₉₃  │
     ...   │  ...    ...    ...   ...   ...  │
     feat₂₃│  x₂₂    x₄₅    x₆₈   x₉₁  x₁₁₄ │
           └─────────────────────────┘
              (23 rows × 5 columns)
```

> **Why reshape?** CNNs excel at finding **spatial patterns**. By arranging features as a 2D grid, the CNN can learn correlations between adjacent features within the same time window AND across time windows. This is exactly how the paper treats traffic data as a "mini-image."

### 3.9 Non-IID Client Scenarios

The private data (62,300 samples) is distributed to clients in three different ways to simulate real-world data heterogeneity:

#### Scenario 1: 27 Clients (Shard-Based, K=3 per device)
- Each of the 9 devices spawns 3 clients
- Data is sorted by label, split into shards, and 2 shards given per client
- **Result:** Each client has ~2,300 samples, but only 2 different attack categories
- **Difficulty:** Medium

#### Scenario 2: 89 Clients (Shard-Based, K=L per device)
- Each device spawns L clients (L = number of categories for that device)
- So devices with 11 categories produce 11 clients each
- **Result:** Each client has ~700 samples, typically 1-2 categories
- **Difficulty:** Hard (less data per client)

#### Scenario 3: 89 Clients (Dirichlet α=0.1)
- Same number of clients as Scenario 2
- Data is distributed using Dirichlet distribution with α=0.1 (very heterogeneous)
- **Result:** Very uneven distribution — some clients have lots of one class, nothing of another
- **Difficulty:** Hardest

### 3.10 Output File Structure

```
prepared_data/
├── label_map.json              # {"benign": 0, "gafgyt.combo": 1, ...}
├── feat_min.npy                # (115,) min values for normalisation
├── feat_max.npy                # (115,) max values for normalisation
│
├── private/
│   ├── X.npy                   # (62300, 115) — flat features
│   ├── X_2d.npy                # (62300, 1, 23, 5) — CNN-ready
│   ├── y.npy                   # (62300,) — integer labels [0-10]
│   └── device_ids.npy          # (62300,) — which device [1-9]
│
├── open/
│   ├── X.npy                   # (8900, 115)
│   ├── X_2d.npy                # (8900, 1, 23, 5)
│   ├── y.npy                   # (8900,) — exists but NOT used in training
│   └── device_ids.npy
│
├── test/
│   ├── X.npy                   # (17800, 115)
│   ├── X_2d.npy                # (17800, 1, 23, 5)
│   ├── y.npy                   # (17800,) — used for evaluation only
│   └── device_ids.npy
│
├── scenario_1/                 # 27 clients
│   ├── summary.json            # [{client_id, device_id, num_samples, labels_present}, ...]
│   ├── client_0/
│   │   ├── X_2d.npy            # This client's CNN-ready features
│   │   └── y.npy               # This client's labels
│   ├── client_1/
│   │   └── ...
│   └── ...                     # up to client_26
│
├── scenario_2/                 # 89 clients
│   └── ...
│
└── scenario_3/                 # 89 clients (Dirichlet)
    └── ...
```

---

## 4. How the Data Flows Through the System

Here's the complete data flow from raw files to model evaluation:

```
┌──────────────────────────────────────────────────────────────────┐
│                         TRAINING PHASE                          │
│                                                                  │
│  ┌─────────────┐    Private Data        ┌─────────────────────┐ │
│  │ Client k's  │◄───(X_2d, y)───────────│ prepared_data/      │ │
│  │ Classifier  │    per scenario         │ scenario_X/         │ │
│  │   (wᵏᶜ)     │                         │ client_K/           │ │
│  └──────┬──────┘                         └─────────────────────┘ │
│         │                                                        │
│         │ Trained classifier                                     │
│         ▼                                                        │
│  ┌─────────────┐    Open Data           ┌─────────────────────┐ │
│  │  Confidence  │◄───(X_2d)─────────────│ prepared_data/open/ │ │
│  │  Scoring     │    (no labels used!)   └─────────────────────┘ │
│  └──────┬──────┘                                                 │
│         │                                                        │
│         │ Confidence scores (which samples am I sure about?)     │
│         ▼                                                        │
│  ┌─────────────┐                                                 │
│  │ Discriminator│ ← "unfamiliar" open + "familiar" private      │
│  │   (wᵏᵈ)     │    trains to tell them apart                   │
│  └──────┬──────┘                                                 │
│         │                                                        │
│         │ Filter: discard predictions on unfamiliar samples      │
│         ▼                                                        │
│  ┌─────────────┐                                                 │
│  │ Hard Labels  │ → integers: [3, -1, 0, 7, -1, 2, ...]        │
│  │ (per sample) │   (-1 = "I don't know this type of traffic")  │
│  └──────┬──────┘                                                 │
│         │                                                        │
│         │ Upload to server (only ~8.9 KB per round!)             │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    CENTRAL SERVER                            │ │
│  │                                                              │ │
│  │  Collect hard labels from ALL clients                        │ │
│  │  For each of the 8,900 open samples:                         │ │
│  │    - Gather all non-(-1) predictions                         │ │
│  │    - Take MAJORITY VOTE → global label                       │ │
│  │                                                              │ │
│  │  Broadcast global labels back to all clients                 │ │
│  └──────┬──────────────────────────────────────────────────────┘ │
│         │                                                        │
│         │ Global hard labels Pˢ (consensus predictions)          │
│         ▼                                                        │
│  ┌─────────────┐                                                 │
│  │ Distillation │ ← Train classifier on open data using Pˢ     │
│  │ Training     │   (treats global labels as "teacher" labels)  │
│  └─────────────┘                                                 │
│                                                                  │
│  ← Repeat for T rounds (200) →                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                       EVALUATION PHASE                           │
│                                                                  │
│  Test Data (17,800 samples) → Classifier → Predictions          │
│  Compare predictions vs ground truth labels                      │
│  → Accuracy, F1-Score, Precision, Recall                        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. The SSFL Algorithm — Step by Step

Each communication round consists of 5 steps. Here's what happens in plain English, with the corresponding equation from the paper and the exact function in our code:

### Step 1: Train the Classifier (Eq. 11)
**What:** Each client trains its CNN classifier on its own private labeled data using standard supervised learning (cross-entropy loss + Adam optimizer).

**Code:** `train.py::train_classifier()`

**Why:** This is the "base knowledge" — the client learns to recognize the traffic categories it has seen locally.

**Limitation:** Because each client only has a subset of traffic categories (non-IID), the classifier will be good at recognizing some attacks but bad at others.

### Step 1b: Compute Confidence Scores (Eq. 12)
**What:** The trained classifier makes predictions on every sample in the shared open dataset. For each sample, we record:
- The predicted class (argmax of softmax)
- The confidence (max softmax probability)

**Code:** `train.py::compute_confidence_scores()`

**Intuition:** If the classifier outputs `[0.01, 0.02, 0.95, 0.01, ...]` for a sample, it's very confident (0.95) that it's class 2. If it outputs `[0.1, 0.12, 0.11, 0.09, ...]`, it's not confident at all.

### Step 2: Train the Discriminator (Eqs. 13-14)
**What:** We build a binary classifier (the discriminator) that learns to distinguish "familiar" from "unfamiliar" traffic:
- **Unfamiliar samples** = open data samples where the classifier had **low confidence** (below threshold θ)
- **Familiar samples** = all of the client's private data

**Code:** `train.py::build_discriminator_dataset()` + `train.py::train_discriminator()`

**The threshold θ:** Set to the **median** of the client's confidence scores. This is adaptive — each client gets its own threshold based on how confident it generally is.

**Why median?** The paper tested fixed thresholds (0.7, 0.8, 0.9) and found that the median works best because it adapts to each client's confidence level. A client with few categories will naturally be less confident on unfamiliar traffic categories, so a fixed threshold would either be too strict or too lenient.

### Step 3: Filter and Upload (Eq. 16)
**What:** For each open data sample:
1. Ask the discriminator: "Is this sample familiar?"
2. If YES (familiar): use the classifier's prediction as the hard label
3. If NO (unfamiliar): mark as -1 (skip this sample)

**Code:** `train.py::filter_and_predict()`

**Why this matters:** Without the discriminator, a client might confidently but incorrectly predict "class 3" for a Mirai attack it has never seen. The discriminator catches these cases and says "I've never seen traffic like this in my private data, so I shouldn't vote on it."

**Upload:** Each client sends an array of 8,900 integers (one per open sample) to the server. Each integer is either a class label (0-10) or -1. This is extremely communication-efficient.

### Step 4: Vote and Broadcast (Eq. 17) — Server Side
**What:** The server collects predictions from all K clients and for each open sample:
1. Ignores all -1 votes (unfamiliar)
2. Counts how many clients voted for each class
3. The class with the most votes wins → global hard label

**Code:** `run_simulation.py::majority_vote()` (or `server_app.py::SSFLStrategy.aggregate_fit()`)

**Example:**
```
Sample #42:
  Client 0: predicted class 3
  Client 1: predicted -1 (unfamiliar)
  Client 2: predicted class 3
  Client 3: predicted class 5
  Client 4: predicted -1 (unfamiliar)
  → Vote counts: class 3 = 2, class 5 = 1
  → Global label = 3 (majority wins)
```

### Step 5: Distillation (Eq. 18)
**What:** Each client trains its classifier on the open data using the global hard labels as "teacher labels." This allows clients to learn about traffic categories they've never seen in their own private data.

**Code:** `train.py::distillation_train()`

**Why this is powerful:** Client 7 (Samsung webcam) has never seen Mirai attacks. But after voting, the global labels for the open data include correct Mirai labels (from clients that have seen Mirai). By training on these labels, Client 7 learns to recognize Mirai attacks too!

---

## 6. The CNN Architecture — Why a CNN?

### 6.1 Architecture Details (Table I from Paper)

```
Input: (batch, 1, 23, 5) — one "image" per traffic sample

BLOCK 1 — 4 convolutional layers:
  Conv2d(1→64, kernel=3×3, padding=1) + ReLU
  Conv2d(64→64, kernel=3×3, padding=1) + ReLU
  Conv2d(64→64, kernel=3×3, padding=1) + ReLU
  Conv2d(64→64, kernel=3×3, padding=1) + ReLU

BLOCK 2 — 4 convolutional layers:
  Conv2d(64→128, kernel=3×3, padding=1) + ReLU
  Conv2d(128→128, kernel=3×3, padding=1) + ReLU
  Conv2d(128→128, kernel=3×3, padding=1) + ReLU
  Conv2d(128→128, kernel=3×3, padding=1) + ReLU

Flatten: 128 × 23 × 5 = 14,720

MLP Head:
  Linear(14720 → 256) + ReLU
  Linear(256 → 128) + ReLU
  Linear(128 → 11)   ← Classifier (or → 2 for Discriminator)
```

**Total parameters:** ~3.8 million (classifier), same for discriminator (separate copy)

### 6.2 Why CNN and Not Something Else?

The paper chose CNN for specific reasons:

1. **Spatial pattern extraction:** After reshaping 115 features into a 23×5 matrix, the CNN treats traffic samples like tiny images. The convolutional filters can detect patterns that span across features within a time window (vertically) and across time windows (horizontally).

2. **Translation invariance:** CNNs can recognize a pattern regardless of where it appears in the feature matrix — useful because attack signatures might manifest in different features.

3. **Parameter sharing:** Convolutional weights are shared across the entire input, making the model more parameter-efficient than a fully-connected network.

4. **Deep feature extraction:** 8 convolutional layers provide sufficient depth to learn hierarchical features — from low-level statistics to high-level attack signatures.

**Important note from the paper:** *"Our federated training scheme does not aggregate model parameters, which means that it can work even if clients adopt different model structures."* — This is a key design feature. Because SSFL exchanges labels, not weights, different clients could theoretically use different architectures.

---

## 7. Flower Infrastructure — How It All Connects

### 7.1 What is Flower?

[Flower (flwr)](https://flower.ai) is an open-source federated learning framework. It provides:
- A server/client architecture
- Communication primitives
- Simulation capabilities (run K virtual clients on one machine)

### 7.2 Why We Need a Custom Strategy

Flower comes with built-in strategies like `FedAvg` (Federated Averaging). But **we can't use FedAvg** because:

| FedAvg | SSFL |
|---|---|
| Aggregates model **weights** | Aggregates **hard labels** |
| Server computes weighted average | Server does **majority voting** |
| Sends merged weights back | Sends voted labels back |
| Works with `NDArrays` of model params | Works with integer arrays |

So we implemented a **custom** `SSFLStrategy` that overrides:
- `configure_fit()` — sends global hard labels (not weights) to clients
- `aggregate_fit()` — performs majority voting (not averaging) on received hard labels

### 7.3 Project File Structure

```
capstone_project/
├── pyproject.toml                 # Dependencies + Flower config
├── run_simulation.py              # Entry point (direct execution, no Ray)
├── conf/
│   └── config.yaml                # All hyperparameters
├── prepared_data/                 # Pre-processed dataset (see Section 3)
│   └── ...
└── src/
    ├── __init__.py                # Package init
    ├── model.py                   # CNN backbone + Classifier + Discriminator
    ├── data.py                    # PyTorch DataLoader wrappers
    ├── train.py                   # All 6 SSFL training functions
    ├── client_app.py              # SSFLClient (NumPyClient subclass)
    ├── server_app.py              # SSFLStrategy (custom Flower Strategy)
    ├── config.py                  # Runtime config bridge
    └── utils.py                   # Device detection, seeding, metrics
```

### 7.4 How `run_simulation.py` Works

Our simulation works by **directly orchestrating** clients in-process (not using Ray, which had compatibility issues). The flow:

```python
# Pseudocode of the simulation loop
for round in range(1, 201):
    for client in all_clients:
        # 1. Send global labels to client (or empty on round 1)
        # 2. Client runs: distillation → train classifier → confidence →
        #                 discriminator → filter → produce hard labels
        # 3. Collect hard labels
    
    # Server: majority vote across all clients' hard labels
    global_labels = majority_vote(all_hard_labels)
    
    # Evaluate using test set
    accuracy, f1 = evaluate(classifier, test_data)
    print_progress(round, accuracy, f1)
```

**To run:**
```bash
# Quick test (3 rounds)
uv run python run_simulation.py --scenario 1 --rounds 3

# Full experiment
uv run python run_simulation.py --scenario 1 --rounds 200
```

### 7.5 Communication Format

Each round, the actual data exchanged:

**Client → Server:**
```
Array of shape (8900,), dtype=int16
Values: class labels 0-10, or -1 for unfamiliar
Size: 8900 × 2 bytes = 17.8 KB per client
```

**Server → Client (after voting):**
```
Array of shape (8900,), dtype=int16
Values: voted class labels 0-10, or -1 if no client voted
Size: 8900 × 2 bytes = 17.8 KB
```

Compare this to FedAvg which would transmit ~3.8M parameters × 4 bytes = **15.2 MB** per client per round!

### 7.6 Flower's `ClientApp` vs Our Direct Approach

The project has **two execution paths**:

1. **`run_simulation.py`** (Direct Execution) — Currently used. Creates `SSFLClient` instances directly, loops through rounds manually. Simple, debuggable, no Ray dependency.

2. **`src/server_app.py` + `src/client_app.py`** (Flower Native) — For use with `flwr run` or Flower's simulation engine. Uses Flower's `ServerApp`, `ClientApp`, and `Strategy` abstractions. More "correct" from a framework perspective but requires Ray.

Both paths use the same underlying training logic in `train.py`.

---

## 8. Current Project Status — What's Done, What's Left

### ✅ Completed (Phase 1 + Phase 2)

| Component | Status | Details |
|---|---|---|
| Dataset pipeline | ✅ Done | `prepare_dataset.py` — all 89K samples processed |
| Data splits | ✅ Done | Private/Open/Test + 3 scenarios |
| CNN model | ✅ Done | `model.py` — Backbone, Classifier, Discriminator |
| Data loaders | ✅ Done | `data.py` — PyTorch DataLoaders for all splits |
| Training functions | ✅ Done | `train.py` — all 6 algorithm steps |
| Client logic | ✅ Done | `client_app.py` — full SSFL iteration |
| Server logic | ✅ Done | `server_app.py` — custom strategy with voting |
| Simulation runner | ✅ Done | `run_simulation.py` — direct execution loop |
| Config system | ✅ Done | `config.yaml` + `pyproject.toml` |
| Import verification | ✅ Done | All shapes and imports verified |

### ⏳ Remaining (Phase 3 + Phase 4)

| Task | Priority | Description |
|---|---|---|
| End-to-end run | 🔴 High | Run simulation for 5+ rounds, verify training converges |
| Scenario 1 full run | 🔴 High | 27 clients, 200 rounds — should reach ~92% accuracy |
| Scenario 2 full run | 🟡 Medium | 89 clients, 200 rounds — target ~89% accuracy |
| Scenario 3 full run | 🟡 Medium | 89 clients, Dirichlet — target ~85% accuracy |
| Server global model | 🟡 Medium | Train server model via Eq. 10 (currently evaluating client model) |
| Confusion matrices | 🟡 Medium | Reproduce Fig. 3 from paper |
| Accuracy curves | 🟡 Medium | Reproduce Fig. 4 — accuracy over rounds |
| Ablation: no discriminator | 🟢 Low | Remove discriminator, measure accuracy drop |
| Ablation: no voting | 🟢 Low | Replace voting with direct aggregation |
| Ablation: vary θ | 🟢 Low | Test fixed thresholds (0.7, 0.8, 0.9) vs median |
| Soft vs hard labels | 🟢 Low | Communication overhead comparison |

---

## 9. Anticipated Questions & Answers

### Q1: "How is the Flower infrastructure going to be handled?"

**Answer:**

The Flower infrastructure has two modes in our project:

**Mode 1 — Direct Simulation (what we're currently using):**
The file `run_simulation.py` directly creates all client instances (`SSFLClient`) in-memory, loops through rounds manually, and calls the `majority_vote()` function for server-side aggregation. This avoids the complexity of Ray and networking. We chose this because:
- It's easier to debug
- It runs on any OS without Ray issues
- It produces identical results to distributed execution
- It's faster for development

**Mode 2 — Flower Native (ready but not our primary path):**
We have `src/server_app.py` (with `SSFLStrategy`) and `src/client_app.py` (with `client_fn`) that are fully compatible with Flower's `flwr run` command. The `pyproject.toml` already has the `[tool.flwr.app]` section configured:

```toml
[tool.flwr.app.components]
serverapp = "src.server_app:app"
clientapp = "src.client_app:app"
```

**The custom strategy is the key piece.** We subclass Flower's `Strategy` class and override:
- `configure_fit()` — packages global hard labels into Flower's `FitIns` (instead of model weights)
- `aggregate_fit()` — implements majority voting (instead of `FedAvg`'s weighted averaging)
- `initialize_parameters()` — returns `None` (no initial model weights to share)

**Why not FedAvg?** FedAvg averages model parameters across clients. SSFL never exchanges parameters. Clients exchange integer prediction labels. The aggregation is voting, not averaging. Using FedAvg would be fundamentally wrong.

---

### Q2: "Can we use something other than a CNN?"

**Answer: Yes, and the paper explicitly supports this.**

The paper states: *"Our federated training scheme does not aggregate model parameters, which means that it can work even if clients adopt different model structures."*

This is a major advantage of SSFL over FedAvg. In FedAvg, all clients MUST use the same architecture because the server averages their weights element-by-element. In SSFL, since only labels are exchanged, each client could theoretically use a different model.

**Alternative architectures that could work:**

| Architecture | Suitability | Notes |
|---|---|---|
| **MLP (Multi-Layer Perceptron)** | ✅ Works | The paper tested this. Uses flat 115-d input (no 2D reshape). Simpler but lower accuracy than CNN. Paper showed MLP achieves lower accuracy and slower convergence. |
| **LSTM** | ✅ Works | (See Q3 below for detailed analysis) |
| **1D-CNN** | ✅ Works well | Apply 1D convolution along the 115 features. Simpler than 2D-CNN, might work nearly as well. |
| **Transformer** | ✅ Could work | Self-attention on the 5 time windows or 23 features. Would be overkill for this dataset size but would work. |
| **Random Forest / XGBoost** | ⚠️ Partially | Would work for classification but distillation step (Eq. 18) requires gradient-based training. Would need modification. |
| **GRU** | ✅ Works | Similar to LSTM, slightly simpler. Good if treating time windows as a sequence. |

**Why the paper chose CNN:** The 2D reshape creates a structure where CNN excels — it can detect patterns that are local in both the feature dimension and the time-window dimension. The paper's ablation showed CNN outperforms MLP and LSTM on this specific task.

**If a teacher asks "why not try X?"** — The honest answer is: the paper's SSFL framework is architecture-agnostic. We're using CNN because that's what the paper recommends for best results, but swapping in a different model only requires changing `model.py` (and possibly `data.py` if the input shape changes). The entire `train.py`, `client_app.py`, and `server_app.py` would remain unchanged.

---

### Q3: "Can we use LSTM?"

**Answer: Yes, absolutely. Here's how it would work:**

**LSTM (Long Short-Term Memory)** is a type of recurrent neural network designed for sequential data. It's a valid choice because the traffic features naturally form a time series across the 5 time windows.

**How to adapt the data for LSTM:**
- Instead of reshaping to (1, 23, 5) for CNN, we'd reshape to **(5, 23)** — a sequence of 5 timesteps, each with 23 features
- The LSTM processes the sequence: 100ms → 500ms → 1.5s → 10s → 1min
- This captures temporal evolution of traffic patterns

**What the LSTM architecture would look like:**
```
Input: (batch, 5, 23) — 5 timesteps, 23 features each

LSTM(input_size=23, hidden_size=128, num_layers=2, batch_first=True)
  → Output: (batch, 5, 128) — take last timestep → (batch, 128)

MLP Head:
  Linear(128, 128) + ReLU
  Linear(128, 11)  ← 11 traffic categories
```

**What the paper says about LSTM:**
The paper actually tested LSTM as one of its baselines (Table II). Results:

| Model | Scenario 1 Accuracy | Scenario 2 Accuracy | Scenario 3 Accuracy |
|---|---|---|---|
| **CNN (paper's choice)** | ~92% | ~89% | ~85% |
| **LSTM** | Lower | Lower | Lower |
| **MLP** | Lower | Lower | Lower |

The paper concluded that *"the detection performance of both the LSTM model and the MLP model is lower than our CNN model"* but noted that *"the convergence speed of MLP and LSTM models is slower than that of CNN models"* as well.

**Why LSTM performs worse on this specific task:**
1. The "sequence" is only 5 timesteps — LSTMs shine with longer sequences (e.g., 50+ timesteps)
2. The 5 time windows are not a natural temporal sequence — they're different aggregation windows of the same traffic, not consecutive time steps
3. CNN's ability to detect 2D patterns (across features AND time windows simultaneously) is better suited for this particular data representation

**If you want to implement LSTM anyway:**
The change is minimal — only `model.py` and `data.py` need modification. Everything else (Flower infrastructure, training loops, voting, distillation) stays exactly the same because SSFL is model-agnostic.

---

### Q4: "What's the difference between SSFL and FedAvg?"

| Feature | FedAvg | SSFL (Our Method) |
|---|---|---|
| What's exchanged | Model parameters (weights) | Hard labels (integers) |
| Aggregation method | Weighted average of parameters | Majority voting of labels |
| Communication per client per round | ~15.2 MB (for this CNN) | ~17.8 KB |
| Privacy | Weights can be reverse-engineered | Labels cannot reconstruct data |
| Handles non-IID? | Poorly — averaging conflicting models | Well — discriminator filters bad predictions |
| Requires same model on all clients? | Yes (must average weights element-wise) | No (any architecture works) |
| Convergence speed | Slow (200 rounds not enough) | Fast (~150 rounds to converge) |

---

### Q5: "What is knowledge distillation in this context?"

**Knowledge distillation** traditionally means training a small "student" model to mimic a large "teacher" model. In SSFL, the concept is adapted:

- **Teacher:** The global hard labels produced by majority voting (representing the collective knowledge of all clients)
- **Student:** Each client's local classifier
- **Training data:** The shared open dataset (with global labels as pseudo-ground-truth)

So distillation here means: "Train my local model to agree with what all clients collectively think the labels should be." This lets each client learn from the knowledge of other clients **without ever seeing their data or models**.

---

### Q6: "Why use hard labels instead of soft labels?"

The paper compares both:

**Soft labels:** A probability vector `[0.01, 0.05, 0.82, 0.03, ...]` — 11 float64 values per sample
- Upload size per client per round: 8900 × 11 × 8 bytes = **783.2 KB**

**Hard labels:** A single integer `2` — 1 int16 value per sample
- Upload size per client per round: 8900 × 2 bytes = **17.8 KB**

**Result:** Hard labels achieve the **same accuracy** as soft labels but with **44× less communication**. The paper tested rounding soft labels to 2, 4, 6, 8 decimal places and found that even heavily rounded soft labels perform similarly — proving that the hard label (extreme rounding) is sufficient.

---

### Q7: "What is the discriminator and why do we need it?"

The discriminator is a binary classifier that answers one question for each open data sample: **"Is this traffic similar to what I've seen in my private data?"**

**Without discriminator (what goes wrong):**
Client 7 (Samsung webcam) has never seen Mirai attacks. When asked to predict on an open sample that is actually a Mirai attack, the classifier might confidently but incorrectly say "gafgyt.tcp" (because TCP floods and UDP floods look similar in some features). This wrong prediction gets voted on and corrupts the global labels.

**With discriminator (how it fixes the problem):**
The discriminator is trained to recognize that Mirai traffic looks nothing like what Client 7 has seen. It marks those samples as "unfamiliar" (-1), so Client 7 doesn't vote on them. Only clients that have actually seen Mirai traffic vote on Mirai samples.

**Ablation study from the paper:** Removing the discriminator drops accuracy by the largest margin of any component. It's the single most important mechanism in SSFL.

---

### Q8: "How long does training take?"

Estimated times (on a single machine with GPU):

| Scenario | Clients | Rounds | Estimated Time |
|---|---|---|---|
| Scenario 1 | 27 | 200 | ~2-4 hours (GPU), ~8-12 hours (CPU) |
| Scenario 2 | 89 | 200 | ~6-10 hours (GPU), ~24+ hours (CPU) |
| Scenario 3 | 89 | 200 | ~6-10 hours (GPU), ~24+ hours (CPU) |

Each round involves: (per client) training classifier for 5 epochs + training discriminator for 5 epochs + distillation for 5 epochs. With 27-89 clients, this adds up.

**Quick test:** `--rounds 3` should complete in a few minutes and verify the pipeline works.

---

### Q9: "What results should we expect?"

**Paper's reported results (Table II):**

| Scenario | Accuracy | F1-Score | Precision |
|---|---|---|---|
| 1 (27 clients) | 92.89% | 91.06% | 92.44% |
| 2 (89 clients) | 87.34% | 85.26% | 84.60% |
| 3 (89 clients, Dirichlet) | 85.21% | 84.41% | 83.60% |

**Key observations to mention:**
- Scenario 1 is easiest (fewer clients, more data per client)
- Scenario 3 is hardest (extreme non-IID from Dirichlet distribution)
- Even in the hardest scenario, SSFL achieves 85%+ accuracy — much better than FD (50%) or DS-FL (68%)

**Convergence:** SSFL reaches high accuracy within 10 rounds, converges around round 150.

---

### Q10: "What are the hyperparameters and where do they come from?"

All from Section V-C of the paper:

| Parameter | Value | Why |
|---|---|---|
| Learning rate | 0.0001 | Adam optimizer default, works well for this CNN |
| Batch size | 100 | Standard mini-batch size |
| Local epochs | 5 | Number of training epochs per round per client |
| Discriminator epochs | 5 | Same as local epochs |
| Distillation epochs | 5 | Same as local epochs |
| Communication rounds T | 200 | Enough for convergence (actually converges ~150) |
| Confidence threshold θ | Median | Adaptive per-client, proven better than fixed values |
| Random seed | 42 | For reproducibility |

---

## 10. Glossary

| Term | Definition |
|---|---|
| **SSFL** | Semisupervised Federated Learning — the method we're implementing |
| **Federated Learning (FL)** | Training ML models across multiple devices without sharing raw data |
| **FedAvg** | Federated Averaging — the standard FL algorithm that averages model weights |
| **Non-IID** | Not Independent and Identically Distributed — each client sees different data distributions |
| **Knowledge Distillation** | Training a student model to mimic a teacher's predictions |
| **Hard Labels** | Integer class predictions (e.g., 3) — one value per sample |
| **Soft Labels** | Probability vectors (e.g., [0.01, 0.05, 0.82, ...]) — L values per sample |
| **Open Data / Open Set** | Unlabeled data shared across all clients for prediction exchange |
| **Private Data** | Each client's local labeled data, never shared |
| **Classifier (wᵏᶜ)** | CNN model that predicts traffic category (1 of 11 classes) |
| **Discriminator (wᵏᵈ)** | CNN model that predicts familiar vs unfamiliar (binary) |
| **Confidence Score (cⱼᵏ)** | Max softmax probability — how sure the classifier is about a prediction |
| **Threshold θ** | Boundary for familiar/unfamiliar — set to median of confidences |
| **Voting** | Server aggregation: majority vote across all clients' predictions per sample |
| **Flower (flwr)** | Open-source federated learning framework used for our infrastructure |
| **Strategy** | Flower component that defines how the server aggregates client results |
| **N-BaIoT** | The IoT botnet traffic dataset from UCI ML Repository |
| **Gafgyt** | IoT botnet malware (a.k.a. Bashlite) |
| **Mirai** | IoT botnet malware that caused the 2016 Dyn DNS attack |
| **DDoS** | Distributed Denial of Service — flooding a target with traffic |
| **IDS** | Intrusion Detection System |
| **CNN** | Convolutional Neural Network |
| **LSTM** | Long Short-Term Memory (a type of recurrent neural network) |
| **MLP** | Multi-Layer Perceptron (fully connected neural network) |
| **Adam** | Adaptive Moment Estimation — the optimizer used for training |
| **Cross-Entropy** | Loss function for multi-class classification |
| **Softmax** | Function that converts logits to probabilities (sums to 1) |
| **Dirichlet Distribution** | Probability distribution used to create highly non-IID data splits |
| **Ray** | Distributed computing framework, used by Flower for simulation (we bypass it) |

---

*Last updated: April 6, 2026*
