# SSFL System Design — Concept Guide

> **Paper:** Zhao et al., *"Semisupervised Federated-Learning-Based Intrusion Detection Method for Internet of Things"*, IEEE IoT Journal, Vol. 10, No. 10, May 2023.  
> **Scope:** This document explains the four core concepts behind our Flower-based SSFL implementation — the round structure, client-server communication, majority voting, and hard label production — at the design level, without diving into implementation code.

---

## Table of Contents

1. [The Round — What Happens in One Cycle](#1-the-round--what-happens-in-one-cycle)
2. [Client–Server Communication — What Gets Exchanged and How](#2-clientserver-communication--what-gets-exchanged-and-how)
3. [Hard Labels — What They Are and How a Client Produces Them](#3-hard-labels--what-they-are-and-how-a-client-produces-them)
4. [Majority Voting — How the Server Aggregates](#4-majority-voting--how-the-server-aggregates)

---

## 1. The Round — What Happens in One Cycle

### 1.1 What Is a Round?

The entire SSFL training process is divided into **T = 200 communication rounds**. Each round is one full cycle of:

> **Server sends something → every client does local work → every client sends something back → server aggregates**

This is the fundamental building block of federated learning. After 200 rounds, each client's model has been improved by knowledge distilled from all other clients — without any client ever sharing its private data or its model weights.

### 1.2 Why Flower?

**Flower** is an open-source federated learning framework. Its job is purely **orchestration** — it manages the round loop, decides when to call which functions, routes messages between the server and clients, and serializes data for transmission. Flower does not define the learning algorithm; that is entirely our responsibility.

```
Flower provides:         We implement:
┌──────────────────┐     ┌──────────────────────────────────────────┐
│ Round loop       │     │ What the server sends each round         │
│ Client spawning  │     │ What each client does with what it gets  │
│ Message routing  │     │ What each client sends back              │
│ Serialization    │     │ How the server aggregates the responses  │
│ Strategy interface│    │ The voting algorithm                     │
│ Client interface │     │ The training procedure (all 5 steps)     │
└──────────────────┘     └──────────────────────────────────────────┘
```

### 1.3 The Five Steps Inside One Round

Every round maps directly to **Algorithm 1** from the paper. The diagram below shows what happens and where Flower is involved:

```
╔══════════════════════════════════════════════════════════════════════════╗
║                         ROUND  t  (of 200)                             ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │  SERVER  (Flower calls configure_fit — we implement the body)   │    ║
║  │  Sends: global hard labels Pˢ from round t-1                    │    ║
║  │         (empty array on round 1 — cold start)                   │    ║
║  └────────────────────────────┬────────────────────────────────────┘    ║
║                               │ Flower routes this to all clients        ║
║                               ▼                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │  EACH CLIENT  (Flower calls fit — we implement the entire body) │    ║
║  │                                                                  │    ║
║  │  [if round ≥ 2] Step 5  ← DISTILLATION on open data with Pˢ    │    ║
║  │                                                                  │    ║
║  │               Step 1  ← Train classifier on private data        │    ║
║  │               Step 1b ← Score every open sample for confidence  │    ║
║  │               Step 2  ← Train discriminator (familiar/not?)     │    ║
║  │               Step 3  ← Filter open predictions → hard labels   │    ║
║  │                                                                  │    ║
║  │  Returns: hard labels array (one integer per open sample)        │    ║
║  └────────────────────────────┬────────────────────────────────────┘    ║
║                               │ Flower collects all client returns        ║
║                               ▼                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │  SERVER  (Flower calls aggregate_fit — we implement the body)   │    ║
║  │  Step 4: Majority voting across all clients' hard labels        │    ║
║  │  Result: new global labels Pˢ  (stored for next round)          │    ║
║  │  Also:   evaluate on held-out test set → log accuracy / F1      │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 1.4 What "Rounds" Accomplish Over Time

| Round range | What is happening |
|---|---|
| 1–5 | Cold start. Classifiers are undertrained. Many open samples get abstain votes (-1). Global labels are sparse but already useful for common traffic types. |
| 6–30 | Distillation starts transferring knowledge. A client that has never seen Mirai attacks begins to recognize them from the global labels voted by clients that have. |
| 30–100 | Coverage improves rapidly. Fewer -1 votes, better global label quality, distillation signal strengthens. |
| 100–200 | Convergence. Models stabilize around 85–92% accuracy depending on the scenario. |

The distillation–voting feedback loop is what drives improvement across rounds. No single client could reach this accuracy alone.

---

## 2. Client–Server Communication — What Gets Exchanged and How

### 2.1 What SSFL Exchanges vs. FedAvg

In standard federated learning (FedAvg), the thing being transmitted is **model weights** — millions of floating-point numbers. In SSFL, it is **hard-label predictions** on a shared open dataset — a short list of integers.

| | FedAvg | SSFL (ours) |
|---|---|---|
| **Server → Client** | Global model weights (~3.8 MB) | Global hard labels (~17.4 KB) |
| **Client → Server** | Locally trained model weights (~3.8 MB) | Filtered hard-label predictions (~17.4 KB) |
| **Server aggregation** | Weighted average of all weight tensors | Majority vote on integer predictions |
| **Privacy** | Weights can be reverse-engineered | Integer labels cannot recover raw data |
| **Bandwidth per round** | ~3.8 MB × 2 × K clients | ~17.4 KB × 2 × K clients |

> The 17.4 KB figure comes from 8,900 open samples × 2 bytes per int16 label. This is the paper's main communication efficiency claim.

### 2.2 The Two Flower Interfaces We Implement

Flower defines two abstract interfaces. We provide the concrete implementations.

#### The Server Interface — `Strategy`

Flower calls three methods on our server object during each round:

```
configure_fit(round, parameters, client_manager)
  └── Called by Flower at the START of each round.
      Our job: decide what to send to clients.
      We send: the global hard labels from last round (or empty on round 1).
      We return: a list of (client, message) pairs — one per client.
      Flower then routes each message to the right client.

aggregate_fit(round, results, failures)
  └── Called by Flower AFTER all clients have returned.
      Our job: aggregate the results.
      We receive: a list of (client_id, result) pairs.
      Each result contains that client's hard-label array.
      We run majority voting → store new global labels.
      Flower expects us to return aggregated parameters — we return None
      because there are no model weights to aggregate.

configure_evaluate() / aggregate_evaluate()
  └── Flower's hooks for client-side evaluation.
      We stub these out (no-op) — in SSFL, evaluation is done
      centrally on the server, not distributed to clients.
```

#### The Client Interface — `NumPyClient`

Flower calls these methods on each client instance:

```
get_parameters(config)
  └── Called by Flower once at startup.
      In FedAvg: return initial model weights.
      In SSFL: return empty array — we do not share weights.

fit(parameters, config)
  └── Called by Flower ONCE PER ROUND on each client.
      parameters: contains the global hard labels from the server.
      Our job: run the full local training procedure (Steps 1–3),
               then return our hard-label predictions.
      We return: [hard_labels_array], num_samples, metrics_dict

evaluate(parameters, config)
  └── Called by Flower for optional client-side evaluation.
      We return a no-op — server handles all evaluation.
```

### 2.3 The Actual Data Flowing Each Round

**Server → Client (at round start via `configure_fit`):**

```
On round 1:
  payload = empty array   ← no global labels exist yet

On round ≥ 2:
  payload = int array of shape (8900,)
  
  Index:  [  0,   1,   2, ..., 8899 ]
  Value:  [  3,  -1,   0, ...,    7 ]
           ↑              ↑
           open sample 0  open sample 8899
           voted label: 3  voted label: 7
           (-1 means no consensus was reached)
```

**Client → Server (at round end, returned from `fit`):**

```
payload = int array of shape (8900,)

Index:  [  0,   1,   2, ..., 8899 ]
Value:  [  3,  -1,   0, ...,   -1 ]
         ↑              ↑
         "I predict open  "I don't recognize
          sample 0 = class 3"  this traffic type"
          (familiar, confident)  (unfamiliar, abstain)
```

Both payloads are arrays of exactly 8,900 integers — one per open sample. The -1 value is the abstain signal (see Section 3).

### 2.4 How Flower Carries This Data

Flower's built-in message type, `Parameters`, was designed to carry float32 model weights. We repurpose it to carry integer label arrays. This works because Flower treats `Parameters` as an opaque byte buffer — it serializes and routes it without inspecting the contents. On both ends, we cast between int16 and float64 to satisfy the type system, then cast back.

```
Client side (sending):
  hard_labels (int16 array)
    → cast to float64
    → ndarrays_to_parameters()   ← Flower utility
    → Parameters object (bytes)
    → returned inside FitRes     ← Flower message type

Server side (receiving):
  FitRes.parameters (bytes)
    → parameters_to_ndarrays()   ← Flower utility
    → float64 array
    → cast to int16
    → hard_labels (8900,)
```

The same process runs in reverse for the server-to-client direction.

---

## 3. Hard Labels — What They Are and How a Client Produces Them

### 3.1 What Is a Hard Label?

A **hard label** is a single integer representing a class prediction. For our system, that integer is in the range 0–10 (the 11 traffic categories) or **-1** (the abstain signal).

```
0 = benign traffic
1 = gafgyt.combo attack
2 = gafgyt.junk attack
3 = gafgyt.scan attack
4 = gafgyt.tcp attack
5 = gafgyt.udp attack
6 = mirai.ack attack
7 = mirai.scan attack
8 = mirai.syn attack
9 = mirai.udp attack
10 = mirai.udpplain attack
-1 = "I don't recognize this" (abstain)
```

Each client produces **8,900 hard labels** per round — one for each sample in the shared open dataset.

### 3.2 Why Hard Labels Instead of Probabilities?

The paper could have used **soft labels** (probability distributions, e.g. [0.02, 0.01, 0.95, ...]) — but it uses hard labels (just the integer argmax, e.g. 3) for two reasons:

1. **Communication efficiency:** One int16 per sample vs. 11 float32 values per sample. Hard labels are 44× smaller.
2. **Aggregation simplicity:** Voting on integers is straightforward. Averaging probability distributions is more complex and doesn't work well when clients have very different class distributions.

### 3.3 The Three-Step Process to Produce Hard Labels

Producing hard labels for the open dataset is not trivial. A client cannot just run its classifier and return the predictions — that would produce overconfident, often wrong results for traffic types the client has never seen. The SSFL algorithm adds a **discriminator gating mechanism** to filter out low-quality predictions before they are sent to the server.

The three steps are:

---

#### Step A — Train the Classifier, Score the Open Data

First, the client trains its CNN classifier (`wᵏᶜ`) on its own private labeled data. This is standard supervised learning.

Then, the trained classifier runs inference on all 8,900 open samples. For each sample, the classifier outputs a **softmax probability distribution** across 11 classes. Two values are recorded:

```
For each open sample j:
  predicted_class[j]  = argmax(softmax output)   ← the hard label candidate
  confidence[j]       = max(softmax output)       ← how sure the model is

Example:
  softmax output = [0.01, 0.02, 0.94, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]
  predicted_class = 2    (gafgyt.junk — the dominant class)
  confidence      = 0.94 (very sure)

  softmax output = [0.09, 0.11, 0.10, 0.08, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.08]
  predicted_class = 1    (technically the max, but barely)
  confidence      = 0.11 (not at all sure — 11 classes, almost uniform)
```

The **confidence threshold θ** is set to the **median** of all 8,900 confidence scores for this client. Any sample above the median is "familiar" — the client is confident enough to have an opinion. Any sample below the median is "unfamiliar" — the client abstains.

> The threshold is adaptive (median-based) rather than fixed because different clients have different base confidence levels depending on how many traffic categories they have seen.

---

#### Step B — Train the Discriminator

The **discriminator** (`wᵏᵈ`) is a second CNN trained to answer one binary question: *"Does this open sample look like traffic I've seen before?"*

It is built from two groups of samples:

```
Familiar samples  (label = 1):
  → All of the client's private data         ← definitely known territory
  → Open samples where confidence > θ        ← the client was confident here

Unfamiliar samples (label = 0):
  → Open samples where confidence ≤ θ        ← the client was unsure here
```

The discriminator trains on this binary dataset. After training, it can make a binary judgment on any open sample: *familiar* or *unfamiliar*.

**Why is this necessary?** A classifier can be confidently wrong. A client that has never seen Mirai attacks might still output a high softmax value for some Mirai sample — just because its classifier assigns everything to a few known classes. The discriminator catches this: if an open sample doesn't look like anything in the client's private distribution, the discriminator will flag it as unfamiliar regardless of what the classifier said.

```
Example: Client whose private data only has gafgyt attacks

  Open sample j = a Mirai.syn packet
  Classifier output: [0.05, 0.04, 0.78, 0.04, 0.04, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
  → Classifier says: "gafgyt.junk" with confidence 0.78  (wrong!)

  Discriminator: "Does this look like my private data?"
  → The traffic statistics are structurally different from all known gafgyt
  → Discriminator output: UNFAMILIAR (0)

  Result: this client abstains on sample j  (-1)
  The wrong prediction never reaches the server.
```

---

#### Step C — Filter and Produce the Final Hard Labels

With both models trained, the client runs them together over all 8,900 open samples:

```
For each open sample j:

  1. Ask discriminator: familiar (1) or unfamiliar (0)?
  
  2. If UNFAMILIAR:
       hard_label[j] = -1      ← abstain, do not vote

  3. If FAMILIAR:
       hard_label[j] = predicted_class[j]   ← from Step A
```

The result is an array of 8,900 values. Each value is either a class label (0–10) or the abstain signal (-1).

```
Example output from one client:
  Index:  [   0,   1,   2,   3,   4,   5, ..., 8899 ]
  Value:  [   3,  -1,   0,   3,  -1,   4, ...,   -1 ]
           ↑       ↑                            ↑
           familiar  unfamiliar                 unfamiliar
           → class 3  → abstain                 → abstain
```

This array is what the client sends to the server. The -1 values do not participate in voting.

---

## 4. Majority Voting — How the Server Aggregates

### 4.1 The Problem Voting Solves

Each client has a biased, incomplete view of the traffic space. A doorbell-based client knows gafgyt attacks well but has never seen Mirai. A webcam-based client has the reverse bias. No single client's hard labels are trustworthy for all 8,900 open samples.

Voting solves this by treating each client as an **expert witness** — but only for the traffic types it recognizes. Clients that recognize a sample cast a vote; clients that don't recognize it abstain (-1). The majority wins.

### 4.2 The Voting Procedure

After all K clients have returned their hard-label arrays, the server has a **K × 8900 matrix**:

```
         open sample:   0    1    2    3    4  ...  8899
                       ─────────────────────────────────
Client 0  │            3   -1    0    3   -1  ...    7
Client 1  │           -1   -1    0    3    4  ...   -1
Client 2  │            3    2    0   -1    4  ...    7
Client 3  │           -1   -1   -1    3   -1  ...    7
Client 4  │            3   -1    0    3    4  ...   -1
  ...     │           ...  ...  ...  ...  ...        ...
Client K-1│            3    2    0    3    4  ...    7
```

For each of the 8,900 open samples (each column), the server:

1. **Collects all votes** for that sample
2. **Discards -1 votes** (abstentions — the discriminator said unfamiliar)
3. **Counts votes per class** using a bincount over valid votes
4. **Assigns the majority class** as the global label

```
Voting for open sample j = 3  (column index 3):

  Client 0 voted:  3
  Client 1 voted:  3
  Client 2 voted: -1   ← discarded (unfamiliar)
  Client 3 voted:  3
  Client 4 voted:  3
  Client 5 voted:  5
  ...
  Client 26 voted: 3

  Valid votes: [3, 3, 3, 3, 5, ..., 3]
  Vote counts: {class 3: 21 votes,  class 5: 2 votes,  others: 0}
  Global label: 3   ← majority wins
```

### 4.3 Edge Cases

| Situation | What Happens |
|---|---|
| All clients abstain (-1) for a sample | Global label stays -1 — no consensus. Sample is skipped in next round's distillation. |
| Two classes tie | `argmax` over the vote count array returns the lower class index (deterministic tie-break). |
| Only one client votes on a sample | That single vote wins. Unanimous by default. |
| A client is missing / failed | Flower reports it as a failure. We skip it — the round proceeds with the remaining clients. |

### 4.4 Why Voting Instead of Averaging?

In FedAvg, the server averages model weights. Averaging makes sense for continuous floats. It does not make sense for class labels — the average of "class 3" and "class 7" is not "class 5." Majority voting is the correct operation for discrete categorical predictions.

```
FedAvg aggregation:
  Weights from client 0:  [0.31, 0.12, ...]
  Weights from client 1:  [0.29, 0.14, ...]
  Weights from client 2:  [0.30, 0.11, ...]
  → Average:              [0.30, 0.12, ...]   ← meaningful

SSFL aggregation:
  Prediction from client 0:  3
  Prediction from client 1:  3
  Prediction from client 2:  5
  → Average: (3+3+5)/3 = 3.67   ← meaningless
  → Majority vote: 3            ← correct
```

### 4.5 What Gets Stored and Sent Next Round

After voting, the server stores the resulting `(8900,)` array as its **global labels** (`Pˢ`). This is the only server-side state that persists between rounds.

At the start of the next round, `configure_fit()` broadcasts this array to every client. Clients use it for distillation (Step 5 / Phase 5 of the next round), training their classifiers on the open data using these consensus labels as supervision.

```
Voting output (stored as global labels):
  Index:  [   0,   1,   2, ..., 8899 ]
  Value:  [   3,   2,   0, ...,    7 ]
           ↑       ↑                ↑
           consensus   consensus    consensus
           = class 3   = class 2    = class 7
           (-1 entries remain for samples with no consensus)
```

This array is the only thing the server ever sends to clients. There is no global model. There are no shared weights. The "knowledge" of the federation lives entirely in this short integer array.

---

## Summary: Flower vs. Our Implementation

| Concept | Flower's Role | Our Implementation |
|---|---|---|
| **Round loop** | Drives the loop via `ServerConfig(num_rounds=200)` | No code needed — Flower handles this |
| **Calling client logic** | Calls `fit()` on each client each round | The entire body of `fit()` — Steps 1–3 + distillation |
| **Calling server logic** | Calls `configure_fit()` and `aggregate_fit()` | Bodies of both methods — what to send, how to vote |
| **Routing messages** | Routes `FitIns` server→client, `FitRes` client→server | Packing/unpacking the label arrays into Flower message types |
| **Hard label production** | Not involved | Classifier scoring + discriminator filtering (Steps 1b–3) |
| **Majority voting** | Not involved | Column-wise argmax over the (K × 8900) vote matrix |
| **Distillation** | Not involved | Training classifier on open data with global labels |
| **Evaluation** | Provides hook (`aggregate_evaluate`) | Server-side test set evaluation after each vote |
| **Serialization** | `ndarrays_to_parameters` / `parameters_to_ndarrays` | Casting int16 ↔ float64 to fit Flower's type system |
