# Classification and Filtering in the Epigenetics HMM

This document explains how the HMM-based classification and filtering system works at both a conceptual and code level.

## Table of Contents
1. [The Problem: Events and Off-Pathway Data](#1-the-problem-events-and-off-pathway-data)
2. [The Fork Architecture](#2-the-fork-architecture)
3. [The Forward-Backward Algorithm](#3-the-forward-backward-algorithm)
4. [Filter Score: Quality Control](#4-filter-score-quality-control)
5. [Classification: Soft vs Hard Calls](#5-classification-soft-vs-hard-calls)
6. [The 26% Threshold Strategy](#6-the-26-threshold-strategy)
7. [Code Walkthrough](#7-code-walkthrough)

---

## 1. The Problem: Events and Off-Pathway Data

### What is an Event?

An **Event** represents a single DNA molecule passing through the nanopore. Physically, it's a burst of electrical current lasting approximately 8.5 seconds.

```
Raw Signal:                Events:
                          ┌──────┐  ┌──────┐  ┌──────┐
~~~~~~▄████▄~~~~~~▄██▄~~~~│Event1│  │Event2│  │Event3│
      └────┘      └──┘    └──────┘  └──────┘  └──────┘
```

After segmentation, each event becomes a list of segment means (current levels in pA):

```python
# A single event after segmentation
event = [45.2, 48.1, 52.3, 47.8, 41.2, ...]  # Current means in pA
```

### The Off-Pathway Problem

Not every detected event represents a clean DNA read. Common problems include:
- DNA getting stuck or tangled
- Enzyme motor falling off mid-sequence
- Random junk (not DNA) blocking the pore

These **off-pathway events** produce noisy data that would corrupt classification accuracy. The system needs a way to identify and filter them out.

---

## 2. The Fork Architecture

The HMM uses a **fork structure** to enable classification. At positions where methylation status affects the current signal, the model splits into parallel paths.

### Model Layout

```
Position:     0-2     3-11          12-24      25-33           34-36      37-42          43-54
              │         │              │          │               │          │              │
Structure:   CAT   {C,mC,hmC}        CAT    {C,mC,hmC}         CAT    {T,X,CAT}          CAT
             stem    FORK #1        stem     FORK #2          stem    FORK #3          stem
                  (unzip read)              (synth read)             (label)
```

### Fork Visualization

```
                     ┌─── C path (positions 25-33) ───┐
Before Fork 2 ───────┼─── mC path ────────────────────┼───── After Fork 2
                     └─── hmC path ───────────────────┘
```

**Key insight**: At each fork position, the three paths have **different expected current distributions**. The HMM learns these during training.

### Position Mapping in Code

The code uses specific position ranges for each fork:

| Fork | Positions | States Named | Purpose |
|------|-----------|--------------|---------|
| Cytosine Fork | 25-33 | `M-C:25-end`, `M-mC:25-end`, `M-hmC:25-end`, ... | Classify modification |
| Label Fork | 37-42 | `M-T:37-end`, `M-X:37-end`, `M-CAT:37-end`, ... | Validate sample type |

---

## 3. The Forward-Backward Algorithm

The classification relies on the **Forward-Backward algorithm**, a dynamic programming method that computes:

1. **Forward pass**: Probability of observing data *up to* each position, for each state
2. **Backward pass**: Probability of observing data *from* each position onward, for each state
3. **Combined**: Expected transition counts between all state pairs

### What We Get

```python
trans, ems = model.forward_backward(event)

# trans: shape (num_states, num_states)
#   - trans[i, j] = expected number of transitions from state i to state j
#   - This tells us "how much probability flow" went through each path

# ems: expected emissions per state (less commonly used for classification)
```

The **transition matrix** is the key output. By summing expected transitions through specific states, we measure how much "probability mass" flowed through each fork path.

---

## 4. Filter Score: Quality Control

The **Filter Score** measures confidence that an event followed the expected path through both forks.

### Conceptual Formula

```
Filter Score = P(traversed cytosine fork) × P(traversed label fork)
```

### How Flow is Calculated

For each path in a fork, we find the **minimum flow** through all positions in that path. This represents the bottleneck—how much probability mass actually made it through the entire path.

```
                    Position 25      26       27      ...     33
                         │           │        │               │
C path flow:          [0.82]------[0.78]---[0.75]---...--->min = 0.72
                                                              ↑
                                            Bottleneck determines path flow
```

### Code Implementation

```python
def calculate_filter_score(trans, indices):
    # Cytosine fork: positions 25-33 for each modification type
    cyt_scores = {}
    for tag in ['C', 'mC', 'hmC']:
        names = ["M-{}:{}-end".format(tag, i) for i in range(25, 34)]
        # Min flow through all positions = bottleneck
        flow = min(trans[indices[name]].sum() for name in names if name in indices)
        cyt_scores[tag] = flow

    # Label fork: positions 37-42
    label_scores = {}
    for tag in ['X', 'T', 'CAT']:
        names = ["M-{}:{}-end".format(tag, i) for i in range(37, 43)]
        matching = [name for name in names if name in indices]
        if matching:
            flow = min(trans[indices[name]].sum() for name in matching)
            label_scores[tag] = flow
        else:
            label_scores[tag] = 0

    # Filter Score = product of fork totals
    cyt_total = sum(cyt_scores.values())     # Sum of C + mC + hmC flow
    label_total = sum(label_scores.values())  # Sum of T + X + CAT flow

    return cyt_total * label_total, cyt_scores, label_scores
```

### Interpreting Filter Scores

| Filter Score | Interpretation |
|--------------|----------------|
| ~1.0 | Excellent alignment; clean on-pathway event |
| 0.1 - 0.5 | Moderate confidence; usable for training |
| < 0.1 | Low confidence; likely off-pathway or noisy |
| ~0 | Event didn't traverse expected forks; garbage |

---

## 5. Classification: Soft vs Hard Calls

### Hard Classification

The path with the **highest expected flow** wins:

```python
# In analyze_events_explained():
if cyt_sum > 0:
    d['Classification'] = max(['C', 'mC', 'hmC'], key=lambda k: d[k])
    d['Confidence'] = d[d['Classification']] / cyt_sum
```

Example:
```
C flow:    0.92   ← Winner
mC flow:   0.05
hmC flow:  0.03

Classification: C (with 92% confidence)
```

### Soft Call (Label-Validated Score)

The **Soft Call** uses a clever validation trick. In the experimental design:
- **C** samples are labeled with **T** (Thymine)
- **mC** samples are labeled with **CAT** (sequence marker)
- **hmC** samples are labeled with **X** (abasic site)

The soft call computes a **dot product** between cytosine classification and expected label:

```python
# In analyze_events():
score = d['C'] * d['T'] + d['mC'] * d['CAT'] + d['hmC'] * d['X']
d['Soft Call'] = score / d['Filter Score'] if d['Filter Score'] != 0 else 0
```

**Why this works**: If the HMM correctly classifies both the cytosine AND the corresponding label, the dot product is high. Mismatches (e.g., calling "mC" but seeing "T" label) produce low scores.

| Soft Call | Meaning |
|-----------|---------|
| ~1.0 | Cytosine and label classifications agree perfectly |
| ~0.5 | Partial agreement or uncertainty |
| ~0 | Classifications don't match expected pairing |

---

## 6. The 26% Threshold Strategy

### The Human Benchmark

In prior work, human experts manually reviewed nanopore data and discarded 74% as "junk," keeping only the cleanest 26% of events. On this hand-picked subset, humans achieved ~90% classification accuracy.

### The Automated Approach

1. **Score all events** using the Filter Score
2. **Sort by confidence** (Filter Score descending)
3. **Keep top 26%** (highest confidence events)
4. **Classify** using hard classification on remaining events

### Results Comparison

```
                    Events Used    Accuracy
Human curation:        26%           90%
HMM (threshold):       26%           97%    ← Beats humans!
```

### Threshold Selection

The threshold of **0.1** was determined empirically to provide optimal separation:

```python
def train(model, events, threshold=0.10):
    # Score all events
    data = analyze_events(events, model)

    # Filter to high-confidence events
    events = [event for score, event
              in zip(data['Filter Score'], events)
              if score > threshold]

    # Train only on clean data
    model.train(events, max_iterations=10, use_pseudocount=True)
```

---

## 7. Code Walkthrough

### Complete Classification Pipeline

Here's the full flow from raw events to classifications:

```python
def analyze_events(events, hmm):
    """
    Main analysis function from epigenetics.py
    """
    data = {
        'Filter Score': [], 'C': [], 'mC': [], 'hmC': [],
        'X': [], 'T': [], 'CAT': [], 'Soft Call': []
    }

    # Build state name -> index mapping
    indices = {state.name: i for i, state in enumerate(hmm.states)}
    tags = ('C', 'mC', 'hmC', 'X', 'T', 'CAT')

    for event in events:
        d = {key: None for key in data.keys()}

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Run Forward-Backward to get expected transitions
        # ═══════════════════════════════════════════════════════════════
        trans, ems = hmm.forward_backward(event)

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Calculate flow through CYTOSINE fork (positions 25-33)
        # ═══════════════════════════════════════════════════════════════
        for tag in 'C', 'mC', 'hmC':
            names = ["M-{}:{}-end".format(tag, i) for i in range(25, 34)]
            # Minimum = bottleneck flow through this path
            d[tag] = min([trans[indices[name]].sum() for name in names])

        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Calculate flow through LABEL fork (positions 37-42)
        # ═══════════════════════════════════════════════════════════════
        for tag in 'X', 'T', 'CAT':
            names = ["M-{}:{}-end".format(tag, i) for i in range(37, 43)]
            d[tag] = min([trans[indices[name]].sum() for name in names])

        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Calculate FILTER SCORE
        # ═══════════════════════════════════════════════════════════════
        # Product of: (total cytosine flow) × (total label flow)
        d['Filter Score'] = sum(d[tag] for tag in tags[:3]) * \
                           sum(d[tag] for tag in tags[3:])

        # ═══════════════════════════════════════════════════════════════
        # STEP 5: Calculate SOFT CALL (label-validated classification)
        # ═══════════════════════════════════════════════════════════════
        # Dot product of cytosine × matching label
        score = d['C'] * d['T'] + d['mC'] * d['CAT'] + d['hmC'] * d['X']
        d['Soft Call'] = score / d['Filter Score'] if d['Filter Score'] != 0 else 0

        # Append results
        for key, value in d.items():
            data[key].append(value)

    return pd.DataFrame(data)
```

### State Naming Convention

Understanding state names is crucial for reading the code:

```
State Name Format: M-{modification}:{position}-end

Examples:
  M-C:25-end     → Match state for Cytosine at position 25
  M-mC:30-end    → Match state for Methylcytosine at position 30
  M-hmC:33-end   → Match state for Hydroxymethylcytosine at position 33
  M-T:40-end     → Match state for Thymine label at position 40
```

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLASSIFICATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Raw Event                                                              │
│   [45.2, 48.1, 52.3, 47.8, ...]                                         │
│        │                                                                 │
│        ▼                                                                 │
│   ┌────────────────────────┐                                            │
│   │   Forward-Backward     │                                            │
│   │   Algorithm            │                                            │
│   └────────────────────────┘                                            │
│        │                                                                 │
│        ▼                                                                 │
│   Transition Matrix (trans)                                             │
│   [expected transitions between all state pairs]                        │
│        │                                                                 │
│        ├──────────────────────────────────────────┐                     │
│        ▼                                          ▼                     │
│   ┌─────────────────────┐                 ┌─────────────────────┐       │
│   │ CYTOSINE FORK       │                 │ LABEL FORK          │       │
│   │ (positions 25-33)   │                 │ (positions 37-42)   │       │
│   │                     │                 │                     │       │
│   │ C:   0.72 ──┐       │                 │ T:   0.89 ──┐       │       │
│   │ mC:  0.05  ├─→0.78  │                 │ X:   0.05  ├─→0.97  │       │
│   │ hmC: 0.01 ──┘       │                 │ CAT: 0.03 ──┘       │       │
│   └─────────────────────┘                 └─────────────────────┘       │
│        │                                          │                     │
│        └─────────────────┬────────────────────────┘                     │
│                          ▼                                               │
│   ┌──────────────────────────────────────┐                              │
│   │ FILTER SCORE = 0.78 × 0.97 = 0.76    │ ← Confidence measure        │
│   └──────────────────────────────────────┘                              │
│                          │                                               │
│                          ▼                                               │
│   ┌──────────────────────────────────────┐                              │
│   │ Filter Score > 0.1? ─────────────────┼──→ No: Discard (off-pathway)│
│   │         │                            │                              │
│   │         Yes                          │                              │
│   │         ▼                            │                              │
│   │ Classification: C (highest flow)     │                              │
│   │ Soft Call: 0.72 × 0.89 / 0.76 = 0.84│                              │
│   └──────────────────────────────────────┘                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **Event** | Single DNA molecule read | List of current means in pA |
| **Fork** | Parallel paths for different modifications | States at positions 25-33 and 37-42 |
| **Forward-Backward** | Compute probability flow | `model.forward_backward(event)` |
| **Filter Score** | Quality control; identify off-pathway | `cyt_total × label_total` |
| **Hard Classification** | Winner-take-all decision | `max(C, mC, hmC)` by flow |
| **Soft Call** | Label-validated confidence | Dot product of classification × label |
| **Threshold** | Filter poor-quality events | Events with Filter Score > 0.1 |

The elegance of this system lies in using the **same HMM structure** for both:
1. **Filtering** (are the forks traversed?)
2. **Classification** (which path was taken?)

This allows a single forward-backward pass to simultaneously assess quality AND make predictions.
