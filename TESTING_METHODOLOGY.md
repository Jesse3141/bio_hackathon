# Testing Methodology for Cytosine Classification HMMs

> **Related Documentation**: [README.md](README.md) | [RESULTS.md](RESULTS.md) | [UCSCN/model_summary.md](UCSCN/model_summary.md) | [UCSCN/classification_filtering_explained.md](UCSCN/classification_filtering_explained.md)

This document explains the evaluation framework used in the Schreiber & Karplus cytosine classification system. Use this as a reference for benchmarking new models on similar tasks.

**Source Code**: The metrics are implemented in:
- `UCSCN/epigenetics_patched.py` - Original implementation (lines 488-549: `analyze_events`, 632-647: `test`, 648-678: `n_fold_cross_validation`)
- `methylation_hmm/schreiber_karplus_evaluation.py` - Modern reimplementation

## Overview

The testing framework evaluates HMMs that classify cytosine modifications (C, 5mC, 5hmC) from nanopore ionic current data. The key insight is that **not all events are equally informative** - some events are "on-pathway" (following expected enzyme behavior) while others are "off-pathway" (aberrant). The framework explicitly handles this by ranking events by confidence.

## Core Metrics

### 1. Filter Score

**Purpose:** Identify on-pathway events that traversed both classification forks in the HMM.

**Calculation:**
```
Filter Score = (sum of cytosine fork scores) × (sum of label fork scores)
```

Where each fork score is the **minimum expected transition count** across all positions in that fork:

```python
# Cytosine fork (positions 25-33, classifies C/mC/hmC)
for tag in ['C', 'mC', 'hmC']:
    names = ["M-{}:{}-end".format(tag, i) for i in range(25, 34)]
    score[tag] = min([trans[indices[name]].sum() for name in names])

# Label fork (positions 37-42, validates with T/X/CAT labels)
for tag in ['X', 'T', 'CAT']:
    names = ["M-{}:{}-end".format(tag, i) for i in range(37, 43)]
    score[tag] = min([trans[indices[name]].sum() for name in names])

filter_score = sum(cytosine_scores) * sum(label_scores)
```

**Interpretation:**
- High filter score (>0.1) → event likely traversed both forks completely
- Low filter score (<0.01) → event likely off-pathway or incomplete
- Using minimum ensures the event passed through ALL positions in the fork

### 2. Soft Call

**Purpose:** Measure classification confidence for a single event.

**Calculation:**
```
Soft Call = (C×T + mC×CAT + hmC×X) / Filter Score
```

This is a dot product between:
- Cytosine variant posterior probabilities (C, mC, hmC)
- Corresponding label posterior probabilities (T, CAT, X)

**Interpretation:**
- Range: 0.0 to 1.0
- 1.0 = perfect agreement between cytosine call and label call
- The labels act as independent validation of the cytosine classification

**Why this works:** The experimental design uses different labels for each cytosine variant:
| Cytosine | Label | Reason |
|----------|-------|--------|
| C (canonical) | T (thymidine) | Control |
| mC (5-methylcytosine) | CAT (biotinylated) | Standard methylation marker |
| hmC (5-hydroxymethylcytosine) | X (abasic) | Hydroxymethyl-specific chemistry |

### 3. MCSC (Mean Cumulative Soft Call)

**Purpose:** Report accuracy using only the top N most confident events. This produces an **accuracy curve**, not a single number.

**Calculation:**
```python
# Sort events by Filter Score (descending)
data = data.sort_values('Filter Score', ascending=False)

# MCSC[i] = mean soft call for events ranked i and better
MCSC = [sum(soft_calls[i:]) / (n - i) for i in range(n)]
```

**Visual representation:**
```
Events (sorted by confidence):  [best] ←———————————————————→ [worst]
MCSC curve:                      97% ———→ 94% ———→ 85% ———→ 75%
                                  ↑                          ↑
                              top 26%                    all 100%
```

**Interpretation:**
- MCSC at rank 0 = accuracy using ALL events (~75-80%)
- MCSC at top 25% = accuracy using only the 25% most confident events (~97%)
- The curve shows how accuracy improves as you filter to higher-confidence predictions

**Critical insight:** The paper's headline "97.7% accuracy" is MCSC at top 26% coverage - NOT overall accuracy. Using ALL events, accuracy is only ~75-80%. This is why reporting a single accuracy number is misleading; always report the full curve or specify the coverage threshold.

## Training Thresholds

### Filter Score Threshold for Training

**Purpose:** Exclude off-pathway events from Baum-Welch training.

**The Training Algorithm:**

```python
def train(model, events, threshold=0.10):
    # Step 1: Score ALL training events using the CURRENT model
    # (initially the untrained model with hand-curated emission params)
    data = analyze_events(events, model)

    # Step 2: Filter - keep only events where Filter Score > threshold
    filtered_events = [
        event for score, event in zip(data['Filter Score'], events)
        if score > threshold
    ]

    # Step 3: Run Baum-Welch ONLY on filtered events
    model.train(filtered_events, max_iterations=10, use_pseudocount=True)

    return model
```

**Key insight:** The model scores events using its *current* parameters (from the untrained model's hand-curated distributions), then trains only on events it considers "on-pathway." This is a form of **self-training** where the model's own confidence determines which data it learns from.

**Why this works:**
1. The untrained model has reasonable emission distributions from hand-labeled data
2. Events with high filter scores structurally match the expected HMM topology
3. Training on clean data refines emissions without being corrupted by aberrant events
4. Off-pathway events (enzyme stalls, incomplete translocations, etc.) are excluded

**Threshold scan values tested:** 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 0.75, 0.90

**Optimal threshold:** 0.10 (chosen based on cross-validation accuracy curves)

**Trade-off:**
- Too low (0.01): Include noisy off-pathway events → corrupted training
- Too high (0.5+): Too few training events → poor generalization
- Sweet spot (0.10): ~25-30% of events retained, good signal-to-noise

**Rationale:** Training on off-pathway events would corrupt the emission distributions. The threshold ensures only events that structurally match the expected model topology are used for parameter estimation.

### Event Definition Thresholds

Events are detected from raw current with these criteria:
- **Current band:** 0 - 90 pA
- **Minimum duration:** 500 ms (0.5 seconds)
- **Segmentation:** ~40 segments per second prior

## Validation Procedures

### 1. Train/Test Split

```python
# 70% training, 30% testing
train_fold, test_fold = train_test_split(events, 0.7)

# Further split training for hyperparameter tuning
train_fold, cross_train_fold = train_test_split(train_fold, 0.7)
# Results: 49% train, 21% cross-train, 30% test
```

### 2. N-Fold Cross-Validation

**Standard configuration:** 5-fold CV repeated 10 times

```python
for iteration in range(10):
    random.shuffle(events)
    accuracies.append(n_fold_cross_validation(events, n=5))
```

**Reported metrics:**
- Mean MCSC curve across all folds/iterations
- Min/max range (shown as shaded region in plots)

### 3. Threshold Scanning Validation

Used to find optimal filter score threshold:

```python
for threshold in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 0.9]:
    # Filter training events
    filtered_train = [e for e, s in zip(train, scores) if s > threshold]

    # Train fresh model
    model = load_untrained_model()
    model.train(filtered_train, max_iterations=10)

    # Evaluate on held-out cross-train set
    accuracy_curve = test(model, cross_train_fold)
```

## Diagnostic Plots

### 1. Insert/Delete Plot

Shows expected artifact frequency at each HMM position:
- **Inserts (cyan):** Oversegmentation - extra segments detected
- **Deletes (magenta):** Undersegmentation - missed segments
- **Backslips (orange):** Enzyme backtracking
- **Undersegmented (green):** Adjacent segments that should have been split

**Usage:** Identify systematic problems at specific positions. High artifact rates suggest emission distribution issues or chemistry problems.

### 2. MCSC vs Event Index Plot

Shows accuracy as function of event quality threshold:
- X-axis: Event rank (sorted by Filter Score)
- Y-axis: MCSC at that rank
- Look for: Knee point where accuracy drops (indicates transition from on-pathway to off-pathway events)

### 3. Filter Score vs MCSC Plot

Direct relationship between confidence threshold and expected accuracy:
- Useful for choosing operational threshold based on desired accuracy
- Log scale reveals behavior at low filter scores

### 4. Soft Call Scatter + Rolling Mean

Per-event soft call with local average:
- Scatter: Individual event accuracy
- Rolling mean (window=15): Local accuracy trend
- Reveals whether accuracy degradation is smooth or sudden

## Reproducing Paper Results

To replicate the published benchmarks:

```python
# Load model and data
with open('untrained_hmm.txt', 'r') as f:
    model = Model.read(f)
events = get_events(filenames, model)

# Run 5-fold CV, 10 iterations
accuracies = []
for i in range(10):
    random.shuffle(events)
    accuracies.append(n_fold_cross_validation(events, n=5))
accuracies = np.array(accuracies)

# Key metrics
mean_accuracy_curve = accuracies.mean(axis=0)
top_quartile_accuracy = mean_accuracy_curve[int(len(events) * 0.25)]
# Expected: ~97.7% accuracy at top 26%
```

## Applying to New Models

When testing a new model on this task:

### Minimum Requirements

1. **Implement `analyze_events()`** that returns:
   - `Filter Score`: Confidence that event is on-pathway
   - `Soft Call`: Classification accuracy for single event

2. **Report MCSC curves**, not single accuracy numbers

3. **Use same train/test splits** for fair comparison

### Recommended Comparisons

| Metric | Description | Paper Baseline |
|--------|-------------|----------------|
| MCSC @ 25% | Accuracy on top quarter | ~97.7% |
| MCSC @ 100% | Accuracy on all events | ~75-80% |
| Threshold for 95% | Filter score needed for 95% accuracy | ~0.1 |
| On-pathway % | Events with Filter Score > 0.1 | ~25-30% |

### Ablation Studies

Test model components by:
1. Removing artifact states (oversegmentation, undersegmentation, backslip, flicker)
2. Simplifying to single match state per position
3. Removing label fork validation
4. Varying number of training iterations

## Key Implementation Details

### Baum-Welch Training Parameters

```python
model.train(
    events,
    max_iterations=10,      # Early stopping if converged
    use_pseudocount=True    # Prevent zero probabilities
)
```

### Fresh Model per Fold

**Critical:** Load untrained model for each CV fold to prevent information leakage:

```python
for fold in folds:
    # MUST reload fresh model
    with open('untrained_hmm.txt', 'r') as f:
        model = Model.read(f)
    model = train(model, training_events, threshold=0.1)
```

### Forward-Backward for Scoring

All metrics use expected transition counts from forward-backward algorithm:

```python
trans, ems = model.forward_backward(event)
# trans[i,j] = expected transitions from state i to state j
# ems[t,i] = posterior probability of state i at time t
```

## Summary

| Concept | Purpose | Good Value |
|---------|---------|------------|
| Filter Score | Event quality/confidence | >0.1 for training |
| Soft Call | Single-event accuracy | 0.0-1.0 |
| MCSC | Cumulative accuracy curve | Report at multiple thresholds |
| Training threshold | Exclude off-pathway events | 0.10 |
| CV configuration | Robust evaluation | 5-fold × 10 iterations |

The key insight is that **accuracy should be reported as a function of confidence threshold**, not as a single number. This acknowledges that real-world deployment would use a confidence cutoff, accepting only high-quality predictions.
