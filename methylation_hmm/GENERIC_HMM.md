# GenericHMM: 3-Way Cytosine Modification Classifier

> **Related**: [PLAN.md](PLAN.md) | [RESULTS.md](../RESULTS.md) | [DATA_SUMMARY.md](../nanopore_ref_data/DATA_SUMMARY.md)

## Overview

The `GenericHMM` is a single-model HMM classifier for 3-way cytosine modification detection:
- **C** - Canonical cytosine
- **5mC** - 5-methylcytosine
- **5hmC** - 5-hydroxymethylcytosine

Unlike the binary classifiers (`SimplifiedMethylationClassifier`, `ForkHMMClassifier`), GenericHMM handles all three modification states in a single model with 3-way forks at each cytosine position.

---

## Architecture

### State Structure

```
Total: 24 emitting states (8 positions × 3 modifications)

Position 0 (ref pos 38):   C=0,   5mC=1,   5hmC=2
Position 1 (ref pos 50):   C=3,   5mC=4,   5hmC=5
Position 2 (ref pos 62):   C=6,   5mC=7,   5hmC=8
Position 3 (ref pos 74):   C=9,   5mC=10,  5hmC=11
Position 4 (ref pos 86):   C=12,  5mC=13,  5hmC=14
Position 5 (ref pos 98):   C=15,  5mC=16,  5hmC=17
Position 6 (ref pos 110):  C=18,  5mC=19,  5hmC=20
Position 7 (ref pos 122):  C=21,  5mC=22,  5hmC=23
```

### Fork Structure

At each position, all three modification states are parallel alternatives:

```
                    ┌── M_C ────┐
From prev pos ──►───┤── M_5mC ──├───►── To next pos
                    └── M_5hmC ─┘
```

### Transition Probabilities

| Transition | Probability | Purpose |
|------------|-------------|---------|
| Self-loop | 0.15 | Handle nanopore dwelling/stuck on base |
| Forward (to i+1) | 0.80 | Normal progression to next position |
| Skip (to i+2) | 0.05 | Handle skipped/missed bases |

At each forward/skip transition, probability splits equally among the 3 modification states (÷3).

### Emission Distributions

Each state has a Gaussian emission distribution with parameters learned from data:

| Modification | Mean Current | Std Dev | Signal Shift |
|--------------|--------------|---------|--------------|
| C | 800.5 pA | 108.2 pA | baseline |
| 5hmC | 809.4 pA | 113.5 pA | +8.9 pA |
| 5mC | 830.7 pA | 108.8 pA | +30.2 pA |

**Key insight**: 5hmC is nearly indistinguishable from C (Cohen's d = 0.081).

---

## Usage

### Load Pre-trained Model

```python
from methylation_hmm import GenericHMM

# Load from emission parameters JSON
hmm = GenericHMM.from_params_json('output/rep1/hmm_3way_pomegranate.json')

# Classify a DataFrame
import pandas as pd
df = pd.read_csv('output/rep1/signal_at_cytosines_3way.csv')
results = hmm.classify_dataframe(df)

for r in results[:5]:
    print(f"{r.read_id}: {r.prediction} (confidence: {r.confidence:.2f})")
```

### Evaluate on Labeled Data

```python
metrics = hmm.evaluate(df)

print(f"Overall accuracy: {metrics.accuracy:.1%}")
print(f"Per-class accuracy:")
for mod, acc in metrics.per_class_accuracy.items():
    print(f"  {mod}: {acc:.1%}")
print(f"\nConfusion matrix:\n{metrics.confusion_matrix}")
```

### Train from CSV

```python
# Build from signal CSV (computes emission params automatically)
hmm, test_df = GenericHMM.from_training_csv(
    'output/rep1/signal_at_cytosines_3way.csv',
    test_split=0.2
)

# Evaluate
metrics = hmm.evaluate(test_df)
```

### Classify Wide-Format Data

If your data has columns `38`, `50`, `62`, `74`, `86`, `98`, `110`, `122`:

```python
# Wide format: one row per read, columns are positions
wide_df = df.pivot_table(
    index='read_id',
    columns='position',
    values='mean_current'
).reset_index()

results = hmm.classify_dataframe(wide_df)
```

### Save/Load Model

```python
# Save trained model
hmm.save('output/generic_hmm_trained.pt')

# Load model
hmm = GenericHMM.load('output/generic_hmm_trained.pt')
```

---

## Results

### Overall Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | 46.8% |
| C Accuracy | 54.3% |
| 5mC Accuracy | 57.6% |
| 5hmC Accuracy | 18.9% |

### Confusion Matrix

|  | Pred C | Pred 5mC | Pred 5hmC |
|--|--------|----------|-----------|
| **True C** | 30,755 | 19,025 | 6,813 |
| **True 5mC** | 8,467 | 16,220 | 3,492 |
| **True 5hmC** | 11,169 | 10,189 | 4,986 |

### Paper-Style Confidence Filtering

Unlike binary classifiers, confidence filtering does NOT significantly improve 3-way accuracy:

| Coverage | Accuracy | Notes |
|----------|----------|-------|
| 10% | 46.3% | No improvement |
| 25% | 48.6% | Slight improvement |
| 50% | 50.0% | Peak accuracy |
| 100% | 46.8% | Overall accuracy |

**Per-class accuracy at 25% (high-confidence) threshold:**
- **C**: 36.6% - often confused with 5mC
- **5mC**: 91.1% - excellent due to clear signal difference
- **5hmC**: 6.5% - very poor, overlaps with C and 5mC

### Why 5hmC is Hard to Classify

The signal difference between C and 5hmC is minimal:

| Comparison | Δ Current | Cohen's d | Effect Size |
|------------|-----------|-----------|-------------|
| C vs 5mC | +30.2 pA | 0.279 | Small-medium |
| C vs 5hmC | +8.9 pA | **0.081** | **Negligible** |
| 5mC vs 5hmC | -21.3 pA | 0.192 | Small |

With ~110 pA standard deviation, the +8.9 pA 5hmC shift is within noise. Single-read classification cannot reliably distinguish C from 5hmC.

---

## API Reference

### GenericHMM Class

```python
class GenericHMM:
    """3-way fork HMM for C/5mC/5hmC classification."""

    # Class constants
    POSITIONS = [38, 50, 62, 74, 86, 98, 110, 122]
    MODIFICATIONS = ['C', '5mC', '5hmC']

    # Transition probabilities
    P_SELF_LOOP = 0.15
    P_SKIP = 0.05
    P_FORWARD = 0.80
```

### Methods

| Method | Description |
|--------|-------------|
| `from_params_json(path)` | Load from emission parameters JSON |
| `from_training_csv(path, test_split)` | Build from signal CSV |
| `predict(X)` | Predict class indices (0=C, 1=5mC, 2=5hmC) |
| `predict_proba(X)` | Get (n_samples, 3) probability matrix |
| `classify_dataframe(df)` | Classify from DataFrame, return results |
| `evaluate(df)` | Evaluate on labeled DataFrame |
| `fit(X, y)` | Train with Baum-Welch (optional) |
| `save(path)` | Save model to file |
| `load(path)` | Load model from file |

### ClassificationResult

```python
@dataclass
class ClassificationResult:
    read_id: str
    prediction: str        # 'C', '5mC', or '5hmC'
    prediction_idx: int    # 0, 1, or 2
    log_probs: Dict[str, float]
    confidence: float      # max_prob - second_max_prob
```

### EvaluationMetrics

```python
@dataclass
class EvaluationMetrics:
    accuracy: float
    per_class_accuracy: Dict[str, float]
    confusion_matrix: np.ndarray
    n_samples: int
    n_correct: int
```

---

## Data Requirements

### Input Format

The classifier accepts DataFrames in two formats:

**Long format** (from signal extraction):
```
sample,chrom,position,read_id,mean_current,std_current,dwell_time,n_samples
control,adapter_01,38,uuid-1234,866.5,105.2,0.0024,12
5mC,adapter_01,38,uuid-5678,893.2,108.1,0.0028,14
```

**Wide format** (one row per read):
```
read_id,38,50,62,74,86,98,110,122
uuid-1234,866.5,816.7,794.8,799.9,789.8,777.7,777.7,779.3
```

### Pre-computed Parameters

Emission parameters are stored in `output/rep1/hmm_3way_pomegranate.json`:

```json
{
  "type": "pomegranate_hmm_emissions",
  "modification_states": ["C", "mC", "hmC"],
  "positions": [
    {
      "position": 38,
      "states": {
        "C": {"distribution": "NormalDistribution", "parameters": {"mean": 866.86, "std": 106.23}},
        "mC": {"distribution": "NormalDistribution", "parameters": {"mean": 893.65, "std": 108.18}},
        "hmC": {"distribution": "NormalDistribution", "parameters": {"mean": 856.94, "std": 110.00}}
      }
    },
    ...
  ]
}
```

---

## Comparison with Other Classifiers

| Classifier | Classes | Architecture | Accuracy |
|------------|---------|--------------|----------|
| SimplifiedMethylationClassifier | 2 (C/5mC) | Two separate HMMs | **71.3%** |
| ForkHMMClassifier | 2 (C/5mC) | Single HMM, 2-way forks | 59.5% |
| **GenericHMM** | **3 (C/5mC/5hmC)** | Single HMM, 3-way forks | 46.8% |

The lower accuracy of GenericHMM is expected due to:
1. Harder 3-way classification problem
2. 5hmC signal overlap with C
3. Single-model fork architecture (vs two-HMM likelihood ratio)

---

## Future Improvements

To improve 5hmC classification:

1. **Three-HMM Likelihood Ratio**: Build separate HMMs for C, 5mC, 5hmC and use pairwise likelihood ratios
2. **Multi-read Consensus**: Aggregate predictions across multiple reads covering the same position
3. **K-mer Context Features**: Use surrounding sequence context to improve discrimination
4. **Hierarchical Classification**: First classify C vs modified, then 5mC vs 5hmC
5. **Confidence Filtering**: Focus on high-confidence predictions only

---

## Files

| File | Description |
|------|-------------|
| `generic_hmm.py` | Main GenericHMM class implementation |
| `output/rep1/hmm_3way_pomegranate.json` | Pre-computed emission parameters |
| `output/rep1/signal_at_cytosines_3way.csv` | Training/evaluation data |
| `output/generic_hmm_mca_curve.csv` | MCA curve data for paper-style evaluation |
| `docs/plans/2026-01-12-generic-hmm-design.md` | Design document |
