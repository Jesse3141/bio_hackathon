# Profile HMM for Nanopore Methylation Detection

> **Related Documentation**: [../README.md](../README.md) | [../RESULTS.md](../RESULTS.md) | [../TESTING_METHODOLOGY.md](../TESTING_METHODOLOGY.md) | [../SIGNAL_ALIGNMENT_SETUP.md](../SIGNAL_ALIGNMENT_SETUP.md)

---

## Progress Tracker

### Phase 1: Code Implementation [COMPLETE]

| Module | Status | Notes |
|--------|--------|-------|
| `__init__.py` | [x] | Package init with all exports |
| `config.py` | [x] | HMMConfig dataclass |
| `kmer_model.py` | [x] | 9mer table loader |
| `data_loader.py` | [x] | TSV parsing, normalization |
| `distributions.py` | [x] | Emission factory |
| `hmm_builder.py` | [x] | DenseHMM construction |
| `classification.py` | [x] | Forward-backward classifier |
| `training.py` | [x] | Baum-Welch training |
| `evaluation.py` | [x] | Metrics & curves |

### Phase 2: Data Preparation [COMPLETE]

| Task | Status | Notes |
|------|--------|-------|
| Run uncalled4 signal alignment | [x] | 2,617 control + 1,274 modified reads aligned |
| Export signal at cytosine positions | [x] | 30,402 measurements at 8 positions |
| Control sample data | [x] | From `control1_filtered_adapter_01.pod5` |
| 5mC sample data | [x] | From `5mc_filtered_adapter_01.pod5` |
| Generate emission parameters | [x] | `output/hmm_emission_params_pomegranate.json` |

### Phase 3: Model Training & Evaluation [COMPLETE]

| Task | Status | Notes |
|------|--------|-------|
| Validate package imports | [x] | All imports working |
| Build likelihood-ratio HMMs | [x] | 8-state control + 8-state modified models |
| Train on labeled data | [x] | Baum-Welch on train split |
| Evaluate accuracy | [x] | **70.9% accuracy** on test set (642 reads) |
| Per-class accuracy | [x] | Control: 72.2%, Modified: 67.9% |

**Key Result**: Achieved 70.9% accuracy on held-out test data using likelihood-ratio classification between two HMMs (control vs modified), based on +36.7 pA signal difference between C and 5mC.

### Phase 4: Classifier Comparison & Cross-Validation [COMPLETE]

| Task | Status | Notes |
|------|--------|-------|
| AUC/ROC plots for simplified classifier | [x] | ROC AUC = 0.759, AP = 0.616 |
| Per-position AUC analysis | [x] | Position 38 best (AUC=0.734), Position 110 worst (AUC=0.561) |
| Fork HMM implementation | [x] | Single model with C/5mC fork paths at each position |
| 5-fold cross-validation | [x] | Both classifiers evaluated |
| Classifier comparison | [x] | Simplified outperforms Fork HMM |

**Cross-Validation Results:**

| Classifier | CV Accuracy | CV AUC |
|------------|-------------|--------|
| Simplified (Two-HMM) | **71.3% ± 2.0%** | **0.772 ± 0.022** |
| Fork HMM (Single Model) | 59.5% ± 1.3% | 0.598 ± 0.015 |

**Key Insight**: The two-HMM likelihood-ratio approach outperforms a single fork-based HMM because:
1. Separate training preserves distinct emission patterns for each class
2. The simplified classifier benefits from Baum-Welch optimizing each model independently
3. Top 26% confidence filtering achieves **85%** accuracy (simplified) vs 61% (fork)

### Phase 5: Schreiber-Karplus Style Evaluation [COMPLETE]

| Task | Status | Notes |
|------|--------|-------|
| Map paper metrics to our models | [x] | Filter Score → confidence, MCSC → MCA |
| Implement SchreiberKarplusEvaluator | [x] | Full evaluation module |
| Run evaluation on both classifiers | [x] | Results in RESULTS.md |
| Generate MCA curve plots | [x] | `output/plots/schreiber_karplus_mca_comparison.png` |

**Schreiber-Karplus Comparison:**

| Metric | Paper Baseline | Simplified | Fork HMM |
|--------|----------------|------------|----------|
| Accuracy @ 25% | **97.7%** | **85.0%** | 60.0% |
| Accuracy @ 50% | ~90% | 81.3% | 59.2% |
| Overall Accuracy | ~75-80% | 70.9% | 56.4% |

**Gap Analysis**: 12.7 percentage point gap from paper's top-25% accuracy due to:
- No label fork validation (T/X/CAT dual signal)
- Different chemistry (R10.4.1 vs R7.3)
- Simpler model (8 states vs ~300+ states with artifact handling)

---

## All Phases Complete

The methylation HMM pipeline is now fully functional with:
- Signal alignment from POD5 files via Dorado + uncalled4
- Trained HMM classifier achieving **70.9% accuracy**
- Saved model: `output/trained_methylation_model.pt`
- 18 passing tests validating the pipeline

**Future improvements** (optional):
- Add more training data to improve accuracy
- Implement confidence-based filtering (top N% most confident predictions)
- Extend to 5hmC detection with three-class classification

---

## Quick Start (Recommended: Simplified Pipeline)

```python
import sys
sys.path.insert(0, '/home/jesse/repos/bio_hackathon')

from methylation_hmm import SimplifiedMethylationClassifier, run_full_pipeline

# Option 1: Train from scratch
classifier, metrics = run_full_pipeline(
    training_csv='output/hmm_training_sequences.csv',
    emission_params_json='output/hmm_emission_params_pomegranate.json',
    output_model='output/trained_methylation_model.pt'
)
print(f"Accuracy: {metrics.accuracy:.1%}")

# Option 2: Load pre-trained model
classifier = SimplifiedMethylationClassifier.from_json(
    'output/hmm_emission_params_pomegranate.json'
)

# Classify new data
import pandas as pd
new_data = pd.read_csv('your_data.csv')  # Must have columns: 38, 50, 62, 74, 86, 98, 110, 122
results = classifier.classify_dataframe(new_data)
for r in results[:5]:
    print(f"{r.read_id}: {r.prediction} (confidence: {r.confidence:.2f})")
```

## Alternative: Full Profile HMM (Original Design)

```python
from methylation_hmm import (
    default_config, build_hmm_from_config,
    DataLoader, MethylationClassifier, BaumWelchTrainer
)

# 1. Build model
config = default_config()
model, builder = build_hmm_from_config(config)

# 2. Load data
loader = DataLoader()
control_reads = loader.load_and_preprocess('control.tsv', label='control')
methyl_reads = loader.load_and_preprocess('methylated.tsv', label='5mC')

# 3. Train (optional)
trainer = BaumWelchTrainer(model, builder, config)
model = trainer.train(control_reads, methyl_reads)

# 4. Classify
classifier = MethylationClassifier(model, builder, config)
results = classifier.classify_batch(control_reads + methyl_reads)
df = classifier.results_to_dataframe(results)
print(df)
```

---

## Overview

Build a simplified Profile HMM to classify cytosines as **C** or **5mC** from nanopore signal data using pomegranate.

**Input**: TSV with columns `(read_id, segment_idx, mean_current, std)`
**Output**: Per-read, per-position classification with confidence scores

---

## Architecture

### State Structure (Simplified)

Each position has 2 emitting states:
- **M_i (Match)**: Normal distribution from 9mer model
- **I_i (Insert)**: Wide Normal for noise/extra segments

Delete transitions handled implicitly via transition matrix.

```
Position i:
    ┌────────────────────────┐
    │  ┌──────┐              │
 ──►│  │ M_i  │──────────────┼──►
    │  └──┬───┘              │
    │     │ (self-loop 0.1)  │
    │  ┌──▼───┐              │
    │  │ I_i  │──────────────┼──►
    │  └──────┘              │
    └────────────────────────┘
```

### Binary Fork at Cytosine Positions

At positions **38, 50, 62, 74, 86, 98, 110, 122**:

```
         ┌── M_C_i ───┐
    ──►──┤            ├──►──
         └── M_5mC_i ─┘
```

Each path has different emission parameters:
- **C path**: Canonical 9mer z-score
- **5mC path**: Shifted z-score (~+0.8 units, refined during training)

### Model Size

- 147 non-fork positions × 2 states = 294 states
- 8 fork positions × 3 states = 24 states
- **Total: 318 emitting states**

---

## File Structure

```
methylation_hmm/
├── __init__.py             # Package exports
├── config.py               # HMMConfig dataclass with hyperparameters
├── kmer_model.py           # Load 9mer_levels_v1.txt, lookup functions
├── distributions.py        # Create Normal distributions for states
├── hmm_builder.py          # Build DenseHMM with transitions/emissions
├── data_loader.py          # Parse TSV, group by read, normalize
├── training.py             # Baum-Welch training loop
├── classification.py       # Forward-backward, fork posteriors
├── evaluation.py           # Accuracy, confusion matrix, curves
├── simplified_pipeline.py  # [NEW] Likelihood-ratio classifier (recommended)
└── tests/
    ├── test_hmm_pipeline.py       # Core HMM tests (12 tests)
    └── test_simplified_pipeline.py # Simplified pipeline tests (6 tests)
```

---

## Implementation Steps

### Step 1: Config & Data Loading

**File: `config.py`**
```python
@dataclass
class HMMConfig:
    reference_path: str
    cytosine_positions: tuple = (38, 50, 62, 74, 86, 98, 110, 122)
    p_match: float = 0.90
    p_insert: float = 0.05
    p_delete: float = 0.05
    p_match_self_loop: float = 0.10
    methylation_shift: float = 0.8
```

**File: `data_loader.py`**
- `load_tsv(path) -> DataFrame`
- `group_by_read(df) -> List[SegmentedRead]`
- `normalize_to_zscore(reads) -> List[SegmentedRead]` - Converts raw pA to z-scores using per-read statistics

**Normalization** (handles raw pA from uncalled4):
```python
def normalize_to_zscore(segments: np.ndarray) -> np.ndarray:
    mean = segments.mean()
    std = segments.std()
    return (segments - mean) / std
```

### Step 2: K-mer Model

**File: `kmer_model.py`**
```python
class KmerModel:
    def __init__(self, model_path: str):
        # Load 262K 9mer -> z-score mappings

    def get_emission_params(self, sequence: str, position: int,
                            modification: str = 'C') -> Tuple[float, float]:
        # Returns (mean, std) for Normal distribution
```

**Source**: `/home/jesse/repos/bio_hackathon/nanopore_ref_data/kmer_models/9mer_levels_v1.txt`

### Step 3: HMM Construction

**File: `hmm_builder.py`**
```python
class ProfileHMMBuilder:
    def build_model(self, sequence: str) -> DenseHMM:
        # 1. Create emission distributions for each position
        # 2. Build transition matrix (n_states x n_states)
        # 3. Set start/end probabilities
        # 4. Return DenseHMM

    def get_fork_state_indices(self) -> Dict[int, Dict[str, int]]:
        # Maps position -> {'C': idx, '5mC': idx}
```

**Transition Probabilities**:
| From | To | Probability |
|------|-----|-------------|
| Entry | Match | 0.90 |
| Entry | Insert | 0.05 |
| Entry | Skip (next) | 0.05 |
| Match | Exit | 0.90 |
| Match | Match (self) | 0.10 |
| Insert | Exit | 0.70 |
| Insert | Insert (self) | 0.30 |

### Step 4: Classification

**File: `classification.py`**
```python
class MethylationClassifier:
    def classify_read(self, segments: Tensor) -> List[ClassificationResult]:
        # 1. Run forward-backward
        # 2. Extract expected transitions at each fork
        # 3. Compute P(C) and P(5mC) for each position
        # 4. Return calls with confidence
```

**Fork Posterior Calculation**:
```
P(5mC at pos i) = Σ ξ(t, j→k) for all transitions into 5mC_i states
P(C at pos i) = Σ ξ(t, j→k) for all transitions into C_i states
Normalize: P(C) + P(5mC) = 1
```

### Step 5: Training

**File: `training.py`**
```python
class BaumWelchTrainer:
    def train(self, control_data: Tensor, methylated_data: Tensor) -> DenseHMM:
        # 1. Compute filter scores for all reads
        # 2. Keep reads with score > 0.1
        # 3. Run Baum-Welch for 10 iterations
        # 4. Update emission parameters
```

**Filter Score** = Product of posterior flow through all 8 forks
**Threshold** = 0.1 (keeps ~26% highest-confidence reads)

### Step 6: Evaluation

**File: `evaluation.py`**
- `accuracy_at_threshold(predictions, labels, threshold)`
- `accuracy_vs_confidence_curve()` - Replicate paper's Figure 4
- `per_position_accuracy()` - Check all 8 positions

---

## Key Files to Reference

| File | Purpose |
|------|---------|
| `nanopore_ref_data/kmer_models/9mer_levels_v1.txt` | 9mer emission model |
| `nanopore_ref_data/all_5mers.fa` | Reference sequences |
| `filtered_pod_files/adapter_1_seq` | 155bp reference for adapter 01 |
| `filtered_pod_files/methilation_labels_for_adapter_1` | Ground truth labels |
| `UCSCN/epigenetics.py` | Reference for fork structure pattern |

---

## Verification Plan

1. **Unit Tests**
   - Load 9mer model, verify 262K entries
   - Build HMM for short sequence, check state count
   - Verify fork structure at expected positions

2. **Integration Test**
   - Generate synthetic reads with known labels
   - Run classification, verify >90% accuracy

3. **End-to-End Test**
   - Load real TSV from control/5mC POD5
   - Train model
   - Evaluate on held-out reads
   - Target: ~97% accuracy on top 26% confidence reads

---

## Dependencies

```bash
pip install pomegranate torch pandas numpy
```

Pomegranate v1.0+ required for DenseHMM API with:
- `forward_backward()` returning expected transitions
- `fit()` for Baum-Welch training
- Torch tensor-based distributions
