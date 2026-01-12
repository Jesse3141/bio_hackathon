# Evaluation Scripts for Methylation HMM Classifiers

## Overview

Create CLI scripts to evaluate full-sequence HMM classifiers that model the entire 155bp reference sequence. The key variable is **emission parameter source**: single adapter (context-specific) vs all adapters (pooled).

## Why Full-Sequence Models Only

As documented in `data_processing.ipynb`:

| Data Format | Positions | Use Case |
|-------------|-----------|----------|
| **Sparse** | 8 cytosines only | Simple Gaussian classification (not evaluated here) |
| **Full Sequence** | ~155 bases | HMM that models transitions + handles artifacts |

The full-sequence HMM:
- Observes current at EVERY position (not just cytosines)
- Has match states for all positions with forks at cytosine sites
- Can model oversegmentation, undersegmentation, backslips via artifact states
- Uses Viterbi decoding over entire path

## Emission Parameter Comparison

From `data_processing.ipynb` (Position 38 example):

| Source | C Mean | C Std | Δ (5mC-C) | Notes |
|--------|--------|-------|-----------|-------|
| **Single adapter** (adapter_01) | 897.5 pA | 78.8 pA | +42.9 pA | Context-specific, lower variance |
| **All adapters** (pooled) | 866.9 pA | 106.2 pA | +26.8 pA | Higher variance from context mixing |

**Key insight**: Single-adapter models have ~30% lower std, potentially better separation.

## Models to Evaluate

| # | Model | Architecture | Mode | Emission Params Source |
|---|-------|--------------|------|------------------------|
| 1 | FullSequenceHMM | Profile HMM with forks | Binary | Single adapter (adapter_01) |
| 2 | FullSequenceHMM | Profile HMM with forks | Binary | All adapters (pooled) |
| 3 | FullSequenceHMM | Profile HMM with forks | 3-way | Single adapter (adapter_01) |
| 4 | FullSequenceHMM | Profile HMM with forks | 3-way | All adapters (pooled) |

The **only difference** between models 1 & 2 (and 3 & 4) is where emission parameters come from:
- **Single adapter**: Lower variance, context-specific μ and σ
- **All adapters**: Higher variance due to pooled 5-mer contexts

## Evaluation Metrics (for ALL models)

- Overall accuracy
- Per-class accuracy (C, 5mC, 5hmC)
- Accuracy by site type: non_cpg (0), cpg (1), homopolymer (2)
- Accuracy at top K confidence: 25%, 50%, 100%
- Confusion matrix
- 5-fold cross-validation with mean/std

---

## Implementation Plan

### Phase 1: Create Full-Sequence HMM Classifier

#### 1.1 FullSequenceHMM Class

**File:** `methylation_hmm/full_sequence_hmm.py`

A single HMM class that models the entire 155bp sequence with forks at cytosine positions.

```python
class FullSequenceHMM:
    """
    Profile HMM for full 155bp sequence classification.

    Architecture (from data_processing.ipynb):
    - Match states for ALL 155 positions
    - Fork states at 8 cytosine positions (38, 50, 62, 74, 86, 98, 110, 122)
    - Insert states for artifact handling

    States per position:
    - Non-fork: 2 states (Match + Insert)
    - Fork (binary): 3 states (C + mC + Insert)
    - Fork (3-way): 4 states (C + mC + hmC + Insert)
    """

    def __init__(self,
                 n_classes: int = 2,  # 2 = binary (C/mC), 3 = 3-way (C/mC/hmC)
                 emission_params_source: str = 'single'):  # 'single' or 'pooled'
        pass

    def fit(self, sequences: List[np.ndarray], labels: np.ndarray):
        """Train on full-length sequences (~155 observations each)."""
        pass

    def predict(self, sequences: List[np.ndarray]) -> np.ndarray:
        """Classify sequences via Viterbi decoding at fork positions."""
        pass

    def predict_proba(self, sequences: List[np.ndarray]) -> np.ndarray:
        """Return posterior probabilities for each class."""
        pass

    @classmethod
    def from_emission_params(cls, params_file: str, source: str = 'single'):
        """Load emission params from JSON (single adapter or pooled)."""
        pass
```

#### 1.2 Data Extraction for Full Sequences

**File:** `methylation_hmm/data_extraction.py`

Extract full-length sequences from BAM + POD5 (as shown in `data_processing.ipynb`).

```python
def extract_full_sequences(
    bam_file: str,
    pod5_file: str,
    max_reads: int = None
) -> List[Dict]:
    """
    Extract mean current at EVERY base position for each read.

    Returns list of:
    {
        'read_id': str,
        'ref_name': str,
        'currents': np.ndarray,  # ~155 values
        'n_positions': int
    }
    """
    pass
```

#### 1.3 Emission Parameter Computation

**File:** `methylation_hmm/emission_params.py`

Compute emission parameters from either single adapter or all adapters.

```python
def compute_emission_params(
    signal_df: pd.DataFrame,
    adapter_filter: str = None,  # None = all adapters, else 'adapter_01'
    mode: str = 'binary'  # 'binary' or '3way'
) -> Dict:
    """
    Compute Gaussian emission parameters for HMM states.

    Returns dict with μ, σ for each (position, modification_state) pair.
    """
    pass
```

---

### Phase 2: Create Unified Evaluation Framework

#### 2.1 Core Metrics Module

**File:** `methylation_hmm/evaluation/framework.py`

```python
@dataclass
class UnifiedMetrics:
    overall_accuracy: float
    per_class_accuracy: Dict[str, float]
    accuracy_by_site_type: Dict[str, float]  # {'non_cpg': ..., 'cpg': ..., 'homopolymer': ...}
    accuracy_at_top_25pct: float
    accuracy_at_top_50pct: float
    accuracy_at_100pct: float
    confusion_matrix: np.ndarray
    auc_macro: float
    cv_mean_accuracy: Optional[float]
    cv_std_accuracy: Optional[float]

class UnifiedEvaluator:
    def evaluate(self, classifier, test_df, bed_df) -> UnifiedMetrics
    def cross_validate(self, classifier_class, df, n_folds=5) -> UnifiedMetrics
```

#### 2.2 Site Type Metrics

**File:** `methylation_hmm/evaluation/site_type_metrics.py`

```python
def compute_accuracy_by_site_type(
    y_true, y_pred, positions_with_types: Dict[int, int]
) -> Dict[str, float]
```

Uses BED file site_type scores: 0=non_cpg, 1=cpg, 2=homopolymer

#### 2.3 Output Formatters

**File:** `methylation_hmm/evaluation/output_formatters.py`

- `save_json(metrics, path)` - structured JSON
- `save_csv(metrics, path)` - flat CSV for aggregation
- `generate_plots(metrics, output_dir)` - confusion matrix heatmap, MCA curve

---

### Phase 3: Create CLI Scripts

#### 3.1 Individual Evaluation Script

**File:** `methylation_hmm/cli/run_evaluation.py`

```bash
python run_evaluation.py \
  --mode binary|3way \
  --emission-source single|pooled \
  --output results/evaluation/ \
  --cv-folds 5
```

Arguments:
- `--mode`: binary (C/5mC) or 3way (C/5mC/5hmC)
- `--emission-source`: single (adapter_01 only) or pooled (all adapters)
- `--output`: Directory for results
- `--cv-folds`: Cross-validation folds (default 5)
- `--seed`: Random seed (default 42)

#### 3.2 Master Evaluation Script

**File:** `methylation_hmm/cli/run_all_evaluations.py`

Runs all 4 model configurations and generates comparison report:
```bash
python run_all_evaluations.py --output results/full_evaluation/
```

Configurations run:
1. `binary_single` - Binary mode, single adapter emission params
2. `binary_pooled` - Binary mode, pooled emission params
3. `3way_single` - 3-way mode, single adapter emission params
4. `3way_pooled` - 3-way mode, pooled emission params

Outputs:
- `results/full_evaluation/comparison_table.md` - summary table
- `results/full_evaluation/{config_name}/metrics.json` - per-config results
- `results/full_evaluation/{config_name}/confusion_matrix.png`
- `results/full_evaluation/{config_name}/mca_curve.png`

---

### Phase 4: Output Specifications

#### JSON Output Format (`metrics.json`)

```json
{
  "model": "FullSequenceHMM",
  "mode": "binary",
  "emission_source": "single",
  "emission_params_from": "5mers_rand_ref_adapter_01",
  "sequence_length": 155,
  "overall": {"accuracy": 0.707, "n_samples": 5000},
  "per_class": {"C": 0.72, "5mC": 0.69},
  "by_site_type": {"non_cpg": 0.75, "cpg": 0.68, "homopolymer": 0.62},
  "confidence_stratified": {
    "top_25pct": 0.89,
    "top_50pct": 0.81,
    "top_100pct": 0.707
  },
  "confusion_matrix": [[1800, 700], [765, 1735]],
  "cross_validation": {
    "mean_accuracy": 0.706,
    "std_accuracy": 0.011,
    "fold_accuracies": [0.70, 0.71, 0.69, 0.72, 0.71]
  },
  "emission_params_summary": {
    "mean_delta_5mC_C": 42.9,
    "mean_std_C": 78.8,
    "mean_std_5mC": 80.3
  }
}
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `methylation_hmm/full_sequence_hmm.py` | FullSequenceHMM classifier (models entire 155bp) |
| `methylation_hmm/data_extraction.py` | Extract full sequences from BAM + POD5 |
| `methylation_hmm/emission_params.py` | Compute emission params (single vs pooled) |
| `methylation_hmm/evaluation/__init__.py` | Package init |
| `methylation_hmm/evaluation/framework.py` | UnifiedMetrics, UnifiedEvaluator |
| `methylation_hmm/evaluation/site_type_metrics.py` | Site-type accuracy computation |
| `methylation_hmm/evaluation/output_formatters.py` | JSON/CSV/plot generation |
| `methylation_hmm/cli/__init__.py` | Package init |
| `methylation_hmm/cli/run_evaluation.py` | Single config evaluation CLI |
| `methylation_hmm/cli/run_all_evaluations.py` | All 4 configs evaluation CLI |

## Files to Modify

| File | Change |
|------|--------|
| `methylation_hmm/__init__.py` | Export FullSequenceHMM and evaluation module |

---

## Verification

1. **Unit tests**: Add tests in `methylation_hmm/tests/test_evaluation.py`
2. **Integration test**: Run single config evaluation, verify output format
3. **End-to-end**: Run `run_all_evaluations.py` and check all 4 config results

```bash
# Test single evaluation (binary mode, single adapter emission params)
python methylation_hmm/cli/run_evaluation.py \
  --mode binary --emission-source single \
  --output /tmp/test_eval

# Verify output
cat /tmp/test_eval/metrics.json
ls /tmp/test_eval/*.png

# Run full evaluation suite (all 4 configurations)
python methylation_hmm/cli/run_all_evaluations.py \
  --output results/full_evaluation/
```

---

## Implementation Order

1. Phase 1.1: Create FullSequenceHMM class (core classifier)
2. Phase 1.2: Create data extraction utilities
3. Phase 1.3: Create emission parameter computation (single vs pooled)
4. Phase 2.1-2.3: Create evaluation framework
5. Phase 3.1: Create single evaluation CLI
6. Phase 3.2: Create master evaluation script
7. Run all 4 configurations and generate comparison results

---

## Expected Hypothesis

Based on `data_processing.ipynb` analysis:

| Configuration | Expected Accuracy | Reason |
|---------------|-------------------|--------|
| Binary + Single | **Highest** | Lower variance (σ≈79 pA), larger Δ (+43 pA) |
| Binary + Pooled | Medium | Higher variance (σ≈106 pA), smaller Δ (+27 pA) |
| 3-way + Single | Medium | 5hmC/C overlap still problematic |
| 3-way + Pooled | Lowest | Both higher variance and 5hmC/C overlap |

**Key question to answer**: Does context-specific training (single adapter) significantly outperform pooled training?
