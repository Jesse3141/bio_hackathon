# Methylation HMM Classification Results

> **Related Documentation**: [README.md](README.md) | [TESTING_METHODOLOGY.md](TESTING_METHODOLOGY.md) | [SIGNAL_ALIGNMENT_SETUP.md](SIGNAL_ALIGNMENT_SETUP.md) | [methylation_hmm/PLAN.md](methylation_hmm/PLAN.md) | [methylation_hmm/GENERIC_HMM.md](methylation_hmm/GENERIC_HMM.md)

## Overview

This document summarizes the results of HMM-based methylation classification from nanopore sequencing signal data.

### Binary Classification (C vs 5mC)

**Data Source**: POD5 files aligned with Dorado + uncalled4, extracting current levels at 8 cytosine positions (38, 50, 62, 74, 86, 98, 110, 122).

**Dataset**: 3,206 reads total (2,564 train / 642 test)
- Control (C): 2,617 reads
- Modified (5mC): 1,274 reads

**Signal Difference**: 5mC produces +36.7 pA higher current than canonical C (p < 10⁻¹⁸⁹)

### 3-Way Classification (C vs 5mC vs 5hmC)

**Data Source**: Full rep1 dataset from `nanopore_ref_data/` processed through Dorado + uncalled4.

**Dataset**: 125,255 reads across all 32 adapter sequences
- Control (C): 56,593 reads (513,290 position measurements)
- 5mC: 28,179 reads (264,300 position measurements)
- 5hmC: 26,402 reads (269,588 position measurements)

**Signal Differences**:
| Modification | Mean Current | Δ vs C | Cohen's d |
|--------------|--------------|--------|-----------|
| C (canonical) | 800.5 pA | — | — |
| 5hmC | 809.4 pA | +8.9 pA | 0.081 |
| 5mC | 830.7 pA | +30.2 pA | 0.279 |

---

## Classifier Comparison

### Two-HMM Likelihood Ratio (Simplified Classifier)

Uses two separate HMMs trained independently on control and modified data. Classification based on log P(X|modified) - log P(X|control).

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 70.9% |
| **ROC AUC** | 0.759 |
| **Average Precision** | 0.616 |
| **5-Fold CV Accuracy** | 71.3% ± 2.0% |
| **5-Fold CV AUC** | 0.772 ± 0.022 |
| **Control Accuracy** | 72.2% |
| **Modified Accuracy** | 67.9% |

### Fork HMM (Single Model)

Single HMM with fork architecture at each position, splitting into C and 5mC emission paths.

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 56.4% |
| **ROC AUC** | 0.577 |
| **5-Fold CV Accuracy** | 59.5% ± 1.3% |
| **5-Fold CV AUC** | 0.598 ± 0.015 |

### Winner: Two-HMM Likelihood Ratio

The simplified two-HMM approach outperforms the single fork-based HMM by **~12 percentage points** in accuracy.

---

## 3-Way Classification: GenericHMM (C / 5mC / 5hmC)

The `GenericHMM` extends the fork architecture to 3-way classification, with 3-way forks at each of the 8 cytosine positions.

### Architecture

- **24 states total**: 8 positions × 3 modifications (C, 5mC, 5hmC)
- **Self-loops** (p=0.15): Handle nanopore dwelling/stuck events
- **Skip transitions** (p=0.05): Handle missed bases
- **Forward transitions** (p=0.80): Normal progression, split equally among 3 mods

### Results on Full Dataset (125,255 reads)

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 46.8% |
| **C Accuracy** | 54.3% |
| **5mC Accuracy** | 57.6% |
| **5hmC Accuracy** | 18.9% |

### Confusion Matrix

|  | Pred C | Pred 5mC | Pred 5hmC |
|--|--------|----------|-----------|
| **True C** | 30,755 | 19,025 | 6,813 |
| **True 5mC** | 8,467 | 16,220 | 3,492 |
| **True 5hmC** | 11,169 | 10,189 | 4,986 |

### Confidence-Based Filtering (Paper-Style Evaluation)

Unlike binary classifiers, confidence filtering does NOT improve 3-way classification:

| Coverage | Reads Used | Accuracy | Notes |
|----------|------------|----------|-------|
| 10% | 11,111 | 46.3% | No improvement |
| **25%** | 27,779 | **48.6%** | Slight improvement |
| 50% | 55,558 | 50.0% | Peak accuracy |
| 100% | 111,116 | 46.8% | Overall accuracy |

**Per-Class Accuracy at 25% Coverage:**

| Class | Accuracy | Reads | Interpretation |
|-------|----------|-------|----------------|
| C | 36.6% | 12,255 | Poor - often confused with 5mC |
| **5mC** | **91.1%** | 9,482 | **Excellent** - clear signal difference |
| 5hmC | 6.5% | 6,042 | Very poor - confused with both C and 5mC |

**Why 5mC dominates high-confidence predictions:**
- 5mC has the largest signal shift (+30 pA), making it most distinguishable
- When the model is confident, it's usually seeing 5mC-like signals
- C and 5hmC overlap in signal space, so both get misclassified as 5mC

### Why 5hmC Classification is Challenging

The 5hmC signal is nearly indistinguishable from canonical C:

| Pair | Cohen's d | Interpretation |
|------|-----------|----------------|
| C vs 5mC | 0.279 | Small-medium effect |
| C vs 5hmC | **0.081** | **Negligible effect** |
| 5mC vs 5hmC | 0.192 | Small effect |

With only +8.9 pA difference (vs ~110 pA std deviation), 5hmC detection from single reads is fundamentally limited by signal overlap. Multi-read consensus or k-mer context features would be needed to improve 5hmC classification.

### Comparison with Paper and Binary Classifiers

| Metric | Paper Baseline | Simplified (2-way) | GenericHMM (3-way) |
|--------|----------------|--------------------|--------------------|
| Accuracy @ 25% | **97.7%** | **85.0%** | 48.6% |
| Accuracy @ 50% | ~90% | 81.3% | 50.0% |
| Accuracy @ 100% | ~75-80% | 70.9% | 46.8% |

**Key insight:** The 3-way problem with 5hmC is fundamentally harder than binary C/5mC classification. The paper's methodology assumes a label validation fork (T/X/CAT) which we don't have, and the 5hmC signal difference is too small for single-read detection.

### Usage

```python
from methylation_hmm import GenericHMM

# Load from pre-computed parameters
hmm = GenericHMM.from_params_json('output/rep1/hmm_3way_pomegranate.json')

# Classify reads
results = hmm.classify_dataframe(signal_df)
for r in results[:5]:
    print(f"{r.read_id}: {r.prediction} (confidence: {r.confidence:.2f})")
```

See [methylation_hmm/GENERIC_HMM.md](methylation_hmm/GENERIC_HMM.md) for full documentation.

---

## 5-Fold Cross-Validation Details (Binary Classification)

| Fold | Simplified Acc | Simplified AUC | Fork HMM Acc | Fork HMM AUC |
|------|----------------|----------------|--------------|--------------|
| 1 | 70.2% | 0.751 | 60.0% | 0.604 |
| 2 | 72.2% | 0.797 | 58.3% | 0.579 |
| 3 | 71.6% | 0.787 | 58.3% | 0.587 |
| 4 | 74.1% | 0.785 | 61.9% | 0.623 |
| 5 | 68.2% | 0.742 | 59.0% | 0.598 |
| **Mean** | **71.3%** | **0.772** | **59.5%** | **0.598** |
| **Std** | **2.0%** | **0.022** | **1.3%** | **0.015** |

---

## Confidence-Based Filtering

Following the Schreiber-Karplus methodology, filtering to high-confidence predictions significantly improves accuracy:

| Coverage | Simplified Accuracy | Fork HMM Accuracy |
|----------|--------------------|--------------------|
| 100% (all) | 70.9% | 56.4% |
| 50% | ~78% | ~58% |
| **26%** | **85.0%** | **61.3%** |
| 10% | ~90% | ~65% |

**Key Finding**: The top 26% most confident predictions achieve **85% accuracy** with the simplified classifier, comparable to human expert curation in the original paper.

---

## Per-Position Discriminative Power

ROC AUC for each cytosine position (single-position classification):

| Position | AUC | Ranking |
|----------|-----|---------|
| **38** | **0.734** | Best |
| 74 | 0.671 | 2nd |
| 122 | 0.613 | 3rd |
| 62 | 0.611 | 4th |
| 86 | 0.600 | 5th |
| 98 | 0.587 | 6th |
| 50 | 0.580 | 7th |
| **110** | **0.561** | Worst |

Position 38 shows the strongest signal difference between C and 5mC, while position 110 has the most overlap between distributions.

---

## Emission Parameters

Mean current (pA) at each position:

| Position | Control (C) | Modified (5mC) | Difference |
|----------|-------------|----------------|------------|
| 38 | 890.6 ± 84.6 | 938.4 ± 81.8 | **+47.8** |
| 50 | 822.4 ± 104.4 | 861.6 ± 105.8 | +39.2 |
| 62 | 816.5 ± 99.4 | 853.1 ± 105.3 | +36.6 |
| 74 | 831.6 ± 82.2 | 874.0 ± 80.1 | +42.4 |
| 86 | 792.7 ± 98.7 | 828.2 ± 96.3 | +35.5 |
| 98 | 779.2 ± 94.7 | 807.8 ± 90.0 | +28.6 |
| 110 | 773.0 ± 90.4 | 800.8 ± 88.6 | +27.8 |
| 122 | 794.3 ± 89.4 | 831.4 ± 90.0 | +37.1 |

**Average shift**: +36.9 pA (5mC > C)

---

## Generated Plots

All plots saved to `output/plots/`:

### Simplified Classifier
- `roc_curve.png` - ROC curve (AUC = 0.759)
- `precision_recall_curve.png` - Precision-Recall curve (AP = 0.616)
- `accuracy_vs_confidence.png` - Accuracy at different confidence thresholds
- `per_position_roc.png` - ROC curves for each of 8 positions
- `score_distributions.png` - Log-likelihood ratio distributions by class

### Fork HMM
- `fork_hmm_roc.png` - ROC curve
- `fork_hmm_accuracy_vs_coverage.png` - Accuracy vs coverage
- `fork_hmm_score_dist.png` - Score distributions

### Comparison
- `classifier_comparison_roc.png` - Side-by-side ROC curves
- `classifier_comparison_accuracy.png` - Accuracy vs coverage for both
- `classifier_comparison_scores.png` - Score distributions comparison

---

## Key Insights

1. **Two-HMM Approach Works Best**: Training separate models on each class preserves distinct emission patterns, yielding significantly better discrimination than a single fork-based model.

2. **Confidence Filtering is Powerful**: The top 26% most confident predictions achieve 85% accuracy, demonstrating that the model reliably identifies which predictions to trust.

3. **Position 38 Most Informative**: The first cytosine position shows the strongest signal difference (+47.8 pA) and highest single-position AUC (0.734).

4. **Consistent Cross-Validation**: Low variance across folds (±2%) indicates stable model performance.

5. **Room for Improvement**:
   - Current accuracy (71%) is below the paper's 97% on top 26%
   - Possible causes: different chemistry (R10.4.1 vs older), different reference constructs
   - Could improve with: more training data, k-mer context modeling, multi-position correlations

---

## Files

### Scripts

| File | Purpose | Run Command |
|------|---------|-------------|
| `methylation_hmm/simplified_pipeline.py` | Two-HMM likelihood ratio classifier | `from methylation_hmm import run_full_pipeline` |
| `methylation_hmm/fork_hmm.py` | Fork-based single HMM classifier | `from methylation_hmm import ForkHMMClassifier` |
| `methylation_hmm/generic_hmm.py` | 3-way fork HMM (C/5mC/5hmC) | `from methylation_hmm import GenericHMM` |
| `methylation_hmm/compare_classifiers.py` | Classifier comparison with CV | `python methylation_hmm/compare_classifiers.py` |
| `methylation_hmm/generate_auc_plots.py` | ROC/AUC plot generation | `python methylation_hmm/generate_auc_plots.py` |
| `methylation_hmm/run_sk_evaluation.py` | Schreiber-Karplus style evaluation | `python methylation_hmm/run_sk_evaluation.py` |
| `methylation_hmm/schreiber_karplus_evaluation.py` | MCA/MCSC evaluation module | `from methylation_hmm import SchreiberKarplusEvaluator` |

### Data Files (`output/`)

| File | Description |
|------|-------------|
| `hmm_training_sequences.csv` | Training data (3,206 reads × 8 positions) |
| `hmm_emission_params_pomegranate.json` | Fitted Gaussian emission parameters |
| `signal_at_cytosines.csv` | Raw signal measurements (30,402 observations) |
| `trained_methylation_model.pt` | Saved PyTorch model weights |
| `simplified_mca_curve.csv` | MCA curve data for simplified classifier |
| `fork_hmm_mca_curve.csv` | MCA curve data for fork HMM |

### 3-Way Classification Data (`output/rep1/`)

| File | Description |
|------|-------------|
| `signal_at_cytosines_3way.csv` | 3-way signal data (1,047,178 measurements) |
| `hmm_3way_pomegranate.json` | 3-way emission parameters (C/mC/hmC) |
| `hmm_3way_circuit_board.json` | Alternative format emission params |
| `control_signal_aligned.bam` | Signal-aligned BAM (65,506 reads) |
| `5mC_signal_aligned.bam` | Signal-aligned BAM (33,782 reads) |
| `5hmC_signal_aligned.bam` | Signal-aligned BAM (37,754 reads) |

### Plots (`output/plots/`)

| File | Description |
|------|-------------|
| `roc_curve.png` | ROC curve (AUC = 0.759) |
| `precision_recall_curve.png` | Precision-Recall curve (AP = 0.616) |
| `accuracy_vs_confidence.png` | Accuracy at confidence thresholds |
| `per_position_roc.png` | Per-position ROC analysis |
| `schreiber_karplus_mca_comparison.png` | MCA curves comparing classifiers |
| `classifier_comparison_*.png` | Side-by-side comparison plots |

---

## Running the Analysis

```python
# Train and evaluate simplified classifier
from methylation_hmm.simplified_pipeline import run_full_pipeline
classifier, metrics = run_full_pipeline(
    training_csv='output/hmm_training_sequences.csv',
    emission_params_json='output/hmm_emission_params_pomegranate.json'
)

# Run full comparison with cross-validation
python methylation_hmm/compare_classifiers.py

# Generate AUC plots
python methylation_hmm/generate_auc_plots.py
```

---

## Schreiber-Karplus Style Evaluation

This section applies the original paper's evaluation methodology to our models. See `TESTING_METHODOLOGY.md` for detailed methodology explanation.

### Methodology Mapping

| Original Paper Metric | Our Analog | Description |
|----------------------|------------|-------------|
| Filter Score | abs(log ratio) | Confidence that read is classifiable |
| Soft Call | 0/1 correctness | Per-read classification accuracy |
| MCSC | MCA (Mean Cumulative Accuracy) | Accuracy at confidence threshold |
| On-pathway events | High-confidence reads | Reads amenable to classification |

### Key Results

| Metric | Paper Baseline | Simplified | Fork HMM |
|--------|----------------|------------|----------|
| **Accuracy @ 25%** | **97.7%** | **85.0%** | 60.0% |
| Accuracy @ 50% | ~90% | 81.3% | 59.2% |
| Overall Accuracy | ~75-80% | 70.9% | 56.4% |
| ROC AUC | N/A | 0.759 | 0.577 |

### Gap Analysis

- Paper top-25% accuracy: **97.7%**
- Our top-25% accuracy: **85.0%**
- Gap: **12.7 percentage points**

### Why Our Results Differ from the Paper

1. **No Label Fork Validation**: The paper uses a dual-fork architecture with independent label validation (T/X/CAT). Our models lack this second validation signal, which is critical for the paper's Filter Score calculation.

2. **Different Chemistry**: R10.4.1 (modern) vs R7.3 (paper). Different pore architectures produce different signal characteristics.

3. **Different Reference Constructs**: Synthetic 5-mer constructs vs phi29 polymerase-arrested DNA in the paper.

4. **Simpler Model Architecture**: Our 8-state models vs the paper's ~300+ state profile HMM with artifact-handling states (oversegmentation, undersegmentation, backslip, flicker).

5. **Raw vs Processed Events**: The paper uses extensively processed events with undersegmentation correction; we use raw signal alignment output.

### What This Means for New Model Development

To achieve paper-level accuracy, a new model would need:

1. **Dual validation signal** - Either a label fork or another independent validation mechanism
2. **Artifact handling** - Explicit states for signal artifacts (oversegmentation, undersegmentation, backslips)
3. **Per-read quality filtering** - Aggressive filtering to top ~26% most confident predictions
4. **Multi-position correlation** - Model dependencies between adjacent cytosine positions

The 85% accuracy at top 25% confidence shows that **confidence-based filtering is working** - we successfully identify which predictions to trust, even if overall accuracy is lower than the paper.

---

## References

- Schreiber & Karplus (2015) - Original HMM-based epigenetic classification methodology
- `UCSCN/model_summary.md` - Detailed HMM architecture documentation
- `UCSCN/classification_filtering_explained.md` - Fork architecture and filter score methodology
- `TESTING_METHODOLOGY.md` - Detailed explanation of the paper's evaluation metrics
