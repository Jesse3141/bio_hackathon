# Full-Sequence HMM Evaluation Results

Generated: 2026-01-13 10:47:44

## Overview

This report compares Full-Sequence HMM classifiers that model the entire 155bp
reference sequence. The key variable is **emission parameter source**:
- **Single adapter**: Context-specific parameters (lower variance)
- **Pooled**: Parameters from all adapters (higher variance)

## Summary Table

| Configuration | Mode | Emission | Overall | 95% CI | Top 25% | AUC | p-value |
|---------------|------|----------|---------|--------|---------|-----|---------|
| binary_single | binary | single | 71.0% | [67.5%, 74.2%] | 85.0% | 0.779 | 6.74e-29 |
| binary_pooled | binary | pooled | 63.3% | [62.6%, 64.0%] | 76.6% | 0.684 | 8.83e-299 |
| 3way_single | 3way | single | 52.0% | [48.7%, 55.2%] | 54.5% | 0.674 | 3.45e-30 |
| 3way_pooled | 3way | pooled | 46.9% | [46.3%, 47.6%] | 48.5% | 0.615 | 0.00e+00 |

## Best Configuration

**binary_single** achieved the highest overall accuracy: 71.0%

## Per-Class Accuracy

| Configuration | C | 5mC | 5hmC |
|---------------|---|-----|------|
| binary_single | 74.4% | 64.0% | N/A |
| binary_pooled | 64.1% | 61.8% | N/A |
| 3way_single | 57.6% | 64.1% | 25.4% |
| 3way_pooled | 54.8% | 56.9% | 19.1% |

## Key Findings

### Binary Mode
- Single adapter: 71.0%
- Pooled: 63.3%
- **Difference: 7.7% (single is better)**

### 3-Way Mode
- Single adapter: 52.0%
- Pooled: 46.9%
- **Difference: 5.0% (single is better)**

## Confidence-Stratified Analysis

| Configuration | Top 25% | Top 50% | All |
|---------------|---------|---------|-----|
| binary_single | 85.0% | 83.5% | 71.0% |
| binary_pooled | 76.6% | 71.0% | 63.3% |
| 3way_single | 54.5% | 56.6% | 52.0% |
| 3way_pooled | 48.5% | 50.3% | 46.9% |

This shows accuracy when only considering the most confident predictions.

## Accuracy by Site Type

Per-position accuracy grouped by cytosine context:
- **non_cpg**: C not followed by G
- **cpg**: CpG dinucleotide
- **homopolymer**: CC run (adjacent cytosines)

| Configuration | non_cpg | 95% CI | cpg | 95% CI | homopolymer | 95% CI |
|---------------|---------|--------|-----|--------|-------------|--------|
| binary_single | 61.8% | [60.2%, 63.5%] | 56.9% | [54.3%, 59.5%] | 60.0% | [56.3%, 63.6%] |
| binary_pooled | 55.6% | [55.3%, 55.9%] | 52.9% | [52.4%, 53.4%] | 53.4% | [52.7%, 54.1%] |
| 3way_single | 45.1% | [43.6%, 46.5%] | 36.6% | [34.4%, 38.9%] | 43.5% | [40.3%, 46.8%] |
| 3way_pooled | 40.8% | [40.5%, 41.1%] | 40.6% | [40.2%, 41.1%] | 40.6% | [40.0%, 41.2%] |

## Cross-Validation Results

Cross-validation was not run.
