# Full-Sequence HMM Evaluation Results

Generated: 2026-01-12 11:49:57

## Overview

This report compares Full-Sequence HMM classifiers that model the entire 155bp
reference sequence. The key variable is **emission parameter source**:
- **Single adapter**: Context-specific parameters (lower variance)
- **Pooled**: Parameters from all adapters (higher variance)

## Summary Table

| Configuration | Mode | Emission | Overall | Top 25% | Top 50% | AUC | CV Acc |
|---------------|------|----------|---------|---------|---------|-----|--------|
| binary_single | binary | single | 71.0% | 85.0% | 83.5% | 0.779 | 70.9% |
| binary_pooled | binary | pooled | 63.3% | 76.6% | 71.0% | 0.684 | 63.3% |
| 3way_single | 3way | single | 52.0% | 54.5% | 56.6% | 0.674 | 53.6% |
| 3way_pooled | 3way | pooled | 46.9% | 48.5% | 50.3% | 0.615 | 46.9% |

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

| Configuration | non_cpg | cpg | homopolymer |
|---------------|---------|-----|-------------|
| binary_single | 61.8% | 56.9% | 60.0% |
| binary_pooled | 55.6% | 52.9% | 53.4% |
| 3way_single | 45.1% | 36.6% | 43.5% |
| 3way_pooled | 40.8% | 40.6% | 40.6% |

## Cross-Validation Results

| Configuration | Mean Acc | Std |
|---------------|----------|-----|
| binary_single | 70.9% | 2.1% |
| binary_pooled | 63.3% | 0.4% |
| 3way_single | 53.6% | 2.6% |
| 3way_pooled | 46.9% | 0.3% |
