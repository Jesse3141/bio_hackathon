# Full-Sequence HMM Evaluation Results

Generated: 2026-01-13 10:49:47

## Overview

This report compares Full-Sequence HMM classifiers that model the entire 155bp
reference sequence. The key variable is **emission parameter source**:
- **Single adapter**: Context-specific parameters (lower variance)
- **Pooled**: Parameters from all adapters (higher variance)

## Summary Table

| Configuration | Mode | Emission | Overall | 95% CI | Top 25% | AUC | p-value |
|---------------|------|----------|---------|--------|---------|-----|---------|
| 3way_single | 3way | single | 52.0% | [48.7%, 55.2%] | 54.5% | 0.674 | 3.45e-30 |

## Best Configuration

**3way_single** achieved the highest overall accuracy: 52.0%

## Per-Class Accuracy

| Configuration | C | 5mC | 5hmC |
|---------------|---|-----|------|
| 3way_single | 57.6% | 64.1% | 25.4% |

## Key Findings

## Confidence-Stratified Analysis

| Configuration | Top 25% | Top 50% | All |
|---------------|---------|---------|-----|
| 3way_single | 54.5% | 56.6% | 52.0% |

This shows accuracy when only considering the most confident predictions.

## Accuracy by Site Type

Per-position accuracy grouped by cytosine context:
- **non_cpg**: C not followed by G
- **cpg**: CpG dinucleotide
- **homopolymer**: CC run (adjacent cytosines)

| Configuration | non_cpg | 95% CI | cpg | 95% CI | homopolymer | 95% CI |
|---------------|---------|--------|-----|--------|-------------|--------|
| 3way_single | 45.1% | [43.6%, 46.5%] | 36.6% | [34.4%, 38.9%] | 43.5% | [40.3%, 46.8%] |

## Cross-Validation Results

| Configuration | Mean Acc | Std |
|---------------|----------|-----|
| 3way_single | 53.6% | 2.6% |
