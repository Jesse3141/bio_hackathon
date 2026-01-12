## Schreiber-Karplus Style Evaluation Results

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

### Per-Class Performance

| Metric | Simplified | Fork HMM |
|--------|------------|----------|
| Control Accuracy | 72.2% | 55.2% |
| Modified Accuracy | 67.9% | 59.1% |

### Interpretation

The paper achieves 97.7% accuracy on the top 26% most confident predictions.
Our simplified classifier achieves **85.0%** on the top 25%.

**Gap Analysis:**
- Paper top-25% accuracy: 97.7%
- Our top-25% accuracy: 85.0%
- Gap: 12.7 percentage points

**Possible reasons for the gap:**
1. Different chemistry (R10.4.1 vs older R7.3)
2. Different reference constructs (synthetic 5mers vs phi29 polymerase)
3. No independent label validation fork (T/X/CAT in paper)
4. Simpler model architecture (8 states vs ~300+ states)