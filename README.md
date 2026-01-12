# Nanopore HMM-Based Methylation Detection

A bioinformatics toolkit for detecting cytosine modifications (C, 5mC, 5hmC) from Oxford Nanopore sequencing data using Hidden Markov Models, based on the Schreiber & Karplus methodology.

---

## Primary Experiment: Full-Sequence HMM Classification

The main experiment evaluates **Full-Sequence HMMs** that model the entire 155bp reference sequence (not just cytosine positions). The key variable is **emission parameter source**: single adapter (context-specific) vs pooled (all adapters).

### Data

| File | Description | Size |
|------|-------------|------|
| `output/rep1/signal_full_sequence.csv` | Mean current at ALL 155 positions per read | 15.5M rows, 1.7 GB |
| `output/rep1/signal_at_cytosines_3way.csv` | Mean current at 8 cytosine positions only | 1.0M rows |
| `nanopore_ref_data/all_5mers_C_sites.bed` | Cytosine positions with site-type annotations | 256 sites |

**Source samples:**
- `control` (62,014 reads) - Canonical cytosine
- `5mC` (33,298 reads) - 5-methylcytosine
- `5hmC` (29,943 reads) - 5-hydroxymethylcytosine

See [nanopore_ref_data/DATA_SUMMARY.md](nanopore_ref_data/DATA_SUMMARY.md) for complete data documentation.

### Model Architecture

The `FullSequenceHMM` class models the **entire 155bp sequence** with fork states at cytosine positions:

```
Position 0-37:   Match states (single state per position)
Position 38:     Fork states (C, 5mC, [5hmC] branches)  ← Cytosine
Position 39-49:  Match states
Position 50:     Fork states                            ← Cytosine
...
Position 122:    Fork states                            ← Cytosine
Position 123-154: Match states

Total states: 163 (binary) or 171 (3-way)
```

**Key files:**

| File | Purpose |
|------|---------|
| `methylation_hmm/full_sequence_hmm.py` | `FullSequenceHMM` classifier implementation |
| `methylation_hmm/emission_params.py` | Emission parameter computation (single vs pooled) |
| `methylation_hmm/cli/run_evaluation.py` | CLI for single configuration evaluation |
| `methylation_hmm/cli/run_all_evaluations.py` | CLI to run all 4 configurations |

### Evaluation Configurations

| # | Config Name | Mode | Emission Source | Expected Accuracy |
|---|-------------|------|-----------------|-------------------|
| 1 | `binary_single` | C vs 5mC | Single adapter | **Highest** (lower variance) |
| 2 | `binary_pooled` | C vs 5mC | All adapters | Medium |
| 3 | `3way_single` | C vs 5mC vs 5hmC | Single adapter | Medium |
| 4 | `3way_pooled` | C vs 5mC vs 5hmC | All adapters | Lowest |

**Hypothesis:** Single-adapter emission parameters have ~30% lower standard deviation, leading to better class separation.

### Running the Evaluation

```bash
# Activate environment
conda activate nanopore

# Run single configuration
python methylation_hmm/cli/run_evaluation.py \
    --mode binary \
    --emission-source single \
    --cv-folds 5 \
    --output results/full_evaluation

# Run all 4 configurations
python methylation_hmm/cli/run_all_evaluations.py \
    --output results/full_evaluation \
    --cv-folds 5
```

### Output Structure

```
results/full_evaluation/
├── binary_single/
│   ├── metrics.json          # Full evaluation metrics
│   ├── metrics.csv           # Flat CSV format
│   ├── emission_params.json  # Gaussian parameters used
│   ├── confusion_matrix.png
│   └── mca_curve.png
├── binary_pooled/
│   └── ...
├── 3way_single/
│   └── ...
├── 3way_pooled/
│   └── ...
├── summary.csv               # All configs in one CSV
└── comparison_table.md       # Markdown comparison report
```

### Metrics Collected

- **Overall accuracy** and per-class accuracy
- **Confidence-stratified accuracy** (top 25%, 50%, 100%)
- **AUC-ROC** (macro-averaged)
- **Confusion matrix**
- **5-fold cross-validation** (mean ± std)
- **Accuracy by site type** (non-CpG, CpG, homopolymer)

---

## Quick Start

```bash
# Test the full-sequence HMM
python -c "
from methylation_hmm import FullSequenceHMM, compute_emission_params_from_full_csv

# Compute emission params from single adapter
params = compute_emission_params_from_full_csv(
    'output/rep1/signal_full_sequence.csv',
    adapter='5mers_rand_ref_adapter_01',
    mode='binary'
)
print(f'Mean Δ(5mC-C): {params.summary()[\"mean_delta_5mC_C\"]:.1f} pA')

# Build and use classifier
hmm = FullSequenceHMM.from_emission_params(params)
print(f'Model has {hmm.n_states} states')
"
```

---

## Key Results

| Configuration | Overall Accuracy | Top 25% | Top 50% | AUC | CV Accuracy |
|---------------|------------------|---------|---------|-----|-------------|
| **binary_single** | **71.0%** | **85.0%** | 83.5% | 0.779 | 70.9% ± 2.1% |
| binary_pooled | 63.3% | 76.6% | 71.0% | 0.684 | 63.3% ± 0.4% |
| 3way_single | 52.0% | 54.5% | 56.6% | 0.674 | 53.6% ± 2.6% |
| 3way_pooled | 46.9% | 48.5% | 50.3% | 0.615 | 46.9% ± 0.3% |

### Key Findings

1. **Single-adapter beats pooled by 7.7%** in binary mode (71.0% vs 63.3%)
2. **Top 25% confidence achieves 85% accuracy** — confidence filtering is effective
3. **3-way classification is harder** due to 5hmC/C signal overlap (only +8.9 pA vs +37 pA for 5mC)
4. **Cross-validation confirms results** — low std indicates stable performance

### Per-Class Accuracy

| Configuration | C | 5mC | 5hmC |
|---------------|---|-----|------|
| binary_single | 74.4% | 64.0% | — |
| binary_pooled | 64.1% | 61.8% | — |
| 3way_single | 57.6% | 64.1% | 25.4% |
| 3way_pooled | 54.8% | 56.9% | 19.1% |

**Why 5hmC is hard:** The signal difference is only +8.9 pA (vs C), compared to +37 pA for 5mC. This creates significant overlap in the emission distributions.

---

## Repository Structure

```
bio_hackathon/
├── README.md                           # This file
├── methylation_hmm/                    # Main HMM package
│   ├── full_sequence_hmm.py            # ★ Full-sequence classifier
│   ├── emission_params.py              # ★ Single vs pooled params
│   ├── cli/
│   │   ├── run_evaluation.py           # ★ Single config CLI
│   │   └── run_all_evaluations.py      # ★ All configs CLI
│   ├── evaluation/                     # Metrics framework
│   ├── simplified_pipeline.py          # Legacy sparse classifier
│   ├── generic_hmm.py                  # Legacy 3-way classifier
│   └── tests/
│
├── output/rep1/                        # Generated data
│   ├── signal_full_sequence.csv        # ★ 15.5M measurements
│   ├── signal_at_cytosines_3way.csv    # Sparse (8 positions)
│   ├── hmm_3way_circuit_board.json     # Pre-computed params
│   └── *_signal_aligned.bam            # Signal-aligned reads
│
├── nanopore_ref_data/                  # Truth set
│   ├── all_5mers.fa                    # Reference (32 constructs)
│   ├── all_5mers_C_sites.bed           # Cytosine positions
│   ├── control_rep1.pod5               # Raw signal - canonical C
│   ├── 5mC_rep1.pod5                   # Raw signal - methylated
│   ├── 5hmC_rep1.pod5                  # Raw signal - hydroxymethylated
│   └── DATA_SUMMARY.md                 # ★ Data documentation
│
├── results/                            # Evaluation outputs
│   └── full_evaluation/                # ★ Full-sequence HMM results
│
└── docs/                               # Additional documentation
```

---

## Documentation Index

| File | Description |
|------|-------------|
| **[nanopore_ref_data/DATA_SUMMARY.md](nanopore_ref_data/DATA_SUMMARY.md)** | Complete data documentation (POD5, BAM, CSV formats) |
| **[results/train_scripts/EVALUATION_PLAN.md](results/train_scripts/EVALUATION_PLAN.md)** | Full-sequence evaluation plan |
| **[SIGNAL_ALIGNMENT_SETUP.md](SIGNAL_ALIGNMENT_SETUP.md)** | Pipeline: POD5 → Dorado → uncalled4 → BAM |
| **[CLAUDE.md](CLAUDE.md)** | Project overview for AI assistants |

---

## Dependencies

```bash
conda create -n nanopore python=3.10 -y
conda activate nanopore
pip install pomegranate torch pandas numpy scipy scikit-learn matplotlib seaborn
pip install pod5 pysam
```

---

## References

- Schreiber & Karplus (2015) - Original HMM-based epigenetic classification methodology
- [ONT Open Data](https://42basepairs.com/browse/s3/ont-open-data/modbase-validation_2024.10) - Truth set source

---

## License

Research/educational use. See individual component licenses.
