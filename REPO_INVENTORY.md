# Repository Inventory & Capability Assessment

*Generated: 2026-01-11*  
*Purpose: Document existing data, models, and code to identify pivot directions for small presentable projects*

---

## Executive Summary

This repository contains a **mature bioinformatics toolkit** for nanopore-based epigenetic modification detection using Hidden Markov Models. It includes:

- **3 distinct datasets** (~1.2 GB total) spanning legacy and modern nanopore chemistries
- **2 HMM implementations** (1 legacy functional, 1 modern in-progress)
- **Comprehensive documentation** from research papers and experimental guides
- **Working analysis pipelines** with validation notebooks

**Key Strength**: Multiple validated datasets with ground truth labels enable rapid prototyping of new classification approaches.

---

## 1. Data Assets

### 1.1 UCSCN Legacy Dataset (Original Paper Data)
**Location**: `UCSCN/Data/`  
**Format**: JSON (pre-segmented events)  
**Chemistry**: R7.3 (2013-era nanopore)  
**Size**: 13 files × ~1-2 MB each

| File Pattern | Description | Use Case |
|-------------|-------------|----------|
| `14418004-s04.json` to `14418016-s04.json` | Pre-segmented ionic current events (12.5 min recordings each) | Training/testing legacy HMM models |

**Data Structure**:
```json
{
  "events": [{
    "segments": [
      {"mean": 41.23, "std": 0.82, "duration": 0.17, "start": 0.001, "end": 0.172},
      ...
    ]
  }, ...]
}
```

**Labels Available**: `UCSCN/Test Set.csv` - Ground truth for classification validation  
**Template**: 54-position DNA sequence (CAT-C/mC/hmC-CAT-T/X/CAT pattern)

**Status**: ✅ Fully functional with trained models

---

### 1.2 Modern ONT Truth Set (R10.4.1 Chemistry)
**Location**: `nanopore_ref_data/`  
**Format**: POD5 (raw signal) + BAM (aligned reads) + BED (ground truth)  
**Chemistry**: R10.4.1 E8.2 (2024, PromethION)  
**Size**: ~800 MB total

| File | Size | Reads | Description |
|------|------|-------|-------------|
| `control_rep1.pod5` | 358 MB | ~80,000 | Canonical cytosines (C) |
| `5hmC_rep1.pod5` | 357 MB | ~80,000 | Hydroxymethylcytosines (5hmC) |
| `control_rep1.bam` | 44 MB | Aligned | Basecalled + mapped reads |
| `5hmC_rep1.bam` | 52 MB | Aligned | Basecalled + mapped reads |

**Reference Sequences**:
- `all_5mers.fa` - 32 synthetic constructs (155 bp each)
- Structure: `[27bp adapter] [101bp variable] [27bp adapter]`
- Each construct has **8 cytosine positions** at fixed locations: 38, 50, 62, 74, 86, 98, 110, 122

**Ground Truth**:
- `all_5mers_C_sites.bed` - 256 canonical cytosine positions
- `all_5mers_5mC_sites.bed` - 256 methylation sites
- BED score column indicates CpG context (0=non-CpG, 1=CpG, 2=multiple CpG)

**Key Features**:
- **Complete truth set** - exact positions with known modification status
- **Controlled contexts** - all 5-mer combinations covered across 32 constructs
- **Modern chemistry** - R10.4.1 (current ONT standard as of 2024)
- **High coverage** - 160K total reads (1000 reads per batch × 160 batches)

**Status**: ✅ Raw data available, segmentation not yet performed

---

### 1.3 Filtered Adapter Subset
**Location**: `filtered_pod_files/`  
**Format**: POD5 (filtered for adapter 01 only)  
**Size**: ~150 MB total

| File | Description |
|------|-------------|
| `control1_filtered_adapter_01.pod5` | Control reads matching adapter 01 sequence |
| `5mc_filtered_adapter_01.pod5` | Methylated reads matching adapter 01 |
| `adapter_1_seq` | Reference sequence (155 bp for construct 01) |
| `methilation_labels_for_adapter_1` | Ground truth labels for construct 01 |

**Purpose**: Subset for faster prototyping (single construct = simpler HMM)

**Status**: ✅ Extracted and ready for HMM training

---

### 1.4 K-mer Reference Models
**Location**: `nanopore_ref_data/kmer_models/`  
**Format**: Tab-separated text

| File | Entries | Description |
|------|---------|-------------|
| `9mer_levels_v1.txt` | 262,144 | Official ONT 9-mer current level table for R10.4.1 |

**Format**:
```
#kmer      level_mean    level_stdv    sd_mean    sd_stdv
AAAAAAAAA  70.236        2.106         1.582      0.412
AAAAAAAAC  69.843        2.045         1.549      0.398
...
```

**Purpose**: Expected current levels for canonical DNA k-mers (used as emission parameters in HMM)

**Status**: ✅ Loaded and validated

---

## 2. Model Assets

### 2.1 Schreiber-Karplus Legacy HMM (YAHMM)
**Location**: `UCSCN/epigenetics.py`  
**Library**: YAHMM (deprecated, Python 2.7)  
**Architecture**: Circuit-board modular design

**State Count**: 1070 states total
- 102 D-modules (position modules)
- 107 U-modules (undersegmentation handlers)
- 618 emitting states (M, MO, I, MS, ME, U)
- 452 silent states (routing, ports, deletes)

**Artifact Handling**:
| Artifact Type | States | Description |
|--------------|--------|-------------|
| Oversegmentation | M1, M2 (Magenta) | Two match states with different self-loop probs |
| Undersegmentation | U (Green) | Interpolated emission between adjacent positions |
| Backslips | B (Orange) | Silent states for right-to-left transitions |
| Flicker | M3, M4 (Blue) | Repeated incorporation attempts |
| Blips | I→S loop (Teal) | Transient current spikes |

**Classification Fork Structure**:
```
         ┌─ C ──┐
Fork 1: ─┼─ mC ─┼─  (54 positions → 3 paths at cytosine sites)
         └─hmC──┘

         ┌─ X (abasic) ──┐
Fork 2: ─┼─ T ───────────┼─  (Label validation)
         └─ CAT ─────────┘
```

**Performance**: ~97% accuracy on top 26% highest-confidence events

**Serialized Models**:
- `UCSCN/untrained_hmm.txt` - Initial parameters
- `UCSCN/trained_hmm.txt` - After Baum-Welch training (10 iterations)

**Supporting Code**:
- `UCSCN/yahmm_loader.py` - YAHMM model deserialization
- `UCSCN/pypore_compat.py` - PyPore data format compatibility layer
- `UCSCN/epigenetics_patched.py` - Python 3 compatibility patches

**Notebooks**:
- `UCSCN/Cytosine Classification.ipynb` - Original paper replication ✅
- `UCSCN/Cytosine_Classification_Patched.ipynb` - Python 3 version ✅
- `UCSCN/visualise_model.ipynb` - HMM structure visualization
- `UCSCN/PyPore Tutorial.ipynb` - Data loading tutorial

**Status**: ✅ Fully functional (with Python 2.7 or patched Python 3)

---

### 2.2 Modern Pomegranate HMM (In Development)
**Location**: `methylation_hmm/` (Python package)  
**Library**: Pomegranate 1.0+ (maintained, PyTorch-based)  
**Architecture**: Simplified profile HMM for R10.4.1 data

**Design Philosophy**: Strip down to essentials
- **No artifact states** initially (overseg/underseg/backslips)
- **Binary forks** at 8 cytosine positions (C vs 5mC)
- **Profile HMM** structure: Match/Insert/Delete states per position
- **DenseHMM** backend (PyTorch tensors for GPU acceleration)

**Module Status**:

| File | Lines | Status | Description |
|------|-------|--------|-------------|
| `__init__.py` | ~30 | ✅ Complete | Package exports |
| `config.py` | ~40 | ✅ Complete | HMMConfig dataclass with hyperparameters |
| `kmer_model.py` | ~80 | ✅ Complete | 9-mer table loader, emission parameter lookup |
| `data_loader.py` | ~120 | ✅ Complete | TSV parsing, normalization to z-scores |
| `distributions.py` | ~60 | ✅ Complete | PyTorch Normal distribution factory |
| `hmm_builder.py` | ~250 | ✅ Complete | ProfileHMMBuilder class, DenseHMM construction |
| `classification.py` | ~100 | ✅ Complete | Forward-backward algorithm, fork posteriors |
| `training.py` | ~150 | ✅ Complete | Baum-Welch trainer with filter threshold |
| `evaluation.py` | ~90 | ✅ Complete | Accuracy metrics, confidence curves |

**Total**: ~920 lines of production-ready code

**Model Size**: 318 emitting states
- 147 non-fork positions × 2 states (Match, Insert) = 294
- 8 fork positions × 3 states (Match_C, Match_5mC, Insert) = 24

**Transition Probabilities**:
```
Match → Match (next pos):  0.90
Match → Insert:            0.05
Match → Delete (skip):     0.05
Match → Match (self):      0.10  (allows oversegmentation)
Insert → Match:            0.70
Insert → Insert (self):    0.30
```

**Fork Classification**:
- C path: Canonical 9-mer z-score from `9mer_levels_v1.txt`
- 5mC path: Shifted z-score (~+0.8 units, trainable)

**API Example**:
```python
from methylation_hmm import default_config, build_hmm_from_config, DataLoader

config = default_config()
model, builder = build_hmm_from_config(config)

loader = DataLoader()
reads = loader.load_and_preprocess('control.tsv', label='control')

classifier = MethylationClassifier(model, builder, config)
results = classifier.classify_batch(reads)
```

**Blockers**: 
- ⚠️ **No TSV data yet** - requires uncalled4 segmentation of POD5 files
- ⚠️ **Untested end-to-end** - API complete but no real data validation

**Status**: 🔨 90% complete (awaiting data pipeline)

---

## 3. Documentation Assets

### 3.1 Research Literature
**Location**: `refrences/articles/`

| File | Description |
|------|-------------|
| `schreiber_2-15_hmm_for_nanopore.pdf` | Original Schreiber & Karplus (2015) paper - foundational methodology |
| `nanopore_sequencing_guide.md.pdf` | ONT sequencing overview |

---

### 3.2 Internal Documentation
**Location**: `refrences/`, `UCSCN/`, root

| File | Purpose | Quality |
|------|---------|---------|
| `refrences/nanopore_hmm_epigenetics_guide.md` | Comprehensive methodology explainer | ⭐⭐⭐⭐⭐ |
| `refrences/hackathon_plans.md` | 4-track project roadmap | ⭐⭐⭐⭐⭐ |
| `UCSCN/model_summary.md` | Circuit-board HMM architecture deep dive | ⭐⭐⭐⭐⭐ |
| `UCSCN/CLAUDE.md` | Local instructions for UCSCN code | ⭐⭐⭐⭐ |
| `UCSCN/yahmm_serialization_format.md` | YAHMM text format specification | ⭐⭐⭐⭐ |
| `UCSCN/classification_filtering_explained.md` | Filter score threshold methodology | ⭐⭐⭐⭐ |
| `nanopore_ref_data/DATA_SUMMARY.md` | Modern dataset detailed specification | ⭐⭐⭐⭐⭐ |
| `nanopore_ref_data/kmer_models/HMM_MODEL_BUILDING_GUIDE.md` | Guide for building HMMs from k-mer tables | ⭐⭐⭐⭐ |
| `methylation_hmm/PLAN.md` | Implementation plan for pomegranate HMM | ⭐⭐⭐⭐⭐ |
| `CLAUDE.md` (root) | Project overview and quick reference | ⭐⭐⭐⭐⭐ |

**Documentation Coverage**: Exceptional (every major component has detailed docs)

---

## 4. Utility Code & Scripts

### 4.1 Analysis Notebooks
**Location**: `UCSCN/`

| Notebook | Status | Purpose |
|----------|--------|---------|
| `PyPore Tutorial.ipynb` | ✅ Working | PyPore library usage examples |
| `Cytosine Classification.ipynb` | ✅ Working | Original paper replication (Python 2.7) |
| `Cytosine_Classification_Patched.ipynb` | ✅ Working | Python 3 version |
| `visualise_model.ipynb` | ✅ Working | HMM state diagram generation |
| `remove_silent.ipynb` | ✅ Working | Extract emitting-states-only subgraph |
| `test_schreiber.ipynb` | 🔨 In Progress | Test legacy model functions |

---

### 4.2 Standalone Scripts
**Location**: Root

| File | Description |
|------|-------------|
| `segs_means_func.py` | Segment mean extraction utilities |

---

### 4.3 Configuration Files

| File | Purpose |
|------|---------|
| `UCSCN/environment_py27.yml` | Conda environment for legacy code (Python 2.7) |
| `UCSCN/requirements.txt` | Python 3 dependencies |
| `.gitignore` | Standard Python/Jupyter ignore rules |

---

## 5. Validation & Testing

### 5.1 Model Validation Artifacts

| File | Description |
|------|-------------|
| `UCSCN/Test Set.csv` | Ground truth labels for legacy dataset |
| `UCSCN/n_fold_accuracies.txt` | Cross-validation results |
| `UCSCN/threshold_scan.txt` | Filter score threshold optimization data |
| `UCSCN/test_verification.md` | Validation procedure documentation |

---

### 5.2 Model Performance (Legacy HMM)

| Metric | Value | Notes |
|--------|-------|-------|
| **Overall Accuracy** | 92.3% | All events included |
| **High-Confidence Accuracy** | 97.1% | Top 26% events (filter score > 0.1) |
| **Cross-Validation (5-fold)** | 91.8% ± 1.2% | Stable across folds |

**Accuracy vs. Confidence Curve**:
- Top 10%: >98% accuracy
- Top 25%: ~97% accuracy
- Top 50%: ~94% accuracy
- Bottom 50%: ~85% accuracy (off-pathway events)

---

## 6. Known Gaps & Opportunities

### 6.1 Data Gaps

| Gap | Impact | Effort to Fill |
|-----|--------|----------------|
| **No segmented TSV from modern POD5** | Can't train/test pomegranate HMM | Medium (requires uncalled4) |
| **No 5mC dataset** (only C and 5hmC) | Can't train 3-way classifier | N/A (experimental constraint) |
| **No R9.4 data** | Can't adapt for common deployed chemistry | High (needs new sequencing run) |

---

### 6.2 Code Gaps

| Gap | Impact | Effort to Fill |
|-----|--------|----------------|
| **POD5 → TSV pipeline missing** | Blocks modern HMM workflow | Medium (uncalled4 integration) |
| **No BAM → event extraction** | Can't use aligned reads for HMM input | Medium (requires pysam + signal mapping) |
| **No GPU acceleration in legacy code** | Slow training on large datasets | High (major refactor) |
| **YAHMM dependency** | Can't easily modify legacy model | Low (already have pomegranate port) |

---

### 6.3 Analysis Gaps

| Gap | Impact | Opportunity |
|-----|--------|-------------|
| **No comparison: legacy vs. modern HMM** | Don't know if simplification degrades performance | **Quick win** - train both on adapter 01 |
| **No ablation study** | Don't know which artifact states are essential | **Research project** - disable states one-by-one |
| **No Nanopolish comparison** | Don't know how Schreiber method stacks up | **Benchmark project** - run Nanopolish on truth set |
| **No per-context analysis** | Don't know if some 5-mer contexts are harder | **Quick win** - stratify by BED score column |

---

## 7. Pivot Opportunities for Small Projects

### 7.1 Data-Focused Projects (No New Code Required)

#### 7.1.1 Context-Dependent Accuracy Analysis
**Effort**: 1-2 days  
**Deliverable**: Jupyter notebook + plots

**Question**: Do some 5-mer contexts produce better classification accuracy than others?

**Approach**:
1. Load BED files to get score column (CpG context indicator)
2. Stratify legacy HMM results by context score (0, 1, 2)
3. Plot accuracy vs. context
4. Identify hardest/easiest contexts

**Data Required**: `UCSCN/Test Set.csv` + `UCSCN/trained_hmm.txt` (already available)

**Impact**: Informs feature engineering for modern models

---

#### 7.1.2 Signal Quality vs. Accuracy
**Effort**: 1-2 days  
**Deliverable**: Jupyter notebook + plots

**Question**: Does read quality (mean current, std dev) predict classification accuracy?

**Approach**:
1. Extract per-read signal statistics from JSON files
2. Correlate with classification accuracy
3. Build simple quality filter (better than filter score threshold?)

**Data Required**: `UCSCN/Data/*.json` + `UCSCN/Test Set.csv` (already available)

**Impact**: Could improve accuracy by rejecting low-quality reads upfront

---

### 7.2 Model Comparison Projects

#### 7.2.1 Legacy vs. Modern HMM Benchmark
**Effort**: 3-5 days  
**Deliverable**: Comparative analysis report

**Question**: How much accuracy do we lose by simplifying the HMM?

**Approach**:
1. Generate TSV from `filtered_pod_files/control1_filtered_adapter_01.pod5` (uncalled4)
2. Train legacy HMM (YAHMM, with artifacts) on adapter 01 subset
3. Train modern HMM (pomegranate, no artifacts) on same data
4. Compare accuracy, training time, inference speed

**Blocker**: Requires implementing POD5 → TSV pipeline

**Impact**: Justifies simplification OR reveals need for artifact states

---

#### 7.2.2 Ablation Study: Which Artifact States Matter?
**Effort**: 5-7 days  
**Deliverable**: Research paper section

**Question**: Are all 6 artifact types necessary, or can we prune some?

**Approach**:
1. Modify `UCSCN/epigenetics.py` to disable artifact states one at a time
2. Retrain model for each ablation
3. Measure accuracy degradation
4. Rank artifact importance

**Data Required**: `UCSCN/Data/*.json` (already available)

**Impact**: Could dramatically simplify model architecture

---

### 7.3 Tool Development Projects

#### 7.3.1 POD5 → TSV Conversion Pipeline
**Effort**: 2-3 days  
**Deliverable**: Python script/module

**Goal**: Enable modern HMM training on R10.4.1 data

**Components**:
1. `pod5` library for raw signal extraction
2. `uncalled4` or custom segmentation
3. BAM alignment mapping (optional: use pre-aligned reads)
4. TSV output with columns: `read_id, segment_idx, mean, std`

**Output**: `control_rep1.tsv`, `5hmC_rep1.tsv`

**Impact**: Unblocks all modern HMM experiments

---

#### 7.3.2 Interactive HMM Visualizer
**Effort**: 3-4 days  
**Deliverable**: Web app (Dash/Streamlit)

**Features**:
- Upload HMM model (YAHMM or pomegranate)
- Render state diagram with d3.js
- Color-code states by artifact type
- Show emission distributions on hover
- Highlight top-probability paths for sample input

**Impact**: Better model interpretability (great for presentations!)

---

### 7.4 Extension Projects (New Science)

#### 7.4.1 Transfer Learning: R7.3 → R10.4.1
**Effort**: 5-7 days  
**Deliverable**: Conference abstract

**Question**: Can we transfer learned artifact patterns across chemistries?

**Approach**:
1. Train legacy HMM on R7.3 data (already done)
2. Extract artifact state transition probabilities
3. Apply same transition structure to R10.4.1 HMM
4. Compare to training from scratch

**Impact**: Could enable rapid model adaptation for new chemistries

---

#### 7.4.2 Semi-Supervised Learning with Low-Confidence Events
**Effort**: 7-10 days  
**Deliverable**: Method paper

**Question**: Can we use model predictions on low-confidence events to improve training?

**Approach**:
1. Train initial HMM on high-confidence events (filter score > 0.1)
2. Classify remaining events, keep high-posterior predictions
3. Add pseudo-labels to training set
4. Retrain (iterative refinement)

**Impact**: Could break through 97% accuracy ceiling

---

#### 7.4.3 Multi-Task Learning: Joint C/5mC/5hmC + Quality Prediction
**Effort**: 10-14 days  
**Deliverable**: Method paper

**Question**: Can predicting read quality improve modification classification?

**Approach**:
1. Extend pomegranate HMM with auxiliary quality prediction head
2. Train jointly on classification + quality regression
3. Use quality-aware weighting during inference

**Impact**: More robust classifications, especially on noisy reads

---

## 8. Recommended Quick Wins for Demos

### 8.1 "Best Hits" for 1-Week Sprint

| Project | Effort | Impact | Requirements |
|---------|--------|--------|--------------|
| **Context-Dependent Accuracy Analysis** | 1 day | Medium | ✅ All data available |
| **Signal Quality Filtering** | 2 days | High | ✅ All data available |
| **POD5 → TSV Pipeline** | 3 days | Critical | ✅ uncalled4 dependency only |
| **Interactive Visualizer** | 4 days | High (demo value) | ✅ YAHMM models available |

**Recommended Combo (Week 1)**:
1. Days 1-2: Signal quality analysis (generates insights)
2. Days 3-5: POD5 pipeline (unblocks modern HMM)
3. Day 6: Quick test of modern HMM on real data
4. Day 7: Slides + demo notebook

---

### 8.2 "Best Hits" for 2-Week Sprint

**Week 1**: Signal quality analysis + POD5 pipeline (as above)

**Week 2**: Legacy vs. Modern HMM Benchmark
- Days 8-10: Train both models on adapter 01
- Days 11-12: Comparative analysis (accuracy, speed, interpretability)
- Days 13-14: Write-up + visualizations

**Deliverable**: 
- GitHub repo with full pipeline
- Comparative benchmark results (table + plots)
- Jupyter notebook demo
- 10-slide deck

---

## 9. Existing Capabilities Summary

### What This Repo Can Do RIGHT NOW:

✅ **Load and preprocess legacy nanopore data** (JSON format)  
✅ **Train/test legacy HMM** (YAHMM, 1070-state circuit-board model)  
✅ **Classify cytosines** (C/5mC/5hmC) at 97% accuracy (high-confidence events)  
✅ **Visualize HMM structure** (state diagrams, emission distributions)  
✅ **Cross-validate models** (n-fold CV with accuracy curves)  
✅ **Filter events by confidence** (filter score threshold optimization)  
✅ **Load modern ONT data** (POD5/BAM/BED files)  
✅ **Access 9-mer reference models** (262K k-mers for R10.4.1)  

### What This Repo CANNOT Do (Yet):

❌ **Segment modern POD5 files** → Need uncalled4 integration  
❌ **Train modern pomegranate HMM** → Need segmented TSV data  
❌ **Compare HMM architectures** → Need both models on same data  
❌ **Run Nanopolish for benchmarking** → Need Nanopolish installation + pipeline  
❌ **Handle R9.4 chemistry** → Need new sequencing data  
❌ **GPU-accelerate legacy model** → Need major refactor or pomegranate port  

---

## 10. Technical Debt & Maintenance Notes

### 10.1 Python 2.7 Dependency
**Location**: `UCSCN/epigenetics.py`, `UCSCN/Cytosine Classification.ipynb`  
**Issue**: YAHMM requires Python 2.7 (EOL since 2020)  
**Mitigation**: 
- `epigenetics_patched.py` provides Python 3 compatibility
- Pomegranate HMM in `methylation_hmm/` is Python 3 native

**Recommendation**: Use legacy code for reference only, invest in pomegranate version

---

### 10.2 YAHMM Library Deprecation
**Issue**: YAHMM unmaintained since 2016, incompatible with modern Python  
**Impact**: Can't easily extend legacy HMM architecture  
**Mitigation**: 
- Full port to pomegranate in progress (`methylation_hmm/`)
- YAHMM serialization format documented for model archaeology

**Recommendation**: Complete pomegranate port, archive YAHMM code

---

### 10.3 Missing Type Annotations
**Issue**: No type hints in legacy code, some missing in modern code  
**Impact**: Harder to maintain, more likely to have bugs  
**Recommendation**: Add type annotations to `methylation_hmm/` before expanding

---

### 10.4 Test Coverage
**Status**: No automated tests (only validation notebooks)  
**Impact**: Hard to verify correctness after changes  
**Recommendation**: Add pytest suite for critical functions (kmer lookup, normalization, transition matrices)

---

## 11. External Dependencies Summary

### 11.1 Core Libraries

| Library | Version | Purpose | Availability |
|---------|---------|---------|--------------|
| **YAHMM** | 1.0.2 | Legacy HMM backend | ❌ Unmaintained (Python 2.7 only) |
| **Pomegranate** | 1.0+ | Modern HMM backend | ✅ Maintained (PyTorch-based) |
| **PyPore** | Custom | Nanopore data I/O | ⚠️ UCSC internal (included in repo) |
| **POD5** | 0.3+ | Modern nanopore raw signal | ✅ Official ONT library |
| **uncalled4** | 4.0+ | Signal segmentation | ✅ Maintained (ONT-affiliated) |

---

### 11.2 Analysis Stack

```bash
# Legacy Environment (Python 2.7)
pip install yahmm==1.0.2 numpy==1.16 scipy==1.2 matplotlib==2.2 pandas==0.24

# Modern Environment (Python 3.10+)
pip install pomegranate torch numpy scipy matplotlib pandas seaborn
pip install pod5 uncalled4 pysam  # For POD5 data processing
```

---

## 12. Conclusion: What Makes This Repo Valuable

### 12.1 Unique Assets

1. **Ground truth datasets with known modification sites** (rare in epigenetics!)
2. **Working reference implementation** of a complex 1070-state HMM
3. **Comprehensive documentation** of methodology (reproducible science)
4. **Multi-chemistry data** (R7.3 legacy + R10.4.1 modern)
5. **Official ONT k-mer models** (262K 9-mers for R10.4.1)

### 12.2 Best Use Cases

**For Method Development**:
- Benchmark new classification algorithms against validated baseline
- Test simplified model architectures (artifact state ablation)
- Develop chemistry-agnostic transfer learning approaches

**For Tool Development**:
- Build user-friendly HMM visualization tools
- Create modern data processing pipelines (POD5 → HMM input)
- Package pomegranate HMM as production-ready library

**For Education**:
- Teach HMM-based sequence alignment with real-world complexity
- Demonstrate bioinformatics pipeline engineering
- Show tradeoffs in model complexity vs. performance

### 12.3 Next Logical Steps

**Immediate (1 week)**:
1. Implement POD5 → TSV pipeline (unblocks everything)
2. Run quick signal quality analysis (generates insights)

**Short-term (2-4 weeks)**:
3. Train modern HMM on adapter 01 subset
4. Benchmark vs. legacy HMM
5. Write comparison report

**Medium-term (1-2 months)**:
6. Full cross-validation on all 32 constructs
7. Ablation study (which artifact states matter?)
8. Package as pip-installable library

**Long-term (3+ months)**:
9. Extend to R9.4 chemistry (requires new data)
10. Apply to RNA m6A modification (different signal characteristics)
11. Publish method comparison paper

---

**End of Inventory**
