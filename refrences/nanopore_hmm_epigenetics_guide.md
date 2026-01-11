# Nanopore HMMs for Epigenetic Marker Detection

A practical guide to understanding and implementing the Schreiber & Karplus methodology for detecting epigenetic modifications in nanopore sequencing data.

---

## 1. Nanopore Sequencing: Sample to Aligned Output (Brief Overview)

### The Pipeline

```
Sample Prep → Translocation → Raw Signal → Segmentation → Basecalling → Alignment
```

**1. Library Preparation**
- DNA/RNA extracted, fragmented if needed
- Adapters ligated (Y-adapters with motor protein binding sites)
- Motor protein (helicase for modern ONT; Φ29 DNAP in this paper) controls strand passage

**2. Translocation Through the Pore**
- Motor enzyme ratchets strand through nanopore one nucleotide at a time
- DNA moves at ~450 bp/s (R9.4) or ~400 bp/s (R10.4); enzyme-controlled stepping in this paper
- The constriction zone (~4-6 nt) determines current—this is the k-mer that the signal depends on

**3. Raw Signal Acquisition**
- Ionic current sampled at ~4 kHz (DNA) or ~3 kHz (RNA)
- Current measured in picoamperes (pA), typically 15-90 pA range
- Each k-mer in the constriction produces a characteristic current level + noise

**4. Signal Processing**
- **Event detection**: Identify regions of stable current (>500 μs, <90 pA, >0 pA)
- **Segmentation**: Split events into discrete "steps" corresponding to individual k-mer positions
- Output: sequence of (mean_current, std_dev, duration) tuples

**5. Basecalling / Alignment**
- **Modern approach**: Neural networks (Guppy, Dorado, Bonito) trained on labeled data
- **Classic approach**: HMMs with k-mer pore models (Nanopolish, signalAlign)
- For known sequences: align segments to reference HMM (this paper's approach)

**6. Post-processing**
- Map basecalled reads to reference genome (minimap2)
- Modification calling overlaid on alignment

### Key Insight for Modification Detection

Modified bases alter the current signature of k-mers containing them. The same 5-mer `CGCGA` produces different distributions depending on whether the C is cytosine, 5mC, or 5hmC. This is the foundation for all nanopore epigenetics.

---

## 2. Deep Dive: The Schreiber & Karplus HMM Methodology

### 2.1 Why HMMs for Nanopore?

Profile HMMs are the natural formalism for sequence alignment when:
- You have a reference "profile" (expected sequence/signal)
- Observations are noisy and may have insertions/deletions
- You need probabilistic inference over alignment paths

Traditional bioinformatics HMMs (PFAM, HMMER) emit **categorical** distributions over amino acids/nucleotides. Nanopore HMMs emit **continuous** distributions (Gaussians over current in pA).

### 2.2 Core HMM Concepts

**States and Emissions**

| State Type | Emission | Purpose |
|------------|----------|---------|
| Match (M) | Gaussian(μ, σ) | Expected current for position i |
| Insert (I) | Uniform(0, 90 pA) | Unexpected segments (noise, blips) |
| Delete (D) | Silent (no emission) | Skipped positions |

**The Fundamental Alignment Equation**

Given observations `X = (x₁, ..., xₜ)` and model parameters θ:
- **Viterbi**: Find most likely state path `argmax_π P(X, π | θ)`
- **Forward-Backward**: Compute posterior `P(πᵢ = s | X, θ)` for all positions

### 2.3 The Modular Architecture

The paper's key architectural innovation is decomposing the HMM into **reusable modules** with standardized "ports" (silent entry/exit states).

```
┌─────────── Module for Position i ───────────┐
│                                              │
│   S1─┬─→ M1 ──┐                              │
│   S2─┼─→ M2 ──┼─→ SE ──┬─→ E1               │
│   S3─┤   ↑    │        ├─→ E2               │
│   ...│   D ───┤        │   ...              │
│   S7─┴─→ I ───┴────────┴─→ E7               │
│                                              │
└──────────────────────────────────────────────┘
```

**Benefits of modular design:**
1. **Separation of concerns**: Internal transition probabilities independent of fork structure
2. **Easy branching**: Classification forks only require `2nm` edges (n ports, m paths)
3. **Software optimization**: Silent states automatically removed by YAHMM

### 2.4 Modeling Nanopore/Enzyme Artifacts

The paper adds five artifact-handling mechanisms beyond standard profile HMMs:

#### (a) Oversegmentation (Magenta: M1, M2)
**Problem**: Segmenter splits one dwell into multiple segments
**Solution**: Two match states with different self-loop probabilities
- M1: Low self-loop (handles single extra segment)  
- M2: High self-loop (handles bursts of oversegmentation)

```
Both emit same Gaussian initially, but M1 → M1 has low prob, M2 → M2 has high prob
```

#### (b) Undersegmentation (Green: U)
**Problem**: Segmenter fails to split where enzyme actually stepped
**Solution**: Dedicated state with emission parameters interpolated from adjacent positions

```python
# Pseudocode for undersegmentation state
U_mean = (M[i].mean + M[i+1].mean) / 2
U_std = sqrt((M[i].std**2 + M[i+1].std**2) / 2)
```

#### (c) Backslips (Orange: B)
**Problem**: Enzyme slips backward under voltage tension
**Solution**: Silent state enabling right-to-left transitions

```
Normal: S1 → M → E1 → S1_next
Backslip: S1 → B_prev → M_prev → E1_prev (backwards!)
```
Transition probability decreases exponentially with slip distance.

#### (d) Flicker (Blue: M3, M4)
**Problem**: Enzyme repeatedly attempts/fails nucleotide incorporation
**Solution**: Additional match states for flickering with previous (M3) or next (M4) position

#### (e) Blip Handling (Teal: I→S loop)
**Problem**: Transient current spikes that return to same level
**Solution**: Insert state can transition back to pre-match silent state

### 2.5 The Fork Structure for Classification

The full model has **three forks**, each with three paths:

```
                    ┌─ C ──┐
Unzipping Fork: ────┼─ mC ─┼────
                    └─ hmC─┘
                    
                    ┌─ C ──┐
Synthesis Fork: ────┼─ mC ─┼────  ← Used for classification
                    └─ hmC─┘
                    
                    ┌─ X (abasic) ──┐
Label Fork:     ────┼─ T ──────────┼────  ← Used for validation
                    └─ CAT ────────┘
```

**Scoring mechanism:**

1. Run forward-backward to get transition matrix T
2. Path score: `Sₚ = min_{i∈p} Σₛ T[s, SEᵢ]` (minimum expected transitions to match states)
3. Filter score: `F = Σc Sc × Σl Sl` (product of cytosine and label scores)
4. Accuracy: `A = (Sc, SmC, ShmC) · (ST, SCAT, SX) / F`

### 2.6 Training Protocol

1. **Initialization**: 27 hand-curated events establish Gaussian parameters
2. **Preprocessing**: Iteratively split undersegmented regions using Viterbi alignment
3. **Filter threshold**: Cross-validation showed F=0.1 optimal for separating on/off-pathway
4. **Baum-Welch**: 10 iterations with edge pseudocounts = initial probabilities
5. **Validation**: 5-fold CV × 10 repeats → 97% accuracy on top 26% of events

### 2.7 What They Actually Achieved

| Metric | Manual Analysis | HMM Method |
|--------|-----------------|------------|
| Error rate | ~10% | 2-3% |
| Throughput | Hours | 9× faster than real-time |
| Bias potential | Unconscious selection | Automated, reproducible |

**Key insight**: Explicit modeling of enzyme/segmenter artifacts beat implicit handling. The modular design made this tractable.

---

## 3. Resources for Epigenetic Marker Detection

### 3.1 Important Clarification: What Nanopore Can Detect

**Nanopore directly detects modifications to nucleic acids**, not proteins:

| Detectable | NOT Directly Detectable |
|------------|------------------------|
| DNA: 5mC, 5hmC, 5fC, 5caC, 6mA, 4mC | Histone modifications (H3K9ac, H3K27me3, etc.) |
| RNA: m6A, m5C, Ψ (pseudouridine), inosine | Protein-DNA interactions |

**H3K9 acetylation** is a histone modification—nanopore sequences DNA/RNA, not chromatin. However, you CAN use nanopore to:
- Map DNA methylation patterns that correlate with chromatin state
- Detect R-loops, secondary structures
- Phase haplotypes to study allele-specific epigenetics

### 3.2 Established DNA Modifications

#### 5mC and 5hmC (Cytosine Methylation Pathway)

**Current Tools:**
- **Dorado** (ONT official): Built-in 5mC/5hmC calling with R10.4.1
- **Megalodon**: ONT's modification caller using neural networks + Remora models
- **Nanopolish**: HMM-based (Simpson et al. 2017)—closest to paper's methodology
- **DeepMod**: Bidirectional LSTM for 5mC/6mA detection
- **DeepSignal**: CNN-based modification detection
- **f5c**: GPU-accelerated Nanopolish reimplementation

**Key Papers:**
1. Simpson et al. (2017) "Detecting DNA cytosine methylation using nanopore sequencing" *Nature Methods* — foundational HMM approach
2. Rand et al. (2017) — signalAlign methodology
3. Nature Comms 2025: "Double and single stranded detection of 5mC and 5hmC with nanopore sequencing" — recent benchmarking with R10.4.1HD

**Signal Resources:**
- ONT official k-mer models: `https://github.com/nanoporetech/kmer_models`
- Nanopolish models: `https://github.com/jts/nanopolish/tree/master/etc/r9-models`

#### 6mA (N6-methyladenine)

Primarily studied in bacteria and some eukaryotes.

**Tools:**
- **DeepMod**: ~0.9 average precision on E. coli
- **mCaller**: Designed specifically for 6mA
- **Tombo**: ONT's statistical modification detection

### 3.3 RNA Modifications (Direct RNA Sequencing)

#### m6A Detection

**This is the most active area for applying HMM/ML methods.**

**Key Tools:**
1. **m6Anet** (Nature Methods 2022)
   - Multiple instance learning framework
   - Handles missing read-level labels
   - Generalizes across species without retraining
   - GitHub: `https://github.com/GoekeLab/m6anet`

2. **Xron** (Genome Research 2024)
   - Hybrid NHMM + neural network
   - Semi-supervised training on synthetic + IP data
   - End-to-end methylation-distinguishing basecaller
   - Uses constrained transition matrix (similar philosophy to paper!)

3. **DENA** (Genome Biology 2022)
   - Trained on in vivo Arabidopsis transcripts
   - Avoids artifacts from saturated synthetic data

4. **EpiNano** — early statistical approach
5. **Tombo** — comparative analysis method
6. **MINES** — m6A detection using electrical signatures

**Benchmark Resource:**
- **NanOlympicsMod**: Framework comparing 14 m6A detection tools
- Paper: Maestri et al. (Brief Bioinform 2024) "Benchmarking computational methods for m6A profiling"

#### Other RNA Modifications

| Modification | Detection Status | Key Paper/Tool |
|--------------|-----------------|----------------|
| m5C | Emerging | Prediction at single-molecule resolution (2024) |
| Pseudouridine (Ψ) | Research stage | Begik et al. Nature Biotech 2021 |
| Inosine | Research stage | Limited tools available |
| 2'-O-methylation | Challenging | Signal differences subtle |

### 3.4 Datasets for Your Project

#### Public Raw Signal Data

1. **NA12878 Human Reference**
   - 30× coverage, R9.4
   - Has bisulfite ground truth for 5mC
   - Used in Nanopolish, DeepSignal training

2. **HEK293T Cell Lines**
   - Multiple direct RNA-seq datasets
   - m6A ground truth from miCLIP/m6ACE-seq
   - Available via ENA/SRA

3. **Yeast Datasets (ime4Δ knockouts)**
   - Wild-type vs m6A-depleted
   - Clean control for method development

4. **Mouse Cerebellum** (2025)
   - 5mC + 5hmC mapped at single-base resolution
   - R10.4.1HD chemistry
   - Published with orthogonal validation

5. **Arabidopsis (DENA paper)**
   - Direct RNA-seq with m6A ground truth
   - mtb and fip37-4 mutants

### 3.5 Applying the Paper's Methodology

#### Promising Targets

**High feasibility (clear signal differences, existing ground truth):**
1. **5mC vs 5hmC vs C** — exactly what the paper did, replicate with modern chemistry
2. **m6A in RNA** — well-established signal differences, multiple validation datasets
3. **6mA in bacterial DNA** — strong signal, good ground truth available

**Medium feasibility:**
1. **5fC and 5caC** — demonstrated distinguishable by Akeson group (2014)
2. **4mC** — less studied but detectable
3. **Multiple modifications simultaneously** — challenge is combinatorial explosion

**Research frontier:**
1. **RNA pseudouridine** — subtle signal differences
2. **Context-dependent modification calling** — using flanking sequence to improve calls

#### Implementation Strategy

1. **Start with the paper's code**
   - Repository: `https://github.com/UCSCNanopore/Data/tree/master/Automation`
   - Familiarize with YAHMM framework

2. **Adapt for modern chemistry**
   - R9.4.1 vs R10.4.1 have different k-mer lengths (6-mer vs 9-mer)
   - Different motor proteins (helicase vs Φ29 DNAP)
   - May need to adjust artifact modules

3. **Build modification-specific modules**
   ```python
   # Conceptual: Module factory for modified bases
   def create_modification_fork(base, modifications, kmer_models):
       """
       Creates a fork structure for classifying modification state
       
       Args:
           base: 'C' or 'A' (the base being modified)
           modifications: ['5mC', '5hmC'] or ['m6A']
           kmer_models: dict of {mod: {kmer: (mean, std)}}
       """
       # Build modules for each modification state
       # Connect via port structure
       # Return HMM subgraph
   ```

4. **Validation approach**
   - Use knockout/depleted samples as negative controls
   - Compare to orthogonal methods (bisulfite, miCLIP)
   - Calculate agreement with existing callers

### 3.6 Key Recent Papers to Study

| Paper | Year | Relevance |
|-------|------|-----------|
| Simpson et al. "Detecting DNA cytosine methylation" | 2017 | Foundational HMM approach |
| Xron (Teng et al.) | 2024 | Modern HMM-NN hybrid for m6A |
| m6Anet (Hendra et al.) | 2022 | MIL framework, good baseline |
| ReQuant | 2025 | Rule-based k-mer imputation for limited data |
| Uncalled4 | 2025 | Fast signal alignment with mod detection |
| Poregen | 2025 | De novo k-mer model generation |
| "Effective training with limited data" | 2024 | Hybrid HMM-DNN for sparse training |

### 3.7 Tool Ecosystem Summary

```
Raw Signal Processing:
├── Dorado (ONT) — modern basecalling + mods
├── Bonito (ONT) — research basecaller
└── Guppy (ONT, deprecated) — legacy but well-documented

Signal Alignment:
├── Nanopolish — HMM-based, closest to paper
├── f5c — GPU Nanopolish
├── Tombo — statistical modification detection
└── Uncalled4 — fast DTW alignment

Modification Detection:
├── DNA
│   ├── Megalodon + Remora (5mC, 5hmC, 6mA)
│   ├── DeepMod (5mC, 6mA)
│   ├── DeepSignal (5mC, 6mA)
│   └── modkit (ONT official)
└── RNA
    ├── m6Anet (m6A)
    ├── Xron (m6A)
    ├── DENA (m6A)
    ├── EpiNano (m6A)
    └── ELIGOS (multiple mods)

Benchmarking:
├── NanOlympicsMod (m6A tools)
└── METEORE (DNA methylation consensus)
```

---

## 4. Practical Next Steps

### Recommended Reading Order

1. **Simpson et al. 2017** — understand the original HMM formulation for nanopore
2. **The Schreiber & Karplus paper** (your PDF) — modular architecture + classification
3. **Xron paper** — modern hybrid approach
4. **m6Anet paper** — if targeting RNA modifications

### Initial Experiments

1. **Reproduce paper results**: Download their data, run their scripts, verify 2-3% error
2. **Port to modern data**: Adapt modules for R10.4.1 9-mer model
3. **Pick a target modification**: m6A in RNA or 5hmC in DNA recommended
4. **Build minimal fork**: Single modification vs unmodified (binary classification)
5. **Scale up**: Multi-modification forks, whole-genome calling

### Technical Considerations

- **Training data scarcity**: Main challenge for new modifications
- **k-mer coverage**: Need examples of modification in diverse sequence contexts
- **Ground truth**: Orthogonal validation essential (bisulfite, IP-based methods)
- **Chemistry changes**: R9 → R10 changed k-mer length and signal characteristics

---

## References

### Foundational
- Schreiber & Karplus (2015) "Analysis of nanopore data using hidden Markov models" *Bioinformatics*
- Simpson et al. (2017) "Detecting DNA cytosine methylation using nanopore sequencing" *Nature Methods*

### Modern Methods
- Hendra et al. (2022) "m6Anet" *Nature Methods*
- Teng et al. (2024) "Xron: Detecting m6A using semi-supervised learning" *Genome Research*
- Bersani et al. (2025) "Double and single stranded detection of 5mC and 5hmC" *Nat Comms*

### Tools & Resources
- Nanopolish: https://github.com/jts/nanopolish
- m6Anet: https://github.com/GoekeLab/m6anet
- ONT k-mer models: https://github.com/nanoporetech/kmer_models
- Paper code: https://github.com/UCSCNanopore/Data/tree/master/Automation
