# Nanopore + HMM Hackathon: 4-Track Plan

**Duration**: ~8 hours  
**Team size**: 4-6 people (1-2 per track)  
**Goal**: Develop intuition for HMM-based nanopore modification detection

---

## Sync Points

| Time | Activity |
|------|----------|
| 0:00 | Kickoff: 15 min overview, assign tracks |
| 2:00 | Check-in #1: Share blockers, early findings |
| 4:00 | Lunch + mid-day sync: What's working? Pivot if needed |
| 6:00 | Check-in #2: Preliminary results |
| 7:30 | Final presentations: 10 min per track |

---

## Track 1: Original Paper Deep Dive

**Goal**: Get the Schreiber & Karplus code running, understand it deeply, then break it systematically.

**Team size**: 1-2 people  
**Skills needed**: Python, basic HMM familiarity

### Phase 1A: Reproduce (Hours 0-3)

**Setup (~30 min)**
```bash
# Clone the repository
git clone https://github.com/jmschreiber/yahmm  # or use pomegranate
git clone https://github.com/UCSCNanopore/Data

# Check Python dependencies
# Paper used Python 2.7 - may need compatibility fixes
# YAHMM is deprecated; pomegranate is the successor
pip install pomegranate numpy scipy matplotlib
```

**Understand the data structure (~30 min)**
- Navigate to `Data/tree/master/Automation`
- Examine the input data format:
  - What does a "segment" look like? (mean, std, duration)
  - How are events organized?
  - Where is the ground truth (C/mC/hmC labels)?

**Key questions to answer:**
1. How many events total? How many per class?
2. What's the distribution of segment counts per event?
3. What's the range of current values?

**Run the pipeline (~1.5 hrs)**
1. Execute the segmentation (or use pre-segmented data)
2. Build the HMM from the paper's specification
3. Run training (Baum-Welch)
4. Evaluate on test set

**Document as you go:**
- What errors did you hit?
- What library translations were needed (YAHMM → pomegranate)?
- Can you reproduce the ~97% accuracy on top 26% of events?

### Phase 1B: Ablation Study (Hours 3-6)

**Goal**: Understand which model components matter by removing them.

**Ablation experiments:**

| Experiment | What to remove | Hypothesis |
|------------|---------------|------------|
| A1 | Backslip states (B) | Accuracy drops on events with enzyme slippage |
| A2 | Oversegmentation handling (M1/M2 → single M) | Accuracy drops on oversegmented events |
| A3 | Undersegmentation states (U) | Accuracy drops, especially after preprocessing removal |
| A4 | Flicker states (M3/M4) | Modest accuracy drop |
| A5 | Blip handling (I→S loop) | Modest accuracy drop |
| A6 | All artifacts (standard profile HMM) | Significant accuracy drop |

**For each ablation:**
1. Modify the HMM construction code
2. Retrain on same training set
3. Evaluate on same test set
4. Record: accuracy, filter score distribution, failure modes

**Analysis questions:**
- Which ablation hurts most?
- Do certain ablations affect specific event types?
- Can you identify events that fail *only* when a specific state is removed?

### Phase 1C: Extensions (Hours 6-7.5, if time)

**Option 1: Improve the model**
- Add duration modeling (paper mentions they only used mean, not std or duration)
- Tie emission distributions across M1/M2/M3/M4 during training
- Experiment with different pseudocount strategies

**Option 2: Visualization**
- Plot Viterbi paths through the HMM for example events
- Visualize which states are used most frequently
- Create a "confusion matrix" of path choices at forks

### Deliverables
- [ ] Working reproduction of paper results
- [ ] Table of ablation results
- [ ] 2-3 key insights about which components matter
- [ ] List of code/compatibility issues encountered

### Resources
- Paper: Schreiber & Karplus 2015 (provided PDF)
- Code: https://github.com/UCSCNanopore/Data/tree/master/Automation
- pomegranate docs: https://pomegranate.readthedocs.io/

---

## Track 2: Nanopolish Comparison

**Goal**: Understand how the paper's approach relates to Nanopolish, the widely-used HMM tool for nanopore modification calling.

**Team size**: 1-2 people  
**Skills needed**: Python, ability to read C++ (Nanopolish core), HMM concepts

### Phase 2A: Nanopolish Architecture (Hours 0-3)

**Setup (~30 min)**
```bash
git clone https://github.com/jts/nanopolish
# Don't need to build unless you want to run it
# Focus on understanding the code

# Key files to examine:
# src/nanopolish_methyltrain.cpp - training
# src/nanopolish_call_methylation.cpp - inference
# src/hmm/ - HMM implementation
```

**Document the Nanopolish HMM structure (~2 hrs)**

Answer these questions by reading the code:

1. **State structure**
   - What states does Nanopolish use?
   - How does it handle k-mers? (one state per k-mer? grouped?)
   - Does it have artifact states like the paper?

2. **Emission distributions**
   - Gaussian? What parameters?
   - How are k-mer models loaded?
   - Are modified k-mers separate states or same states with different emissions?

3. **Transition structure**
   - How are transitions between k-mers handled?
   - Any special handling for stays (same k-mer repeated)?
   - Any backslip equivalent?

4. **Training approach**
   - Supervised? Semi-supervised?
   - What's the ground truth source?
   - Baum-Welch or something else?

**Create a comparison table:**

| Aspect | Schreiber & Karplus | Nanopolish |
|--------|---------------------|------------|
| State per position | 4 match + artifacts | ? |
| Emission model | Gaussian | ? |
| Artifact handling | Explicit states | ? |
| Classification | Fork structure | ? |
| Training | Baum-Welch | ? |

### Phase 2B: Conceptual Comparison (Hours 3-5)

**Key architectural differences to explore:**

1. **Known sequence vs unknown sequence**
   - Paper: Aligns to a *known* reference sequence
   - Nanopolish: Also aligns to reference, but for whole-genome scale
   - How does this change the HMM design?

2. **Classification approach**
   - Paper: Forks in HMM, path determines class
   - Nanopolish: Log-likelihood ratio between methylated/unmethylated models
   - Which is more flexible? More efficient?

3. **Scalability**
   - Paper: ~54 segments, 3 forks
   - Nanopolish: Whole genome, every CpG
   - What compromises does Nanopolish make for scale?

**Try to answer:**
- Could the paper's artifact states improve Nanopolish?
- Could Nanopolish's likelihood-ratio approach work for the paper's task?
- What would a hybrid look like?

### Phase 2C: Practical Transfer (Hours 5-7.5)

**Option A: Port artifact states to Nanopolish-style model**
- Sketch how you'd add backslip handling to Nanopolish's HMM
- What would break? What would improve?

**Option B: Port Nanopolish's approach to paper's task**
- Instead of forks, compute P(data | C) vs P(data | mC) vs P(data | hmC)
- Implement log-likelihood ratio classification
- Compare to paper's fork-based classification

**Option C: Survey other tools**
- Briefly examine: signalAlign, Tombo, f5c
- How do their HMMs compare?
- Create a taxonomy of approaches

### Deliverables
- [ ] Detailed comparison table (Schreiber & Karplus vs Nanopolish)
- [ ] Diagram of Nanopolish HMM structure
- [ ] 2-3 concrete ideas for cross-pollination
- [ ] Assessment: which approach is better for what task?

### Resources
- Nanopolish: https://github.com/jts/nanopolish
- Simpson et al. 2017 paper: "Detecting DNA cytosine methylation using nanopore sequencing"
- Nanopolish methylation docs: https://nanopolish.readthedocs.io/en/latest/quickstart_call_methylation.html

---

## Track 3: Modern Nanopore Adaptation

**Goal**: Understand what would need to change to run the paper's methodology on modern ONT data (R9.4.1 or R10.4.1).

**Team size**: 1-2 people  
**Skills needed**: Python, data exploration, some bioinformatics

### Phase 3A: Understand the Differences (Hours 0-2)

**Create a detailed comparison:**

| Parameter | Paper (M2MspA + Φ29) | ONT R9.4.1 | ONT R10.4.1 |
|-----------|---------------------|------------|-------------|
| Pore protein | M2MspA | CsgG variant | CsgG dual-reader |
| Motor | Φ29 DNA polymerase | Helicase | Helicase |
| k-mer length | ~4 nt | 5-6 nt | 9 nt |
| Sampling rate | ? | 4 kHz | 4 kHz |
| Typical dwell time | ? | ~2 ms/base | ~2.5 ms/base |
| Movement | Synthesis (ratchet) | Unwinding | Unwinding |
| Backslip frequency | Common | Rare (different mode) | Rare |
| Current range | 15-70 pA | 50-120 pA | 50-120 pA |

**Research questions:**
1. What does helicase stalling look like vs Φ29 backslip?
2. How does the dual-reader in R10 affect signal structure?
3. Are oversegmentation/undersegmentation rates similar?

**Resources to consult:**
- ONT technical documentation
- Papers comparing R9 vs R10
- Pore model files from Nanopolish/Dorado

### Phase 3B: Identify Suitable Data (Hours 2-4)

**Requirements for equivalent experiment:**
1. Raw signal data (FAST5/POD5 files)
2. Known sequence with controlled modification
3. Ground truth labels
4. Ideally: multiple modification states (not just binary)

**Candidate datasets:**

| Dataset | Description | Pros | Cons |
|---------|-------------|------|------|
| Lambda phage (methylated vs unmethylated) | Enzymatically methylated bacterial DNA | Clean control, available | Binary only (mC vs C), no hmC |
| NA12878 | Human reference genome | Bisulfite ground truth | No controlled positions |
| Synthetic oligos | Custom sequences with modifications | Perfect control | Need to find/generate |
| ONT modification standards | ONT's internal controls | Designed for this | Availability varies |

**Search for data:**
```bash
# ENA/SRA searches
# Look for: "nanopore" + "methylation" + "control" + "R9" or "R10"
# Check: https://github.com/nanopore-wgs-consortium/NA12878
```

**Document what you find:**
- Dataset name and accession
- Chemistry version
- What modifications are labeled
- Ground truth source
- File sizes / feasibility for hackathon

### Phase 3C: Outline Adaptation Steps (Hours 4-6)

**For each difference, specify what changes:**

**1. k-mer model**
- Paper: ~4-mer emission distributions
- R9: Need 6-mer pore model
- R10: Need 9-mer pore model
- Where to get these? ONT provides, also Nanopolish repo

**2. Artifact states redesign**

| Paper state | Helicase equivalent | Implementation notes |
|-------------|--------------------|--------------------|
| Backslip (B) | Stall/skip? | Research helicase failure modes |
| Flicker (M3/M4) | May not exist | Possibly remove |
| Overseg (M1/M2) | Similar | Keep, tune probabilities |
| Underseg (U) | More important with 9-mer | Keep, maybe expand |

**3. Segmentation**
- Paper used custom segmenter
- Options for ONT: Tombo resquiggle, Nanopolish eventalign
- Or implement paper's segmenter on ONT data

**4. Signal normalization**
- Different current ranges
- Need to normalize or adjust emission distributions

### Phase 3D: Prototype (Hours 6-7.5, if time)

**Minimum viable test:**
1. Take ONE ONT read with known methylation status
2. Manually segment it (or use existing tool)
3. Build minimal HMM (no artifact states, just match/insert/delete)
4. See if Viterbi alignment makes sense

**Questions to answer:**
- Does the basic alignment work?
- Where does it fail?
- What's missing?

### Deliverables
- [ ] Detailed comparison table (paper setup vs ONT)
- [ ] Catalog of available datasets with suitability assessment
- [ ] Step-by-step adaptation roadmap
- [ ] List of open questions / research needed
- [ ] (Stretch) Minimal prototype results

### Resources
- ONT pore models: https://github.com/nanoporetech/kmer_models
- Nanopolish models: https://github.com/jts/nanopolish/tree/master/etc/r9-models
- R9 vs R10 comparison papers

---

## Track 4: m6A RNA Modification

**Goal**: Apply HMM methodology to a different modification (m6A in RNA) using fully labeled synthetic data.

**Team size**: 1-2 people  
**Skills needed**: Python, HMM concepts, willing to work with RNA-seq data

### Phase 4A: Understand m6A Detection (Hours 0-1.5)

**Background reading (skim, don't deep-dive):**
- m6A is most abundant internal mRNA modification
- Occurs at DRACH motifs (D=A/G/U, R=A/G, H=A/C/U)
- Affects mRNA stability, splicing, translation

**Key signal question:**
- How different is A vs m6A in nanopore current?
- Less dramatic than C/mC/hmC differences
- Context-dependent (5-mer matters)

**Understand the data source:**

**IVT (In Vitro Transcribed) RNA:**
- Synthesize RNA with either 100% ATP or 100% m6ATP
- Every adenosine is either methylated or not
- Clean ground truth, but artificial

**Dataset**: Liu et al. 2019 IVT dataset
- Used to benchmark m6Anet, Xron, etc.
- Contains matched methylated/unmethylated reads

### Phase 4B: Get the Data (Hours 1.5-2.5)

**Download and explore:**
```bash
# The Liu et al. 2019 IVT data
# Check ENA/SRA for accession numbers
# Look for: PRJNA548268 or similar

# You need:
# 1. FAST5 files (raw signal)
# 2. Or pre-processed eventalign output
```

**If raw FAST5 is too large/slow:**
- Look for pre-computed features (mean current per 5-mer)
- m6Anet provides processed features for benchmarking
- Can work with summary statistics instead of raw signal

**Explore the data:**
1. How many reads per condition?
2. What's the 5-mer distribution at DRACH sites?
3. What's the current difference between A and m6A?

### Phase 4C: Build Minimal HMM (Hours 2.5-5)

**Start simple:**

**Model 1: Single-site classifier**
- Don't model whole transcript
- Focus on one DRACH 5-mer at a time
- Two-path fork: A vs m6A

```
        ┌─ A path ──┐
Input ──┤           ├── Output
        └─ m6A path ┘
```

**Each path needs:**
- 5 match states (one per position in 5-mer)
- Emission: Gaussian with parameters from pore model
- Transitions: Simple left-to-right, maybe with self-loops

**Implementation steps:**
1. Get 5-mer pore model for RNA (R9.4.1 RNA or RNA004)
2. Get modified 5-mer parameters (harder - may need to estimate from data)
3. Build HMM structure in pomegranate
4. Extract training data: segments at known DRACH sites
5. Train with Baum-Welch (or use fixed parameters from pore model)
6. Classify test sites

### Phase 4D: Evaluate and Compare (Hours 5-7)

**Evaluation:**
1. Accuracy on IVT data (should be high - it's easy)
2. ROC/PR curves
3. Per-5-mer accuracy (some contexts easier than others?)

**Compare to existing tools:**

| Tool | Approach | Expected AUC (IVT) | What to compare |
|------|----------|-------------------|-----------------|
| Your HMM | Profile HMM with Gaussian emissions | ? | Baseline |
| m6Anet | Neural network + MIL | ~0.90 read-level | Does NN help? |
| Xron | NHMM + neural basecaller | ~0.93 read-level | HMM hybrid advantage? |
| EpiNano | SVM on basecalling errors | Lower | Classic ML baseline |
| Tombo | Statistical test | Lower | Simplest approach |

**Key comparison questions:**
1. Your HMM vs EpiNano: Both "classical" ML - which is simpler/better?
2. Your HMM vs m6Anet: How much does the neural network add?
3. Your HMM vs Xron: Xron uses NHMM (constrained transitions) - does that help?

**Practical comparison steps:**
```bash
# If time permits, run m6Anet on same test data
pip install m6anet
m6anet inference --input_dir /path/to/eventalign --out_dir /path/to/output

# Compare predictions at same DRACH sites
```

**Questions to investigate:**
1. Does adding artifact states help? (probably less than for DNA)
2. Which 5-mers are hardest to classify?
3. What's the minimum data needed for reasonable accuracy?

### Phase 4E: Extensions (Hours 7-7.5, if time)

**Option 1: Add complexity**
- Add insert/delete states
- Add context beyond the 5-mer
- Model transition probabilities from data

**Option 2: Try biological data**
- Apply model trained on IVT to real cellular RNA
- Compare to m6Anet predictions
- Where does the simple model break down?

**Option 3: Multiple modifications**
- Can you distinguish A vs m6A vs m1A?
- Would need different training data

### Deliverables
- [ ] Working m6A classifier HMM
- [ ] Accuracy metrics on IVT benchmark
- [ ] Comparison to m6Anet (or other tool)
- [ ] Analysis of what works/doesn't work
- [ ] Ideas for improvement

### Resources
- m6Anet paper and code: https://github.com/GoekeLab/m6anet
- Xron paper (has good HMM discussion)
- Liu et al. 2019 IVT data
- ONT RNA pore models

---

## Cross-Track Integration

**Sync point discussions:**

**At 2:00 (Check-in #1):**
- Track 1: "We got X working / stuck on Y"
- Track 2: "Nanopolish does Z differently"
- Track 3: "Found these datasets, main challenge is..."
- Track 4: "Data is ready, starting HMM build"

**At 4:00 (Lunch sync):**
- Share: What's the most surprising thing you learned?
- Discuss: How do the tracks connect?
- Decide: Any pivots needed?

**At 6:00 (Check-in #2):**
- Track 1: Share ablation results → informs Track 3 (which states matter?)
- Track 2: Share Nanopolish findings → informs Track 4 (likelihood ratio approach?)
- Track 3: Share dataset findings → could Track 4 use same data?
- Track 4: Share early results → validates overall approach

**Final presentations should address:**
1. What did you set out to do?
2. What actually happened?
3. What did you learn about HMMs for nanopore?
4. What would you do next with more time?

---

## Appendix A: Data Resources

### Track 4: m6A IVT Data

**Primary dataset (Liu et al. 2019 - EpiNano paper):**
- **Accession**: PRJNA511582 (NCBI SRA)
- **Description**: Synthetic "curlcake" RNA sequences, IVT with either 100% ATP or 100% m6ATP
- **Replicates**: 2 biological replicates per condition
- **Chemistry**: R9.4.1 RNA
- **Paper**: Liu et al. "Accurate detection of m6A RNA modifications in native RNA sequences" Nature Comms 2019

**Download commands:**
```bash
# Install SRA toolkit if needed
# conda install -c bioconda sra-tools

# Download curlcake IVT data
prefetch PRJNA511582
fastq-dump --split-files SRR_ID  # Replace with specific run IDs

# Or for FAST5 files (needed for signal-level analysis):
# Check if FAST5 available on ENA, or contact authors
```

**Alternative: Pre-processed features**
- m6Anet provides extracted features for benchmarking
- GitHub: https://github.com/GoekeLab/m6anet
- Check their `data/` directory for processed datasets

### Track 3: Modern ONT Methylation Data

**Lambda phage controls:**
- Many labs have generated methylated/unmethylated lambda
- Check ONT's Community resources

**NA12878 Human Reference:**
- **Accession**: PRJEB23027 (Nanopore WGS Consortium)
- **Description**: Human genome, R9.4.1, with bisulfite validation
- **URL**: https://github.com/nanopore-wgs-consortium/NA12878
- **Caveat**: No controlled single positions, whole-genome scale

**HEK293T for m6A (if Track 3/4 want to share data):**
- **Accession**: PRJEB40872 (ENA)
- **Description**: WT and METTL3 KO cells
- **Chemistry**: RNA002 (R9.4.1 RNA)

### Track 2: Nanopolish Resources

**Nanopolish repository:**
```bash
git clone https://github.com/jts/nanopolish
# Key directories:
# src/hmm/ - HMM implementation
# etc/r9-models/ - R9.4.1 pore models
# etc/r10-models/ - R10 pore models (if available)
```

**Pore model files:**
- Located in `nanopolish/etc/r9-models/`
- Format: TSV with columns (kmer, level_mean, level_stdv, ...)
- Methylated models: separate files for CpG methylation

### Track 1: Original Paper Data

**Repository:**
```bash
git clone https://github.com/UCSCNanopore/Data
cd Data/tree/master/Automation
# Contains: scripts, example data, trained models
```

**YAHMM → pomegranate migration:**
- YAHMM deprecated, pomegranate is successor
- API similar but not identical
- Key differences:
  - `State` creation syntax
  - `model.bake()` required before use
  - Transition probability specification

---

## Appendix B: Quick Reference

### pomegranate HMM Example

```python
from pomegranate import *

# Define emission distributions
d1 = NormalDistribution(45.0, 3.0)  # Match state 1
d2 = NormalDistribution(52.0, 4.0)  # Match state 2
d3 = UniformDistribution(0, 90)     # Insert state

# Create states
s1 = State(d1, name="M1")
s2 = State(d2, name="M2")
s_ins = State(d3, name="Insert")

# Build model
model = HiddenMarkovModel()
model.add_states(s1, s2, s_ins)

# Add transitions
model.add_transition(model.start, s1, 1.0)
model.add_transition(s1, s2, 0.9)
model.add_transition(s1, s_ins, 0.1)
model.add_transition(s_ins, s2, 1.0)
model.add_transition(s2, model.end, 1.0)

model.bake()

# Use model
sequence = [44.2, 46.1, 51.8]  # Example current values
logp, path = model.viterbi(sequence)
```

### Key Formulas

**Viterbi (most likely path):**
```
V[t,j] = max_i { V[t-1,i] * a[i,j] } * b[j,x_t]
```

**Forward (total probability):**
```
α[t,j] = Σ_i { α[t-1,i] * a[i,j] } * b[j,x_t]
```

**Baum-Welch update (emission):**
```
μ_new = Σ_t γ[t,j] * x_t / Σ_t γ[t,j]
```

### File Formats

**FAST5**: HDF5 container with raw signal + metadata
**POD5**: Newer ONT format, more efficient
**Eventalign output**: TSV with columns like:
```
contig  position  reference_kmer  read_name  event_mean  event_stdv  event_length
```

---

## Emergency Pivots

**If Track 1 is completely stuck:**
- Skip to ablation with toy data
- Or join Track 2

**If Track 3 can't find suitable data:**
- Pivot to "design study" - outline what data would be needed
- Create synthetic test case

**If Track 4 data is too large:**
- Subsample aggressively
- Work with pre-computed features only
- Focus on single 5-mer context

**If everything is going smoothly:**
- Cross-pollinate: Track 1 person helps Track 4
- Attempt the "stretch" goals
- Start writing up findings early
