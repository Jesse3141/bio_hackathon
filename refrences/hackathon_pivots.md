# 🧬 Quick Pivot Projects: 2-4 Hour Hackathon Ideas

*Based on your repo's existing data, models, and code*

---

## Overview

Your repo is **exceptionally well-positioned** for quick pivots. You have:
- ✅ Ground truth datasets (rare in epigenetics!)
- ✅ Working legacy HMM (1070 states, 97% accuracy)
- ✅ Pre-built pomegranate HMM framework (920 lines, ready to go)
- ✅ 262K k-mer reference table
- ✅ Modern POD5 data with known modification sites

Here are **12 creative pivot projects** organized by theme, each achievable in 2-4 hours.

---

## 🎯 Category 1: "Dissecting the Signal" (Pure Data Analysis)

### 1.1 The 5-mer Fingerprint Gallery
**Time**: 2-3 hours | **Deliverable**: Visual poster + notebook

**The Hook**: "Not all k-mers are created equal—some are methylation crystal balls, others are noise."

**What You Do**:
1. Load `9mer_levels_v1.txt` (262K k-mers)
2. For each cytosine-containing k-mer, compute the theoretical "methylation shift" (expected ΔpA between C and 5mC contexts)
3. Visualize: Which 5-mer contexts have the LARGEST vs. SMALLEST separability?
4. Create a "periodic table" heatmap of k-mer discriminability

**Why It's Cool**: This is essentially **feature engineering** for methylation detection. Show which sequence contexts are "easy wins" vs. "hard problems."

**Code Sketch**:
```python
import pandas as pd
import seaborn as sns

# Load k-mer table
kmers = pd.read_csv('9mer_levels_v1.txt', sep='\t', comment='#')

# Find all k-mers with C in position 5 (center)
c_kmers = kmers[kmers['kmer'].str[4] == 'C']

# Compute hypothetical 5mC shift (literature: ~0.8 pA average)
# Plot histogram of level_stdv to show which contexts are noisier
```

---

### 1.2 The "Confidence Autopsy"
**Time**: 2-3 hours | **Deliverable**: Notebook with actionable insights

**The Hook**: "Why do 74% of reads fail the confidence filter? Let's find out."

**What You Do**:
1. Load legacy JSON data + Test Set.csv
2. For every event that FAILED the filter score > 0.1 threshold:
   - What was its predicted class?
   - What were its signal statistics (mean, std, duration)?
   - Was the ground truth label "hard" (hmC vs. mC) or "easy" (C vs. modified)?
3. Build a decision tree to predict filter failure
4. Visualize: Scatter plot of "passed" vs. "failed" events in signal-space

**Why It's Cool**: Directly addresses the 97%→92% accuracy gap. Could reveal systematic issues (e.g., "all short events fail").

---

### 1.3 Duration Dynamics: The Temporal Signature of Methylation
**Time**: 2-3 hours | **Deliverable**: Publication-quality figure

**The Hook**: "Methylated cytosines don't just change current—they change *dwell time*."

**What You Do**:
1. Parse `UCSCN/Data/*.json` to extract segment durations
2. Group by ground truth label (C, mC, hmC)
3. Plot duration distributions for each modification type
4. Statistical test: Do methylated bases have different dwell times?

**Why It's Cool**: Most HMM work focuses on current levels. Duration is often ignored but could be a **hidden feature** for classification.

**Biological Angle**: Methylation affects polymerase kinetics—this could be a second modality for detection.

---

## 🔬 Category 2: "HMM Surgery" (Model Analysis)

### 2.1 State Autopsy: Where Does the HMM Spend Its Time?
**Time**: 2-3 hours | **Deliverable**: Sankey diagram or state heatmap

**The Hook**: "A 1070-state HMM is a black box. Let's open it."

**What You Do**:
1. Load trained YAHMM model (`trained_hmm.txt`)
2. Run Viterbi decoding on test set
3. Count: How often does each state get visited?
4. Visualize: State visitation frequency as a bar chart or heatmap
5. Identify: Which artifact states (overseg, underseg, backslip, flicker, blip) are actually used?

**Why It's Cool**: If 90% of paths never touch certain artifact states, those states are candidates for **pruning**. This directly informs your pomegranate simplification.

**Advanced**: Make a Sankey diagram showing flow through the "circuit board."

---

### 2.2 The Fork Posterior Landscape
**Time**: 2-3 hours | **Deliverable**: 3D surface plot or interactive widget

**The Hook**: "At every cytosine position, the HMM 'votes' between C/mC/hmC. Let's visualize the vote."

**What You Do**:
1. Run forward-backward on test reads
2. Extract posterior probabilities at each fork position
3. Create a 3D landscape: X=read index, Y=fork position, Z=posterior(mC)
4. Color-code by ground truth label

**Why It's Cool**: Shows **where the model is confident** vs. uncertain. Patterns emerge—maybe position 3 is always easy but position 7 is hard.

---

### 2.3 Transition Matrix Archaeology
**Time**: 2 hours | **Deliverable**: Annotated heatmap

**The Hook**: "The HMM's learned transitions encode biology. What did it learn?"

**What You Do**:
1. Extract transition matrix from trained model
2. Compare to untrained (initial) parameters
3. Visualize: Which transitions changed most during Baum-Welch?
4. Interpret: Does high self-loop on M states indicate oversegmentation is common?

**Why It's Cool**: This is **model interpretability** for HMMs. You can make claims like "The model learned that backslips occur 3x more often than we assumed."

---

## 🧮 Category 3: "Algorithmic Explorations" (DP & Computation)

### 3.1 The Viterbi vs. Posterior Showdown
**Time**: 2-3 hours | **Deliverable**: Comparison plot + accuracy table

**The Hook**: "Maximum likelihood path vs. summed probabilities—which wins for methylation?"

**What You Do**:
1. Implement both classification strategies:
   - **Viterbi**: argmax over paths (what the paper does)
   - **Posterior**: sum forward-backward probabilities at forks
2. Run both on test set
3. Compare: When do they disagree? Which is more accurate?

**Why It's Cool**: This is a fundamental question in HMM inference. Your ground truth data makes it answerable!

**Hypothesis**: Posterior decoding might be better for low-confidence reads where multiple paths are plausible.

---

### 3.2 Banded DP: How Much Speedup for Free?
**Time**: 3-4 hours | **Deliverable**: Speed vs. accuracy curve

**The Hook**: "The HMM alignment doesn't need to consider impossible state jumps. How much can we prune?"

**What You Do**:
1. Implement banded forward algorithm (only compute cells within bandwidth of diagonal)
2. Vary bandwidth: 5, 10, 20, 50, unlimited
3. Measure: Speedup factor vs. accuracy loss
4. Find the "sweet spot"

**Why It's Cool**: This is practical—banded DP is how production aligners work. You'd be showing the accuracy/speed tradeoff for methylation detection.

---

### 3.3 Minimum Edit Distance for Current Signals
**Time**: 2-3 hours | **Deliverable**: Novel alignment metric + notebook

**The Hook**: "DNA alignment has edit distance. What's the equivalent for ionic current?"

**What You Do**:
1. Define a "current edit distance": 
   - Match: |observed - expected| < threshold
   - Insert: extra observed segment (oversegmentation)
   - Delete: missing segment (undersegmentation)
   - Substitute: wrong k-mer expectation
2. Implement with standard DP
3. Compare to HMM log-likelihood as a read quality metric

**Why It's Cool**: This creates a **simpler baseline** that doesn't require HMM machinery. Could be a fast filter for quality control.

---

## 🎨 Category 4: "Visualization & Communication"

### 4.1 The Methylation Movie
**Time**: 3-4 hours | **Deliverable**: Animated GIF/video

**The Hook**: "Watch the HMM decode a read in real-time."

**What You Do**:
1. Pick a single read
2. Run forward algorithm step-by-step
3. For each observation, plot:
   - Top: Current signal (time series)
   - Bottom: State probability distribution (bar chart)
4. Animate frames to show probability mass "flowing" through states

**Why It's Cool**: This is **incredible for presentations**. Nobody does this, and it demystifies how HMMs actually work.

---

### 4.2 The Chemistry Time Machine
**Time**: 2-3 hours | **Deliverable**: Side-by-side comparison figure

**The Hook**: "2013 nanopores vs. 2024 nanopores—how far have we come?"

**What You Do**:
1. Load legacy R7.3 JSON data (UCSCN/)
2. Load modern R10.4.1 POD5 data (nanopore_ref_data/)
3. Plot side-by-side:
   - Signal quality (noise distributions)
   - Segment durations
   - Current level ranges
4. Annotate: What changed and why it matters

**Why It's Cool**: Nobody has this comparison because nobody has both datasets! You do. This is novel.

---

### 4.3 Interactive K-mer Explorer (Streamlit)
**Time**: 3-4 hours | **Deliverable**: Deployed web app

**The Hook**: "Type any DNA sequence, see its expected nanopore signal."

**What You Do**:
1. Streamlit app with text input
2. Parse sequence into k-mers
3. Look up expected current levels from `9mer_levels_v1.txt`
4. Plot: Expected signal trace for the sequence
5. Bonus: Add methylation toggle to show how 5mC shifts the signal

**Why It's Cool**: Interactive demos are presentation gold. Also useful as a lab tool.

---

## 🧬 Category 5: "Biological Questions"

### 5.1 CpG vs. Non-CpG Context Discrimination
**Time**: 2-3 hours | **Deliverable**: Statistical analysis + figure

**The Hook**: "CpG methylation is biologically relevant. Is it also easier to detect?"

**What You Do**:
1. Load BED files with CpG context annotations (score column: 0, 1, 2)
2. Stratify test set accuracy by context
3. Hypothesis test: Is accuracy higher for CpG sites?
4. Biological interpretation: Why might CpG contexts be different?

**Why It's Cool**: Connects computational performance to **biological relevance**. CpG islands are key for gene regulation.

---

### 5.2 The 5hmC vs. 5mC Confusion Matrix Deep Dive
**Time**: 2 hours | **Deliverable**: Detailed error analysis

**The Hook**: "When the HMM is wrong, is it systematically wrong?"

**What You Do**:
1. Build 3×3 confusion matrix (C/mC/hmC × predicted/actual)
2. Focus on off-diagonal: Which modifications get confused?
3. For confused reads, extract signal features
4. Hypothesis: Is hmC→mC confusion driven by similar current levels?

**Why It's Cool**: This error analysis could inform **targeted model improvements**. Maybe you need more training data for specific cases.

---

### 5.3 Position Effects Along the Template
**Time**: 2-3 hours | **Deliverable**: Per-position accuracy plot

**The Hook**: "Are some positions along the 54-mer harder to classify?"

**What You Do**:
1. The template has 8 cytosine positions at fixed locations
2. Extract per-position accuracy from test set
3. Plot: Accuracy vs. position along template
4. Correlate with: Distance from adapter, local sequence context, signal quality

**Why It's Cool**: Position effects could reveal **systematic biases** in library prep or sequencing that the model can't compensate for.

---

## 🏆 Recommended "Best Bets" for a 3-Hour Session

| Project | Impressiveness | Novelty | Difficulty | Best For |
|---------|---------------|---------|------------|----------|
| **5-mer Fingerprint Gallery** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | Visual impact |
| **State Autopsy** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Model insight |
| **Chemistry Time Machine** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Storytelling |
| **Methylation Movie** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | "Wow" factor |
| **Duration Dynamics** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Biological novelty |
| **CpG Context Analysis** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | Biological relevance |

---

## Suggested 3-Hour Sprint Combos

### Combo A: "The Visual Story" (presentation-focused)
1. **Hour 1**: Chemistry Time Machine (R7.3 vs R10.4.1 comparison)
2. **Hour 2**: 5-mer Fingerprint Gallery (k-mer discriminability heatmap)
3. **Hour 3**: Assemble slides with narrative arc

### Combo B: "The Model Deep Dive" (technical audience)
1. **Hour 1**: State Autopsy (where does the HMM spend time?)
2. **Hour 2**: Transition Matrix Archaeology (what did Baum-Welch learn?)
3. **Hour 3**: Fork Posterior Landscape (visualize classification confidence)

### Combo C: "The Biological Angle" (bio audience)
1. **Hour 1**: Duration Dynamics (dwell time as hidden feature)
2. **Hour 2**: CpG Context Discrimination (biology meets performance)
3. **Hour 3**: 5hmC vs 5mC Confusion Analysis (error characterization)

### Combo D: "The Algorithmic Exploration" (CS audience)
1. **Hour 1**: Viterbi vs. Posterior Showdown
2. **Hour 2**: Banded DP speedup analysis
3. **Hour 3**: Current Edit Distance as a baseline metric

---

## Quick-Start Code Snippets

### Loading Legacy Data
```python
import json
import pandas as pd

# Load event data
with open('UCSCN/Data/14418004-s04.json', 'r') as f:
    data = json.load(f)

# Extract segment means
events = []
for event in data['events']:
    for seg in event['segments']:
        events.append({
            'mean': seg['mean'],
            'std': seg['std'],
            'duration': seg['duration']
        })
df = pd.DataFrame(events)
```

### Loading K-mer Table
```python
import pandas as pd

kmers = pd.read_csv(
    'nanopore_ref_data/kmer_models/9mer_levels_v1.txt',
    sep='\t',
    comment='#',
    names=['kmer', 'level_mean', 'level_stdv', 'sd_mean', 'sd_stdv']
)
```

### Loading Test Labels
```python
labels = pd.read_csv('UCSCN/Test Set.csv')
# Columns: event_id, ground_truth, predicted, filter_score, ...
```

---

## Final Thought

Your repo is a goldmine for quick pivots because you have **ground truth data**—the rarest commodity in bioinformatics. Every project above leverages this to answer questions that other researchers *can't* answer. Pick the one that tells the best story for your audience!
