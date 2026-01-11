# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a bioinformatics hackathon project for HMM-based epigenetic modification detection from nanopore sequencing data, based on the Schreiber & Karplus methodology. The goal is to classify cytosine modifications (C, 5mC, 5hmC) using Hidden Markov Models that explicitly model nanopore/enzyme artifacts.

## Repository Structure

- `UCSCN/` - Main codebase from UCSCNanopore/Data repository
  - `epigenetics.py` - Core HMM model construction and training/testing functions
  - `PyPore Tutorial.ipynb` - Tutorial on using PyPore library
  - `Cytosine Classification.ipynb` - Replication of the paper's analysis
  - `Data/` - JSON files with pre-segmented nanopore events (13 files, 12.5 min each)
  - `untrained_hmm.txt` / `trained_hmm.txt` - Serialized HMM models
- `refrences/` - Background documentation
  - `nanopore_hmm_epigenetics_guide.md` - Comprehensive guide to the methodology
  - `hackathon_plans.md` - 4-track hackathon plan with detailed tasks

## Key Dependencies

```bash
pip install pomegranate numpy scipy matplotlib pandas seaborn
# Note: Original code uses YAHMM (deprecated), pomegranate is the successor
# API migration may be required for some functions
```

The code uses:
- **YAHMM/pomegranate** - HMM library (YAHMM deprecated, port to pomegranate needed)
- **PyPore** - Nanopore data handling (custom library from UCSC)

## Architecture: Circuit Board HMM Design

The HMM uses a modular "circuit board" architecture with 7-port modules:

```
┌─────────── Module for Position i ───────────┐
│   S1─┬─→ M1 ──┐                              │
│   S2─┼─→ M2 ──┼─→ SE ──┬─→ E1               │
│   S3─┤   ↑    │        ├─→ E2               │
│   ...│   D ───┤        │   ...              │
│   S7─┴─→ I ───┴────────┴─→ E7               │
└──────────────────────────────────────────────┘
```

**Artifact-handling states:**
- M1/M2 (Magenta): Oversegmentation - two match states with different self-loop probs
- U (Green): Undersegmentation - interpolated emission between adjacent positions
- B (Orange): Backslips - silent states for right-to-left transitions
- M3/M4 (Blue): Flicker - repeated incorporation attempts
- I→S loop (Teal): Blip handling - transient current spikes

**Fork structure for classification:**
```
         ┌─ C ──┐
Fork 1: ─┼─ mC ─┼─  (Cytosine variant)
         └─hmC──┘

         ┌─ X (abasic) ──┐
Fork 2: ─┼─ T ───────────┼─  (Label validation)
         └─ CAT ─────────┘
```

## Key Functions in epigenetics.py

- `EpigeneticsModel(distributions, name)` - Build complete HMM with artifact states
- `BakeModule(distribution, i)` - Create single position module with 7 ports
- `BakeUModule(d_a, d_b)` - Create undersegmentation handling module
- `parse_file(filename, hmm)` - Event detection, segmentation, underseg correction
- `analyze_events(events, hmm)` - Score events using forward-backward algorithm
- `train(model, events, threshold=0.10)` - Baum-Welch training with filter threshold
- `n_fold_cross_validation(events, n=5)` - Cross-validation returning accuracy curves

## Running the Analysis

```python
# Load pre-trained model
with open("untrained_hmm.txt") as infile:
    model = Model.read(infile)

# Load event data from JSON
from PyPore.DataTypes import File
file = File.from_json("14418004-s04.json")

# Get segment means for HMM input
events = [[seg.mean for seg in event.segments] for event in file.events]

# Analyze events
data = analyze_events(events, model)
```

## Data Format

JSON files contain pre-segmented events with structure:
```json
{
  "events": [{
    "segments": [{
      "mean": 41.23,      // Current in pA
      "std": 0.82,
      "duration": 0.17,   // Seconds
      "start": 0.001,
      "end": 0.172
    }, ...]
  }, ...]
}
```

## Hackathon Tracks

See `refrences/hackathon_plans.md` for detailed 4-track plan:
1. **Track 1**: Reproduce paper results, run ablation studies
2. **Track 2**: Compare to Nanopolish HMM architecture
3. **Track 3**: Adapt methodology for modern ONT chemistry (R9.4/R10.4)
4. **Track 4**: Apply to m6A RNA modification detection

## Important Notes

- Original code is Python 2.7 - may need `xrange` → `range`, print function updates
- YAHMM → pomegranate migration: `State` creation and `model.bake()` syntax differ
- Filter score threshold of 0.1 optimal for separating on/off-pathway events
- Paper achieves ~97% accuracy on top 26% of events (highest confidence)
