# Schreiber-Karplus HMM Model Summary

## The Core Problem: Variable Input → Fixed Template Alignment

### Input
Variable-length sequence of segment mean currents:
```
Event 1: [45.2, 52.1, 48.3, ..., 61.2]   →  88 segments
Event 2: [44.8, 51.9, 52.0, ..., 59.8]   →  131 segments
Event 3: [46.0, 53.2, ..., 60.1, 58.9]   →  99 segments
```

### Template
Fixed 54-position DNA sequence (known):
```
CAT-CAT-CAT │ C/mC/hmC (9 pos) │ CAT... │ C/mC/hmC (9 pos) │ T/X/CAT...
     ↓              ↓                         ↓                ↓
Positions 1-3   Positions 4-12         Positions 25-33   Positions 37-42
```

### The Mismatch
88 segments ≠ 54 positions. Why?

The nanopore/enzyme/segmentation process introduces **artifacts**:

| Artifact | Effect on Segments | More or Fewer? |
|----------|-------------------|----------------|
| Oversegmentation | 1 base → 2-3 segments | MORE segments |
| Undersegmentation | 2 bases → 1 segment | FEWER segments |
| Inserts (blips) | Noise adds segments | MORE segments |
| Deletes | Base passes too fast | FEWER segments |
| Backslips | Re-reads positions | MORE segments |
| Flicker | Enzyme stutters | MORE segments |

**The HMM's job**: Find the best alignment despite these artifacts.

---

## Key Assumptions About the Input

### 1. Segments Are Pre-Computed
The model receives SEGMENT MEANS, not raw current. Segmentation already done.
```
Raw current:  ~~~∿∿∿~~~∿∿∿~~~∿∿∿~~~
                   ↓ segmentation
Segments:     ▁▁▁▁▁▃▃▃▃▇▇▇▇▁▁▁▁▁
                   ↓ extract means
HMM input:    [45.2, 52.1, 61.3, 44.8, ...]
```

### 2. DNA Moves Forward (Mostly)
The enzyme ratchets DNA through the pore position by position.
- Primary flow: Position 1 → 2 → 3 → ... → 54
- Occasional backslips are handled but rare.

### 3. Each Position Has a Characteristic Current
The ~5 bases in the pore create a position-specific current level, modeled as Normal(μᵢ, σᵢ):
```
Position 1:  Normal(45.0, 4.0)  ← expect ~45 pA
Position 2:  Normal(52.0, 3.5)  ← expect ~52 pA
...
```

### 4. Modifications Shift the Current
At fork positions, C vs mC vs hmC produce DIFFERENT expected currents:
```
Same position:  C   → Normal(48.0, 4.0)
                mC  → Normal(52.0, 4.0)  ← +4 pA shift
                hmC → Normal(45.0, 5.0)  ← -3 pA shift, more variance
```

### 5. Artifacts Have Predictable Structure
- Oversegmentation: Same current repeated 2-3 times
- Undersegmentation: Average of two adjacent currents
- Inserts: Random current (Uniform distribution)
- Deletes: No observation for that position

---

## The Model's Building Blocks

### D-Module (Position Module)

One D-module per template position (102 total, including fork paths).

**Emitting States** (observe data):

| State | Distribution | What It Says |
|-------|-------------|--------------|
| M | Normal(μᵢ, σᵢ) | "This segment matches position i" |
| MO | Normal(μᵢ, σᵢ) with 80% self-loop | "This is ANOTHER segment for pos i" (oversegmentation) |
| I | Uniform(0, 90) | "This is NOISE - not from template" |
| MS | Normal(μᵢ, σᵢ) | "Enzyme is STUTTERING backward" (flicker) |
| ME | Normal(μᵢ, σᵢ) | "Enzyme is STUTTERING forward" (flicker) |

**Silent State**:

| State | What It Says |
|-------|--------------|
| D | "Position i was SKIPPED entirely - no observation" |

### U-Module (Undersegmentation Module)

Placed BETWEEN every pair of adjacent D-modules (107 total).

| State | Distribution | What It Says |
|-------|-------------|--------------|
| U | Normal((μᵢ+μᵢ₊₁)/2, σ_blend) | "Positions i and i+1 MERGED into one segment" |

Example:
- Pos i: Normal(45, 4)
- Pos i+1: Normal(55, 4)
- U-blend: Normal(50, 4) ← average!

### Fork Structure (Classification)

At positions where modification affects current, model FORKS:
```
                ┌── C path:   Normal(48, 4) ───┐
Previous ───────┼── mC path:  Normal(52, 4) ───┼────── Next
                └── hmC path: Normal(45, 5) ───┘
```

Each path has a DIFFERENT emission distribution. The path that best explains the observed current = the classification.

Example:
```
Observation: 51.8 pA
  P(51.8 | C path)   = Normal(48, 4).pdf(51.8) = 0.060
  P(51.8 | mC path)  = Normal(52, 4).pdf(51.8) = 0.099  ← HIGHEST
  P(51.8 | hmC path) = Normal(45, 5).pdf(51.8) = 0.033

→ More probability flows through the mC path
→ Classification: mC (methylcytosine)
```

---

## How the Pieces Connect: The Port Architecture

Each module is a "circuit board" with 7 PORTS on each side. Ports are SILENT states that route probability flow.

```
   D-module i          U-module           D-module i+1
 ┌───────────┐      ┌───────────┐      ┌───────────┐
 │           │ e1   │           │ e1   │           │
 │   [M,MO]──┼──s1──┼───────────┼──s1──┼──[M,MO]   │
 │           │ e2   │           │ e2   │           │
 │   [M,MO]──┼──s2──┼───────────┼──s2──┼──[M,MO]   │
 │           │ e3   │           │ e3   │           │
 │     ──────┼──s3──┼───[U]─────┼──s3──┼──────     │
 │           │ e4   │           │ e4   │           │
 │     ←─────┼──s4──┼───────────┼──s4──┼─────←     │
 │           │ e5   │           │ e5   │           │
 │   [MS]←───┼──s5──┼───────────┼──s5──┼───←[MS]   │
 │           │ e6   │           │ e6   │           │
 │   [ME]───→┼──s6──┼───────────┼──s6──┼→───[ME]   │
 │           │ e7   │           │ e7   │           │
 │     ←─────┼──s7──┼───────────┼──s7──┼─────←     │
 │           │      │           │      │           │
 └───────────┘      └───────────┘      └───────────┘
```

### Port Semantics

| Port | Direction | What Flows Through It |
|------|-----------|----------------------|
| 1 | → | DELETE: Skip this position, flow to next |
| 2 | → | MATCH: Normal forward progression (main path) |
| 3 | → | UNDERSEG OUT: Exit to U-module for blending |
| 4 | ← | UNDERSEG IN: Return from U-module after blend |
| 5 | ← | FLICKER BACKWARD: MS state stuttering backward |
| 6 | → | FLICKER FORWARD: ME state stuttering forward |
| 7 | ← | BACKSLIP: Enzyme slipped back to previous position |

**Key insight**: The 7 ports allow MODULAR artifact handling. Each artifact type flows through its designated port(s), keeping the main match path (port 2) clean and interpretable.

---

## The Complete Picture: Example Alignment

**Input (88 segments)**:
```
[45.2, 44.8, 52.1, 52.3, 51.9, 48.5, 73.2, 61.0, 60.5, ...]
```

**HMM Alignment**:
```
Seg 1 (45.2) → M:1   "Matched to position 1"
Seg 2 (44.8) → MO:1  "Same position, oversegmentation"
Seg 3 (52.1) → M:2   "Matched to position 2"
Seg 4 (52.3) → MO:2  "Same position, oversegmentation"
Seg 5 (51.9) → MO:2  "STILL position 2! Triple overseg"
Seg 6 (48.5) → U:2-3 "Underseg: positions 2&3 merged"
Seg 7 (73.2) → I:3   "Noise insert, ignored"
Seg 8 (61.0) → M:4   "Matched to position 4 (skipped 3!)"
Seg 9 (60.5) → M:5   "Matched to position 5"
...
```

**Template (54 positions) after alignment**:
```
Pos 1: ✓ observed (segs 1-2)
Pos 2: ✓ observed (segs 3-5 + blend in seg 6)
Pos 3: ✓ observed (blend in seg 6)
Pos 4: ✓ observed (seg 8)
Pos 5: ✓ observed (seg 9)
...
```

→ **88 segments successfully aligned to 54 positions!**

---

## Summary: What Each Piece Contributes

| Component | Allows Model to Explain |
|-----------|------------------------|
| M state | Normal 1:1 segment-to-position matching |
| MO state | Same position seen multiple times (overseg) |
| I state | Extra noise segments not from template |
| D state | Missing positions (no observation) |
| U state | Two positions merged into one observation |
| MS/ME states | Enzyme stuttering (flicker) |
| Port 7 | Backward movement (backslip) |
| Fork paths | Different modifications produce different currents |

The model's job: Find the alignment that **MAXIMIZES probability**, where probability depends on:

1. **Emission fit**: P(observation | state's distribution)
2. **Transition fit**: P(state sequence | transition probabilities)

By summing over ALL possible alignments (Forward-Backward), we get **SOFT assignments**: expected flow through each fork path = classification.

---

## State Count: 1070 Total

- **102 D-modules**: 30 linear positions × 1 + 24 fork positions × 3 paths
- **107 U-modules**: Between each pair of adjacent D-modules
- **618 emitting states**: M, MO, I, MS, ME (per D-module) + U (per U-module)
- **452 silent states**: D, M-start, M-end, ports, model start/end

The model achieves **~97% accuracy** on the top 26% most confident events!


=== WHY 1070 STATES? LET'S COUNT! ===

The model name "EpigeneticsHMM-54" tells us: 54 POSITIONS in the DNA sequence.

From build_profile() in epigenetics.py, the sequence structure is:

POSITION RANGE    TYPE              WHAT IT REPRESENTS
─────────────────────────────────────────────────────────────────
1-3               CAT (linear)      Background sequence (unzipping start)
4-12              {C,mC,hmC} FORK   First cytosine reading (9 positions)
13-15             CAT (linear)      Background
16-18             CAT (linear)      Background  
19-21             CAT (linear)      Background
22-24             CAT (linear)      Background
25-33             {C,mC,hmC} FORK   Second cytosine reading (synthesis)
34-36             CAT (linear)      Background
37-42             {T,X,CAT} FORK    Label reading (6 positions)
43-54             CAT (linear)      Background (4 × CAT = 12 positions)
─────────────────────────────────────────────────────────────────
TOTAL: 54 positions