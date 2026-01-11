# CLAUDE.md - UCSCN Nanopore HMM Codebase

## Python 2.7 Environment

**Conda environment name:** `nanopore_py27`

```bash
# Create from exported YAML
conda env create -f environment_py27.yml

# Or manual setup
conda create -n nanopore_py27 python=2.7 -y
conda activate nanopore_py27
pip install numpy matplotlib pandas seaborn ipykernel
pip install 'cython<3.0' --force-reinstall
pip install yahmm pythonic-porin 'networkx<2.0'
conda install mysql-python -y

# Register Jupyter kernel
python -m ipykernel install --user --name nanopore_py27 --display-name "Python 2.7 (Nanopore)"
```

## Files We Created for Python 2.7 Compatibility

### `pypore_compat.py`
Compatibility layer that bypasses the broken `PyPore.DataTypes` module. Provides:
- `File` - loads pre-segmented JSON data via `File.from_json()`
- `Event` - container for event data
- `Segment` - container for segment metadata (mean, std, duration, etc.)

**Why needed:** `PyPore.cparsers` (Cython module) references a missing `core` module, causing import failures. Since our JSON data is already pre-segmented, we don't need the parsers.

### `epigenetics_patched.py`
Copy of `epigenetics.py` with one import change:
```python
# Original (broken)
from PyPore.DataTypes import *

# Patched (works)
from pypore_compat import File, Segment
```

### `environment_py27.yml`
Exported conda environment with all pinned dependencies.

## Key Version Constraints

| Package | Constraint | Reason |
|---------|------------|--------|
| `cython` | <3.0 | yahmm's .pyx files use Python 2 print syntax |
| `networkx` | <2.0 | yahmm uses deprecated `.edge` attribute removed in 2.x |
| `mysql-python` | any | PyPore imports MySQLdb unconditionally |

## What Works vs What's Broken

### Working
- `PyPore.hmm` - Pure Python, provides `HMMBoard`, distributions, profile builders
- `yahmm` - HMM library with `Model`, `State`, `NormalDistribution`, etc.
- Loading pre-segmented JSON data via `pypore_compat.File.from_json()`

### Broken (don't import)
- `PyPore.DataTypes` - triggers broken cparsers import
- `PyPore.parsers` - depends on broken cparsers
- `PyPore.cparsers` - missing `core` module dependency

## Usage

```python
# Use compatibility layer
from pypore_compat import File
from yahmm import Model
from epigenetics_patched import analyze_events

# Load HMM
with open('untrained_hmm.txt', 'r') as f:
    model = Model.read(f)

# Load events
f = File.from_json('Data/14418004-s04.json')
events = [[seg.mean for seg in event.segments] for event in f.events]

# Analyze
data = analyze_events(events, model)
```

## Running Jupyter

Run JupyterLab from any Python 3 environment and select the "Python 2.7 (Nanopore)" kernel:
```bash
conda activate bio_hack  # or any env with jupyterlab
jupyter lab
```
