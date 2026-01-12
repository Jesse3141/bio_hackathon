"""
Emission parameter computation for full-sequence HMM classifiers.

Computes Gaussian emission parameters (mean, std) from signal data.
Supports two sources:
- Single adapter: Context-specific parameters (lower variance)
- Pooled: All adapters (higher variance due to 5-mer context mixing)

See data_processing.ipynb for explanation of the variance difference.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class EmissionParams:
    """Gaussian emission parameters for a single state."""
    mean: float
    std: float
    count: int


@dataclass
class PositionEmissions:
    """Emission parameters for all modification states at one position."""
    position: int
    base: str
    C: EmissionParams
    mC: Optional[EmissionParams] = None
    hmC: Optional[EmissionParams] = None

    def to_dict(self) -> Dict:
        result = {
            'position': self.position,
            'base': self.base,
            'C': asdict(self.C),
        }
        if self.mC:
            result['mC'] = asdict(self.mC)
        if self.hmC:
            result['hmC'] = asdict(self.hmC)
        return result


@dataclass
class FullSequenceEmissionParams:
    """Complete emission parameters for full-sequence HMM."""
    source: str  # 'single' or 'pooled'
    adapter: Optional[str]  # Adapter name if single, None if pooled
    mode: str  # 'binary' or '3way'
    sequence_length: int
    cytosine_positions: List[int]
    positions: List[PositionEmissions]

    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'adapter': self.adapter,
            'mode': self.mode,
            'sequence_length': self.sequence_length,
            'cytosine_positions': self.cytosine_positions,
            'positions': [p.to_dict() for p in self.positions],
        }

    def save(self, path: str) -> None:
        """Save parameters to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'FullSequenceEmissionParams':
        """Load parameters from JSON file."""
        with open(path) as f:
            data = json.load(f)

        positions = []
        for p_data in data['positions']:
            c_params = EmissionParams(**p_data['C'])
            mc_params = EmissionParams(**p_data['mC']) if 'mC' in p_data else None
            hmc_params = EmissionParams(**p_data['hmC']) if 'hmC' in p_data else None

            positions.append(PositionEmissions(
                position=p_data['position'],
                base=p_data['base'],
                C=c_params,
                mC=mc_params,
                hmC=hmc_params,
            ))

        return cls(
            source=data['source'],
            adapter=data.get('adapter'),
            mode=data['mode'],
            sequence_length=data['sequence_length'],
            cytosine_positions=data['cytosine_positions'],
            positions=positions,
        )

    def get_position_params(self, position: int) -> Optional[PositionEmissions]:
        """Get emission parameters for a specific position."""
        for p in self.positions:
            if p.position == position:
                return p
        return None

    def summary(self) -> Dict:
        """Get summary statistics."""
        c_means = []
        mc_means = []
        hmc_means = []
        c_stds = []
        mc_stds = []
        deltas_mc = []
        deltas_hmc = []

        for p in self.positions:
            if p.position in self.cytosine_positions:
                c_means.append(p.C.mean)
                c_stds.append(p.C.std)
                if p.mC:
                    mc_means.append(p.mC.mean)
                    mc_stds.append(p.mC.std)
                    deltas_mc.append(p.mC.mean - p.C.mean)
                if p.hmC:
                    hmc_means.append(p.hmC.mean)
                    deltas_hmc.append(p.hmC.mean - p.C.mean)

        return {
            'source': self.source,
            'mode': self.mode,
            'n_positions': len(self.positions),
            'n_cytosines': len(self.cytosine_positions),
            'mean_C_current': np.mean(c_means) if c_means else 0,
            'mean_C_std': np.mean(c_stds) if c_stds else 0,
            'mean_delta_5mC_C': np.mean(deltas_mc) if deltas_mc else 0,
            'mean_delta_5hmC_C': np.mean(deltas_hmc) if deltas_hmc else 0,
        }


CYTOSINE_POSITIONS = [38, 50, 62, 74, 86, 98, 110, 122]
SEQUENCE_LENGTH = 155


def compute_emission_params_from_full_csv(
    csv_path: str,
    adapter: Optional[str] = None,
    mode: str = 'binary',
) -> FullSequenceEmissionParams:
    """
    Compute emission parameters from signal_full_sequence.csv.

    Args:
        csv_path: Path to signal_full_sequence.csv
        adapter: Adapter name for single-adapter mode (e.g., '5mers_rand_ref_adapter_01')
                 None for pooled mode (all adapters)
        mode: 'binary' (C/5mC) or '3way' (C/5mC/5hmC)

    Returns:
        FullSequenceEmissionParams with Gaussian parameters for all positions
    """
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filter by adapter if specified
    if adapter:
        df = df[df['chrom'] == adapter]
        source = 'single'
        print(f"Filtered to adapter: {adapter} ({len(df):,} rows)")
    else:
        source = 'pooled'
        print(f"Using all adapters ({len(df):,} rows)")

    # Filter samples based on mode
    if mode == 'binary':
        df = df[df['sample'].isin(['control', '5mC'])]
    else:
        df = df[df['sample'].isin(['control', '5mC', '5hmC'])]

    print(f"Computing emission parameters for {mode} mode...")

    # Get unique positions and their bases
    positions_info = df.groupby('position').agg({
        'base': 'first'  # Get the reference base at each position
    }).reset_index()

    position_params = []

    for _, row in positions_info.iterrows():
        pos = int(row['position'])
        base = str(row['base'])

        pos_data = df[df['position'] == pos]

        # Control (C) parameters
        control_data = pos_data[pos_data['sample'] == 'control']['mean_current']
        c_params = EmissionParams(
            mean=float(control_data.mean()),
            std=float(control_data.std()),
            count=int(len(control_data)),
        )

        # 5mC parameters
        mc_params = None
        mc_data = pos_data[pos_data['sample'] == '5mC']['mean_current']
        if len(mc_data) > 0:
            mc_params = EmissionParams(
                mean=float(mc_data.mean()),
                std=float(mc_data.std()),
                count=int(len(mc_data)),
            )

        # 5hmC parameters (only for 3-way mode)
        hmc_params = None
        if mode == '3way':
            hmc_data = pos_data[pos_data['sample'] == '5hmC']['mean_current']
            if len(hmc_data) > 0:
                hmc_params = EmissionParams(
                    mean=float(hmc_data.mean()),
                    std=float(hmc_data.std()),
                    count=int(len(hmc_data)),
                )

        position_params.append(PositionEmissions(
            position=pos,
            base=base,
            C=c_params,
            mC=mc_params,
            hmC=hmc_params,
        ))

    # Sort by position
    position_params.sort(key=lambda x: x.position)

    return FullSequenceEmissionParams(
        source=source,
        adapter=adapter,
        mode=mode,
        sequence_length=SEQUENCE_LENGTH,
        cytosine_positions=CYTOSINE_POSITIONS,
        positions=position_params,
    )


def compute_emission_params_from_cytosine_csv(
    csv_path: str,
    adapter: Optional[str] = None,
    mode: str = 'binary',
) -> Dict:
    """
    Compute emission parameters from signal_at_cytosines_3way.csv.

    This returns parameters only for the 8 cytosine positions.
    Use this for sparse HMM models.

    Args:
        csv_path: Path to signal_at_cytosines_3way.csv
        adapter: Adapter name for single-adapter mode, None for pooled
        mode: 'binary' or '3way'

    Returns:
        Dict with emission parameters in circuit_board format
    """
    df = pd.read_csv(csv_path)

    if adapter:
        df = df[df['chrom'] == adapter]

    if mode == 'binary':
        df = df[df['sample'].isin(['control', '5mC'])]

    distributions = []

    for pos in CYTOSINE_POSITIONS:
        pos_data = df[df['position'] == pos]

        dist = {'position': pos}

        # Control
        control_data = pos_data[pos_data['sample'] == 'control']['mean_current']
        dist['C'] = {
            'mean': float(control_data.mean()),
            'std': float(control_data.std()),
            'count': int(len(control_data)),
        }

        # 5mC
        mc_data = pos_data[pos_data['sample'] == '5mC']['mean_current']
        if len(mc_data) > 0:
            dist['mC'] = {
                'mean': float(mc_data.mean()),
                'std': float(mc_data.std()),
                'count': int(len(mc_data)),
            }

        # 5hmC
        if mode == '3way':
            hmc_data = pos_data[pos_data['sample'] == '5hmC']['mean_current']
            if len(hmc_data) > 0:
                dist['hmC'] = {
                    'mean': float(hmc_data.mean()),
                    'std': float(hmc_data.std()),
                    'count': int(len(hmc_data)),
                }

        distributions.append(dist)

    return {
        'description': f'{mode} cytosine modification HMM emission parameters',
        'source': 'single' if adapter else 'pooled',
        'adapter': adapter,
        'mode': mode,
        'samples': ['control', '5mC'] if mode == 'binary' else ['control', '5mC', '5hmC'],
        'distributions': distributions,
    }


def compare_emission_sources(
    full_csv_path: str,
    single_adapter: str = '5mers_rand_ref_adapter_01',
    mode: str = 'binary',
) -> Tuple[FullSequenceEmissionParams, FullSequenceEmissionParams]:
    """
    Compute and compare emission parameters from single vs pooled sources.

    Args:
        full_csv_path: Path to signal_full_sequence.csv
        single_adapter: Adapter name for single-adapter mode
        mode: 'binary' or '3way'

    Returns:
        Tuple of (single_params, pooled_params)
    """
    single = compute_emission_params_from_full_csv(
        full_csv_path,
        adapter=single_adapter,
        mode=mode,
    )
    pooled = compute_emission_params_from_full_csv(
        full_csv_path,
        adapter=None,
        mode=mode,
    )

    # Print comparison
    print("\n" + "=" * 70)
    print("EMISSION PARAMETER COMPARISON")
    print("=" * 70)

    single_summary = single.summary()
    pooled_summary = pooled.summary()

    print(f"\n{'Metric':<30} {'Single Adapter':<20} {'Pooled':<20}")
    print("-" * 70)
    print(f"{'Source':<30} {single_summary['source']:<20} {pooled_summary['source']:<20}")
    print(f"{'Mode':<30} {single_summary['mode']:<20} {pooled_summary['mode']:<20}")
    print(f"{'Mean C current (pA)':<30} {single_summary['mean_C_current']:<20.1f} {pooled_summary['mean_C_current']:<20.1f}")
    print(f"{'Mean C std (pA)':<30} {single_summary['mean_C_std']:<20.1f} {pooled_summary['mean_C_std']:<20.1f}")
    print(f"{'Mean Δ 5mC-C (pA)':<30} {single_summary['mean_delta_5mC_C']:<20.1f} {pooled_summary['mean_delta_5mC_C']:<20.1f}")
    if mode == '3way':
        print(f"{'Mean Δ 5hmC-C (pA)':<30} {single_summary['mean_delta_5hmC_C']:<20.1f} {pooled_summary['mean_delta_5hmC_C']:<20.1f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    std_reduction = (pooled_summary['mean_C_std'] - single_summary['mean_C_std']) / pooled_summary['mean_C_std'] * 100
    delta_increase = (single_summary['mean_delta_5mC_C'] - pooled_summary['mean_delta_5mC_C']) / pooled_summary['mean_delta_5mC_C'] * 100
    print(f"  Single-adapter std is {std_reduction:.1f}% lower than pooled")
    print(f"  Single-adapter Δ(5mC-C) is {delta_increase:.1f}% larger than pooled")
    print("  → Single-adapter should have better class separation")
    print("=" * 70)

    return single, pooled


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute emission parameters")
    parser.add_argument("--full-csv", required=True, help="Path to signal_full_sequence.csv")
    parser.add_argument("--adapter", default=None, help="Adapter for single mode (None = pooled)")
    parser.add_argument("--mode", choices=["binary", "3way"], default="binary")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--compare", action="store_true", help="Compare single vs pooled")

    args = parser.parse_args()

    if args.compare:
        single, pooled = compare_emission_sources(args.full_csv, mode=args.mode)
        if args.output:
            single.save(args.output.replace('.json', '_single.json'))
            pooled.save(args.output.replace('.json', '_pooled.json'))
    else:
        params = compute_emission_params_from_full_csv(
            args.full_csv,
            adapter=args.adapter,
            mode=args.mode,
        )
        print("\nSummary:")
        for k, v in params.summary().items():
            print(f"  {k}: {v}")

        if args.output:
            params.save(args.output)
            print(f"\nSaved to {args.output}")
