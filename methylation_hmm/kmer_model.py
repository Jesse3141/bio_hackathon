"""
K-mer model for emission distribution parameters.

Loads the ONT 9mer model and provides z-score lookups for
canonical and methylated bases.
"""

from typing import Dict, Tuple, Optional
from pathlib import Path


class KmerModel:
    """
    Handles 9mer current level lookups from ONT kmer model.

    The model maps each 9mer sequence to a z-scored current level.
    For R10.4.1, positions 6-7 in the 9mer have the strongest signal,
    making them optimal for modification detection.
    """

    def __init__(self, model_path: str):
        """
        Load 9mer model from file.

        Args:
            model_path: Path to 9mer_levels_v1.txt
        """
        self.model_path = Path(model_path)
        self.kmer_table: Dict[str, float] = {}
        self._load_model()

    def _load_model(self) -> None:
        """Load 9mer -> z-score mappings from file."""
        with open(self.model_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    kmer = parts[0].upper()
                    zscore = float(parts[1])
                    self.kmer_table[kmer] = zscore

        print(f"Loaded {len(self.kmer_table)} 9mer entries")

    def get_level(self, kmer: str) -> Optional[float]:
        """
        Get z-score for a 9mer.

        Args:
            kmer: 9-character DNA sequence (ACGT only)

        Returns:
            Z-scored current level, or None if kmer not found
        """
        return self.kmer_table.get(kmer.upper())

    def get_kmer_at_position(self, sequence: str, position: int) -> str:
        """
        Extract 9mer centered at position.

        For positions near edges, pads with 'N'.

        Args:
            sequence: Full reference sequence
            position: 0-indexed position to center on

        Returns:
            9-character string
        """
        # 9mer centered at position: positions [pos-4, pos+4]
        start = position - 4
        end = position + 5

        # Handle edge cases with padding
        if start < 0:
            left_pad = 'N' * abs(start)
            kmer = left_pad + sequence[0:end]
        elif end > len(sequence):
            right_pad = 'N' * (end - len(sequence))
            kmer = sequence[start:] + right_pad
        else:
            kmer = sequence[start:end]

        return kmer[:9]  # Ensure exactly 9 chars

    def get_emission_params(
        self,
        sequence: str,
        position: int,
        modification: str = 'C',
        methylation_shift: float = 0.8,
        methylation_std_factor: float = 1.2
    ) -> Tuple[float, float]:
        """
        Get Normal distribution parameters for a position.

        Args:
            sequence: Reference sequence
            position: 0-indexed position
            modification: 'C' for canonical, '5mC' for methylated
            methylation_shift: Z-score shift for methylation
            methylation_std_factor: Std deviation multiplier for methylation

        Returns:
            (mean, std) for Normal distribution
        """
        kmer = self.get_kmer_at_position(sequence, position)

        # Get canonical z-score (use 0.0 if kmer contains N or not found)
        if 'N' in kmer:
            canonical_zscore = 0.0
        else:
            canonical_zscore = self.kmer_table.get(kmer, 0.0)

        # Base std is 1.0 for z-scored data
        base_std = 1.0

        if modification == 'C':
            return (canonical_zscore, base_std)
        elif modification == '5mC':
            # Methylation typically shifts current level
            shifted_mean = canonical_zscore + methylation_shift
            shifted_std = base_std * methylation_std_factor
            return (shifted_mean, shifted_std)
        else:
            raise ValueError(f"Unknown modification: {modification}")

    def __len__(self) -> int:
        return len(self.kmer_table)

    def __contains__(self, kmer: str) -> bool:
        return kmer.upper() in self.kmer_table
