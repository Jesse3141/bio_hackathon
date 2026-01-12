"""
Generic HMM for cytosine modification classification (binary or 3-way).

This module implements a single HMM with fork architecture at each cytosine position,
using pooled emission parameters from adapter sequences.

Supports two modes:
- Binary (n_classes=2): C vs 5mC classification
- 3-way (n_classes=3): C vs 5mC vs 5hmC classification

Key features:
- 8 cytosine positions (38, 50, 62, 74, 86, 98, 110, 122)
- 2 or 3 modification states per position
- Self-loops for nanopore dwelling/stuck events
- Skip transitions for missed bases
- Empirical Gaussian emissions from training data
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal


@dataclass
class ClassificationResult:
    """Result of classifying a single read."""
    read_id: str
    prediction: str  # 'C', '5mC', or '5hmC'
    prediction_idx: int  # 0, 1, or 2
    log_probs: Dict[str, float]  # {'C': ..., '5mC': ..., '5hmC': ...}
    confidence: float  # max prob - second max prob
    per_position_posteriors: Optional[Dict[int, Dict[str, float]]] = None


@dataclass
class EvaluationMetrics:
    """Metrics from 3-way classification evaluation."""
    accuracy: float
    per_class_accuracy: Dict[str, float]
    confusion_matrix: np.ndarray
    n_samples: int
    n_correct: int


class GenericHMM:
    """
    Fork HMM for C/5mC or C/5mC/5hmC classification.

    Architecture:
    - 16 states (binary) or 24 states (3-way): 8 positions x n_classes
    - Self-loops at each state for dwelling
    - Skip transitions to handle missed bases
    - Equal prior probability for each modification path

    State indexing (3-way example):
        Position 0 (pos 38):  C=0,  5mC=1,  5hmC=2
        Position 1 (pos 50):  C=3,  5mC=4,  5hmC=5
        ...
        Position 7 (pos 122): C=21, 5mC=22, 5hmC=23

    State indexing (binary):
        Position 0 (pos 38):  C=0,  5mC=1
        Position 1 (pos 50):  C=2,  5mC=3
        ...
        Position 7 (pos 122): C=14, 5mC=15
    """

    POSITIONS = [38, 50, 62, 74, 86, 98, 110, 122]
    MODIFICATIONS_3WAY = ['C', '5mC', '5hmC']
    MODIFICATIONS_BINARY = ['C', '5mC']
    MOD_TO_IDX_3WAY = {'C': 0, '5mC': 1, '5hmC': 2, 'mC': 1, 'hmC': 2}
    MOD_TO_IDX_BINARY = {'C': 0, '5mC': 1, 'mC': 1, 'control': 0, 'modified': 1}

    # Transition probabilities
    P_SELF_LOOP = 0.15  # Dwelling/stuck on base
    P_SKIP = 0.05       # Skip next position
    P_FORWARD = 0.80    # Normal progression

    def __init__(self, emission_params: Optional[Dict] = None, n_classes: int = 3):
        """
        Initialize GenericHMM.

        Args:
            emission_params: Pre-computed emission parameters dict.
                Expected format matches hmm_3way_pomegranate.json.
            n_classes: Number of modification classes. 2 for binary (C/5mC),
                3 for 3-way (C/5mC/5hmC). Default is 3.
        """
        if n_classes not in (2, 3):
            raise ValueError(f"n_classes must be 2 or 3, got {n_classes}")

        self.emission_params = emission_params
        self.n_classes = n_classes
        self.model: Optional[DenseHMM] = None
        self.is_fitted = False

        # Set class-specific attributes
        if n_classes == 2:
            self.modifications = self.MODIFICATIONS_BINARY
            self.mod_to_idx = self.MOD_TO_IDX_BINARY
        else:
            self.modifications = self.MODIFICATIONS_3WAY
            self.mod_to_idx = self.MOD_TO_IDX_3WAY

        self.n_states = len(self.POSITIONS) * self.n_classes

    @classmethod
    def from_params_json(cls, params_path: str, n_classes: int = 3) -> 'GenericHMM':
        """
        Load GenericHMM from emission parameters JSON.

        Args:
            params_path: Path to hmm_3way_pomegranate.json or similar
            n_classes: Number of modification classes (2 or 3)

        Returns:
            Initialized GenericHMM ready for prediction
        """
        with open(params_path) as f:
            params = json.load(f)

        hmm = cls(emission_params=params, n_classes=n_classes)
        hmm._build_model()
        hmm.is_fitted = True
        return hmm

    @classmethod
    def from_training_csv(cls, csv_path: str, test_split: float = 0.2,
                          n_classes: int = 3) -> Tuple['GenericHMM', pd.DataFrame]:
        """
        Build GenericHMM from signal CSV and return test data.

        Args:
            csv_path: Path to signal_at_cytosines_3way.csv
            test_split: Fraction to hold out for testing
            n_classes: Number of modification classes (2 or 3)

        Returns:
            (trained_hmm, test_dataframe)
        """
        df = pd.read_csv(csv_path)

        # Filter to relevant samples based on n_classes
        if n_classes == 2:
            df = df[df['sample'].isin(['control', '5mC', 'modified'])]

        # Compute emission parameters from data
        emission_params = cls._compute_params_from_csv(df, n_classes=n_classes)

        hmm = cls(emission_params=emission_params, n_classes=n_classes)
        hmm._build_model()

        # Create train/test split
        # Shuffle by read_id to keep all positions for a read together
        read_ids = df['read_id'].unique()
        np.random.shuffle(read_ids)
        n_test = int(len(read_ids) * test_split)
        test_read_ids = set(read_ids[:n_test])

        test_df = df[df['read_id'].isin(test_read_ids)]
        train_df = df[~df['read_id'].isin(test_read_ids)]

        # Could train further with Baum-Welch here if desired
        hmm.is_fitted = True

        return hmm, test_df

    @staticmethod
    def _compute_params_from_csv(df: pd.DataFrame, n_classes: int = 3) -> Dict:
        """Compute emission parameters from signal CSV.

        Args:
            df: DataFrame with signal data
            n_classes: Number of modification classes (2 or 3)

        Returns:
            Dict with emission parameters in pomegranate format
        """
        positions = [38, 50, 62, 74, 86, 98, 110, 122]

        # Map sample names to modification labels based on n_classes
        if n_classes == 2:
            sample_to_mod = {
                'control': 'C',
                '5mC': 'mC',
                'modified': 'mC'  # legacy name
            }
            mod_states = ['C', 'mC']
        else:
            sample_to_mod = {
                'control': 'C',
                '5mC': 'mC',
                '5hmC': 'hmC',
                'modified': 'mC'  # legacy name
            }
            mod_states = ['C', 'mC', 'hmC']

        params = {
            'type': 'pomegranate_hmm_emissions',
            'modification_states': mod_states,
            'n_classes': n_classes,
            'positions': []
        }

        for pos in positions:
            pos_data = df[df['position'] == pos]
            states = {}

            for sample_name, mod_name in sample_to_mod.items():
                if mod_name in states:
                    continue
                sample_data = pos_data[pos_data['sample'] == sample_name]
                if len(sample_data) == 0:
                    continue

                states[mod_name] = {
                    'distribution': 'NormalDistribution',
                    'parameters': {
                        'mean': float(sample_data['mean_current'].mean()),
                        'std': float(sample_data['mean_current'].std())
                    }
                }

            params['positions'].append({
                'position': pos,
                'states': states
            })

        return params

    def _build_model(self) -> None:
        """Build the DenseHMM from emission parameters."""
        if self.emission_params is None:
            raise ValueError("No emission parameters available")

        # Build distributions for all 24 states
        distributions = self._build_distributions()

        # Build transition matrix
        edges = self._build_transition_matrix()

        # Build start/end probabilities
        starts = self._build_start_probs()
        ends = self._build_end_probs()

        self.model = DenseHMM(
            distributions=distributions,
            edges=edges,
            starts=starts,
            ends=ends,
            max_iter=10,
            tol=0.001
        )

    def _build_distributions(self) -> List[Normal]:
        """Build Normal distributions for all states (16 for binary, 24 for 3-way)."""
        distributions = []

        # Modification keys in params
        mod_keys = ['C', 'mC'] if self.n_classes == 2 else ['C', 'mC', 'hmC']

        # Handle both JSON formats
        if 'positions' in self.emission_params:
            # pomegranate format
            for pos_data in self.emission_params['positions']:
                for mod in mod_keys:
                    if mod in pos_data['states']:
                        params = pos_data['states'][mod]['parameters']
                        mean = params['mean']
                        std = params['std']
                    else:
                        # Fallback: use overall average
                        mean = 800.0
                        std = 100.0

                    distributions.append(Normal(
                        means=torch.tensor([mean]),
                        covs=torch.tensor([[std ** 2]])
                    ))
        elif 'distributions' in self.emission_params:
            # circuit_board format
            for pos_data in self.emission_params['distributions']:
                for mod in mod_keys:
                    if mod in pos_data:
                        mean = pos_data[mod]['mean']
                        std = pos_data[mod]['std']
                    else:
                        mean = 800.0
                        std = 100.0

                    distributions.append(Normal(
                        means=torch.tensor([mean]),
                        covs=torch.tensor([[std ** 2]])
                    ))

        return distributions

    def _build_transition_matrix(self) -> torch.Tensor:
        """
        Build NxN transition matrix with self-loops and skips.

        N = 16 for binary, 24 for 3-way.

        Transitions from state at position i:
        - Self-loop: P_SELF_LOOP (0.15)
        - To position i+1 (n_classes states): P_FORWARD / n_classes each
        - To position i+2 (n_classes states): P_SKIP / n_classes each (skip)
        """
        n = self.n_states
        edges = torch.zeros((n, n), dtype=torch.float32)

        n_mods = self.n_classes
        n_pos = len(self.POSITIONS)

        for pos_idx in range(n_pos):
            for mod_idx in range(n_mods):
                state_idx = pos_idx * n_mods + mod_idx

                # Self-loop
                edges[state_idx, state_idx] = self.P_SELF_LOOP

                # Forward to next position (all modifications)
                if pos_idx < n_pos - 1:
                    next_pos_start = (pos_idx + 1) * n_mods
                    p_each = self.P_FORWARD / n_mods
                    for m in range(n_mods):
                        edges[state_idx, next_pos_start + m] = p_each

                # Skip to position i+2 (all modifications)
                if pos_idx < n_pos - 2:
                    skip_pos_start = (pos_idx + 2) * n_mods
                    p_each_skip = self.P_SKIP / n_mods
                    for m in range(n_mods):
                        edges[state_idx, skip_pos_start + m] = p_each_skip

        return edges

    def _build_start_probs(self) -> torch.Tensor:
        """Build start probabilities - equal probability across first n_classes states."""
        starts = torch.zeros(self.n_states, dtype=torch.float32)
        # Equal probability to start in any modification at position 0
        for i in range(self.n_classes):
            starts[i] = 1.0 / self.n_classes
        return starts

    def _build_end_probs(self) -> torch.Tensor:
        """Build end probabilities - can end from any state at last position."""
        ends = torch.zeros(self.n_states, dtype=torch.float32)
        # Last position is index 7, so states depend on n_classes
        last_pos_start = (len(self.POSITIONS) - 1) * self.n_classes
        for i in range(self.n_classes):
            ends[last_pos_start + i] = 1.0
        return ends

    def _get_state_index(self, position_idx: int, modification: str) -> int:
        """Get state index for a (position_idx, modification) pair."""
        mod_idx = self.mod_to_idx.get(modification, 0)
        return position_idx * self.n_classes + mod_idx

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict modification class for each read.

        Args:
            X: Array of shape (n_samples, 8) - current values at 8 positions

        Returns:
            Array of shape (n_samples,) with class indices (0=C, 1=5mC, 2=5hmC)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Load from params first.")

        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get class probabilities for each read.

        Uses a simplified approach: sum log-probabilities of emissions
        for each modification path through all positions.

        Args:
            X: Array of shape (n_samples, 8) - current values at 8 positions

        Returns:
            Array of shape (n_samples, n_classes) with normalized probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Load from params first.")

        n_samples = X.shape[0]
        n_mods = self.n_classes
        n_pos = len(self.POSITIONS)

        # Compute log-likelihood for each modification path
        log_probs = np.zeros((n_samples, n_mods))

        for mod_idx in range(n_mods):
            for pos_idx in range(n_pos):
                state_idx = pos_idx * n_mods + mod_idx
                dist = self.model.distributions[state_idx]

                # Get observation at this position
                obs = torch.tensor(X[:, pos_idx:pos_idx+1].astype(np.float32))

                # Add log probability
                with torch.no_grad():
                    log_p = dist.log_probability(obs).numpy()
                log_probs[:, mod_idx] += log_p.flatten()

        # Convert to probabilities via softmax
        log_probs -= log_probs.max(axis=1, keepdims=True)  # Numerical stability
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)

        return probs

    def classify_dataframe(self, df: pd.DataFrame) -> List[ClassificationResult]:
        """
        Classify reads from a DataFrame with per-read signals.

        Expects DataFrame with columns for positions and read_id.
        Can handle either wide format (columns: '38', '50', ...) or
        long format (columns: position, mean_current, read_id).

        Args:
            df: DataFrame with signal data

        Returns:
            List of ClassificationResult objects
        """
        # Check format
        pos_cols = [str(p) for p in self.POSITIONS]

        if all(col in df.columns for col in pos_cols):
            # Wide format
            return self._classify_wide_df(df)
        elif 'position' in df.columns and 'mean_current' in df.columns:
            # Long format - pivot to wide
            return self._classify_long_df(df)
        else:
            raise ValueError("DataFrame must have position columns or 'position'/'mean_current' columns")

    def _classify_wide_df(self, df: pd.DataFrame) -> List[ClassificationResult]:
        """Classify from wide-format DataFrame."""
        pos_cols = [str(p) for p in self.POSITIONS]

        # Filter valid rows
        df_valid = df.dropna(subset=pos_cols)

        X = df_valid[pos_cols].values.astype(np.float32)
        read_ids = df_valid['read_id'].values if 'read_id' in df_valid.columns else \
                   [f'read_{i}' for i in range(len(X))]

        probs = self.predict_proba(X)
        preds = np.argmax(probs, axis=1)

        results = []
        for i in range(len(X)):
            sorted_probs = np.sort(probs[i])[::-1]
            confidence = sorted_probs[0] - sorted_probs[1]

            results.append(ClassificationResult(
                read_id=str(read_ids[i]),
                prediction=self.modifications[preds[i]],
                prediction_idx=int(preds[i]),
                log_probs={m: float(np.log(probs[i, j] + 1e-10))
                          for j, m in enumerate(self.modifications)},
                confidence=float(confidence)
            ))

        return results

    def _classify_long_df(self, df: pd.DataFrame) -> List[ClassificationResult]:
        """Classify from long-format DataFrame (pivot first)."""
        # Pivot to wide format
        wide_df = df.pivot_table(
            index='read_id',
            columns='position',
            values='mean_current',
            aggfunc='mean'
        ).reset_index()

        # Rename columns to strings
        wide_df.columns = ['read_id'] + [str(c) for c in wide_df.columns[1:]]

        return self._classify_wide_df(wide_df)

    def evaluate(self, df: pd.DataFrame) -> EvaluationMetrics:
        """
        Evaluate classifier on labeled data.

        Args:
            df: DataFrame with 'sample' column (ground truth) and signal data

        Returns:
            EvaluationMetrics object
        """
        # Map sample names to class indices (use instance mod_to_idx)
        sample_to_idx = self.mod_to_idx.copy()
        # Add additional mappings
        sample_to_idx.update({'C': 0, 'control': 0, 'modified': 1})

        results = self.classify_dataframe(df)

        # Get ground truth
        if 'position' in df.columns:
            # Long format - need to aggregate per read
            read_samples = df.groupby('read_id')['sample'].first().to_dict()
            y_true = np.array([sample_to_idx.get(read_samples.get(r.read_id, 'C'), 0)
                              for r in results])
        else:
            # Wide format
            df_valid = df.dropna(subset=[str(p) for p in self.POSITIONS])
            y_true = df_valid['sample'].map(sample_to_idx).values

        y_pred = np.array([r.prediction_idx for r in results])

        # Compute metrics
        accuracy = (y_pred == y_true).mean()
        n_correct = (y_pred == y_true).sum()

        # Per-class accuracy
        per_class_acc = {}
        for idx, mod in enumerate(self.modifications):
            mask = y_true == idx
            if mask.sum() > 0:
                per_class_acc[mod] = float((y_pred[mask] == idx).mean())
            else:
                per_class_acc[mod] = 0.0

        # Confusion matrix (size depends on n_classes)
        conf_mat = np.zeros((self.n_classes, self.n_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < self.n_classes and 0 <= p < self.n_classes:
                conf_mat[t, p] += 1

        return EvaluationMetrics(
            accuracy=float(accuracy),
            per_class_accuracy=per_class_acc,
            confusion_matrix=conf_mat,
            n_samples=len(y_true),
            n_correct=int(n_correct)
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GenericHMM':
        """
        Train the HMM on labeled data using Baum-Welch.

        Args:
            X: Array of shape (n_samples, 8) - current values
            y: Array of shape (n_samples,) - class labels (0, 1, 2)

        Returns:
            self
        """
        if self.model is None:
            raise ValueError("Model not built. Load params first.")

        # Pomegranate's fit expects (n_samples, seq_len, n_features)
        X_tensor = torch.tensor(X.astype(np.float32)).unsqueeze(-1)

        self.model.fit(X_tensor)
        self.is_fitted = True

        return self

    def save(self, filepath: str) -> None:
        """Save model to file."""
        torch.save({
            'emission_params': self.emission_params,
            'n_classes': self.n_classes,
            'model_state_dict': self.model.state_dict() if self.model else None,
            'is_fitted': self.is_fitted
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'GenericHMM':
        """Load model from file."""
        checkpoint = torch.load(filepath)
        n_classes = checkpoint.get('n_classes', 3)  # Default to 3 for backwards compatibility
        hmm = cls(emission_params=checkpoint['emission_params'], n_classes=n_classes)
        hmm._build_model()
        if checkpoint['model_state_dict']:
            hmm.model.load_state_dict(checkpoint['model_state_dict'])
        hmm.is_fitted = checkpoint['is_fitted']
        return hmm

    def save_params_json(self, filepath: str) -> None:
        """Save emission parameters to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.emission_params, f, indent=2)


def run_evaluation(params_json: str, signal_csv: str, n_classes: int = 3) -> EvaluationMetrics:
    """
    Quick evaluation function.

    Args:
        params_json: Path to emission parameters JSON
        signal_csv: Path to signal_at_cytosines_3way.csv
        n_classes: Number of classes (2 for binary, 3 for 3-way)

    Returns:
        EvaluationMetrics
    """
    mode = "binary" if n_classes == 2 else "3-way"
    print(f"Loading model from {params_json} ({mode} mode)...")
    hmm = GenericHMM.from_params_json(params_json, n_classes=n_classes)

    print(f"Loading signal data from {signal_csv}...")
    df = pd.read_csv(signal_csv)

    # Filter data for binary mode
    if n_classes == 2:
        df = df[df['sample'].isin(['control', '5mC', 'modified'])]

    print(f"Evaluating on {df['read_id'].nunique()} reads...")
    metrics = hmm.evaluate(df)

    print(f"\n=== Results ({mode}) ===")
    print(f"Overall accuracy: {metrics.accuracy:.1%}")
    print(f"Per-class accuracy:")
    for mod, acc in metrics.per_class_accuracy.items():
        print(f"  {mod}: {acc:.1%}")

    # Print confusion matrix with dynamic headers
    print(f"\nConfusion matrix:")
    if n_classes == 2:
        print(f"         Pred C  Pred 5mC")
        for i, mod in enumerate(['C', '5mC']):
            print(f"True {mod:4s}  {metrics.confusion_matrix[i, 0]:6d}  "
                  f"{metrics.confusion_matrix[i, 1]:8d}")
    else:
        print(f"         Pred C  Pred 5mC  Pred 5hmC")
        for i, mod in enumerate(['C', '5mC', '5hmC']):
            print(f"True {mod:4s}  {metrics.confusion_matrix[i, 0]:6d}  "
                  f"{metrics.confusion_matrix[i, 1]:8d}  {metrics.confusion_matrix[i, 2]:9d}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GenericHMM for modification classification")
    parser.add_argument("--params", required=True, help="Path to emission params JSON")
    parser.add_argument("--signal-csv", required=True, help="Path to signal CSV")
    parser.add_argument("--output-model", help="Path to save trained model")
    parser.add_argument("--n-classes", type=int, default=3, choices=[2, 3],
                        help="Number of classes: 2 (binary C/5mC) or 3 (C/5mC/5hmC)")

    args = parser.parse_args()

    metrics = run_evaluation(args.params, args.signal_csv, n_classes=args.n_classes)

    if args.output_model:
        hmm = GenericHMM.from_params_json(args.params, n_classes=args.n_classes)
        hmm.save(args.output_model)
        print(f"\nModel saved to {args.output_model}")
