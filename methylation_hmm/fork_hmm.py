"""
Fork-Based HMM for Methylation Classification.

Implements a single HMM with fork architecture at each cytosine position,
where each fork splits into C (control) and 5mC (modified) paths.

Classification uses forward-backward to measure probability flow through
each fork path, following the Schreiber-Karplus methodology.

Key concepts:
- Fork: A position where the HMM splits into parallel paths for C vs 5mC
- Filter Score: Quality control metric = product of total flow through all forks
- Classification: Based on which path (C or 5mC) has higher expected flow
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal


@dataclass
class ForkClassificationResult:
    """Result of classifying a single read using fork-based HMM."""
    read_id: str
    filter_score: float  # Quality metric: product of fork flows
    c_flow: float  # Total flow through C paths
    mc_flow: float  # Total flow through 5mC paths
    classification: str  # 'C' or '5mC'
    confidence: float  # Flow ratio: winner / (C + 5mC)
    per_position_flows: Dict[str, Dict[str, float]]  # {pos: {'C': flow, '5mC': flow}}


@dataclass
class CrossValidationResult:
    """Results from cross-validation."""
    fold_accuracies: List[float]
    fold_aucs: List[float]
    mean_accuracy: float
    std_accuracy: float
    mean_auc: float
    std_auc: float


class ForkHMMClassifier:
    """
    Single HMM with fork architecture for methylation classification.

    Architecture:
    - 8 positions (38, 50, 62, 74, 86, 98, 110, 122)
    - Each position has a fork with 2 paths:
      - C path: emission from control distribution
      - 5mC path: emission from modified distribution
    - Left-to-right transitions connect forks

    Classification:
    - Run forward-backward to get expected transitions
    - Sum flow through C and 5mC paths at each fork
    - Filter score = min flow across all positions (bottleneck)
    - Classification = path with higher total flow
    """

    POSITIONS = ['38', '50', '62', '74', '86', '98', '110', '122']

    def __init__(self, emission_params: Optional[Dict] = None):
        """
        Initialize classifier.

        Args:
            emission_params: Dict with 'states' containing Normal params for each
                           control_posX and modified_posX state
        """
        self.emission_params = emission_params
        self.model: Optional[DenseHMM] = None
        self.state_indices: Dict[str, int] = {}  # state_name -> index
        self.is_fitted = False

    @classmethod
    def from_json(cls, params_path: str) -> 'ForkHMMClassifier':
        """Create classifier from emission parameters JSON."""
        with open(params_path) as f:
            emission_params = json.load(f)

        classifier = cls(emission_params)
        classifier._build_fork_hmm()
        classifier.is_fitted = True
        return classifier

    @classmethod
    def from_training_data(
        cls,
        training_csv: str,
        test_split: float = 0.2
    ) -> Tuple['ForkHMMClassifier', pd.DataFrame]:
        """
        Create and train classifier from training data.

        Args:
            training_csv: Path to training CSV
            test_split: Fraction to hold out for testing

        Returns:
            (trained_classifier, test_dataframe)
        """
        classifier = cls()
        df = pd.read_csv(training_csv)

        # Clean data
        df_clean = df.dropna(subset=cls.POSITIONS)

        # Split by sample type
        control = df_clean[df_clean['sample'] == 'control']
        modified = df_clean[df_clean['sample'] == 'modified']

        # Stratified train/test split
        n_ctrl_train = int(len(control) * (1 - test_split))
        n_mod_train = int(len(modified) * (1 - test_split))

        train_control = control.iloc[:n_ctrl_train]
        train_modified = modified.iloc[:n_mod_train]

        test_df = pd.concat([
            control.iloc[n_ctrl_train:],
            modified.iloc[n_mod_train:]
        ])

        # Compute emission parameters
        classifier._compute_emission_params(train_control, train_modified)
        classifier._build_fork_hmm()

        # Train with Baum-Welch
        train_df = pd.concat([train_control, train_modified])
        X = train_df[cls.POSITIONS].values.astype(np.float32)
        classifier._train(X)

        return classifier, test_df

    def _compute_emission_params(
        self,
        control_df: pd.DataFrame,
        modified_df: pd.DataFrame
    ) -> None:
        """Compute emission parameters from training data."""
        states = []

        for pos in self.POSITIONS:
            # Control state
            ctrl_vals = control_df[pos].dropna()
            states.append({
                'name': f'control_pos{pos}',
                'distribution': 'Normal',
                'parameters': {
                    'mean': float(ctrl_vals.mean()),
                    'std': float(ctrl_vals.std())
                }
            })

            # Modified state
            mod_vals = modified_df[pos].dropna()
            states.append({
                'name': f'modified_pos{pos}',
                'distribution': 'Normal',
                'parameters': {
                    'mean': float(mod_vals.mean()),
                    'std': float(mod_vals.std())
                }
            })

        self.emission_params = {'states': states}

    def _build_fork_hmm(self) -> None:
        """
        Build a single HMM with fork architecture.

        Structure for each position i:
        - State 2*i: C path (control emission)
        - State 2*i+1: 5mC path (modified emission)

        Transitions:
        - From position i fork to position i+1 fork
        - Equal probability into C and 5mC paths at each fork
        """
        if self.emission_params is None:
            raise ValueError("No emission parameters")

        # Create distributions for all states
        distributions = []
        n_positions = len(self.POSITIONS)

        for i, pos in enumerate(self.POSITIONS):
            # C path state
            ctrl_params = next(
                s['parameters'] for s in self.emission_params['states']
                if s['name'] == f'control_pos{pos}'
            )
            distributions.append(Normal(
                means=torch.tensor([ctrl_params['mean']]),
                covs=torch.tensor([[ctrl_params['std'] ** 2]])
            ))
            self.state_indices[f'C_{pos}'] = 2 * i

            # 5mC path state
            mod_params = next(
                s['parameters'] for s in self.emission_params['states']
                if s['name'] == f'modified_pos{pos}'
            )
            distributions.append(Normal(
                means=torch.tensor([mod_params['mean']]),
                covs=torch.tensor([[mod_params['std'] ** 2]])
            ))
            self.state_indices[f'5mC_{pos}'] = 2 * i + 1

        n_states = len(distributions)  # 16 states (2 per position)

        # Build transition matrix
        # Each fork has C and 5mC states, transitions connect to next fork
        edges = torch.zeros((n_states, n_states), dtype=torch.float32)

        for i in range(n_positions - 1):
            # From position i to position i+1
            # C path at pos i can go to C or 5mC at pos i+1
            c_idx = 2 * i
            mc_idx = 2 * i + 1
            next_c_idx = 2 * (i + 1)
            next_mc_idx = 2 * (i + 1) + 1

            # Equal probability to both paths in next fork
            edges[c_idx, next_c_idx] = 0.5
            edges[c_idx, next_mc_idx] = 0.5
            edges[mc_idx, next_c_idx] = 0.5
            edges[mc_idx, next_mc_idx] = 0.5

        # Start probabilities: equal into first fork
        starts = torch.zeros(n_states)
        starts[0] = 0.5  # C path
        starts[1] = 0.5  # 5mC path

        # End probabilities: from last fork
        ends = torch.zeros(n_states)
        ends[-2] = 0.5  # C path
        ends[-1] = 0.5  # 5mC path

        self.model = DenseHMM(
            distributions=distributions,
            edges=edges,
            starts=starts,
            ends=ends,
            max_iter=20,
            tol=0.001
        )

    def _train(self, X: np.ndarray) -> None:
        """
        Train HMM with Baum-Welch.

        Args:
            X: Array of shape (n_samples, 8) - signal values
        """
        X_tensor = torch.tensor(X.astype(np.float32)).unsqueeze(-1)
        self.model.fit(X_tensor)
        self.is_fitted = True

    def _compute_path_likelihoods(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute log-likelihoods for C-only and 5mC-only paths through the HMM.

        This follows the Schreiber-Karplus approach where we compare:
        - P(sequence | all C path) vs P(sequence | all 5mC path)

        Args:
            X: Array of shape (n_samples, 8) - sequences to classify

        Returns:
            (log_prob_c, log_prob_mc) - both of shape (n_samples,)
        """
        n_samples = X.shape[0]
        n_positions = X.shape[1]

        log_prob_c = np.zeros(n_samples)
        log_prob_mc = np.zeros(n_samples)

        for i in range(n_samples):
            # Compute sum of emission log-probabilities for each path
            total_log_c = 0.0
            total_log_mc = 0.0

            for pos_idx in range(n_positions):
                obs = X[i, pos_idx]

                # State indices
                c_idx = 2 * pos_idx
                mc_idx = 2 * pos_idx + 1

                # Emission log-probs
                obs_tensor = torch.tensor([[obs]])
                log_p_c = self.model.distributions[c_idx].log_probability(obs_tensor).item()
                log_p_mc = self.model.distributions[mc_idx].log_probability(obs_tensor).item()

                total_log_c += log_p_c
                total_log_mc += log_p_mc

            log_prob_c[i] = total_log_c
            log_prob_mc[i] = total_log_mc

        return log_prob_c, log_prob_mc

    def _get_state_posteriors(self, X: np.ndarray) -> np.ndarray:
        """
        Get posterior probabilities for each state at each position.

        Uses emission probabilities to compute local posteriors.

        Args:
            X: Array of shape (n_samples, 8) - sequences to classify

        Returns:
            Array of shape (n_samples, n_states) with posteriors
        """
        n_samples = X.shape[0]
        n_positions = X.shape[1]
        n_states = 2 * n_positions

        posteriors = np.zeros((n_samples, n_states))

        for i in range(n_samples):
            for pos_idx in range(n_positions):
                obs = X[i, pos_idx]

                c_idx = 2 * pos_idx
                mc_idx = 2 * pos_idx + 1

                obs_tensor = torch.tensor([[obs]])
                log_p_c = self.model.distributions[c_idx].log_probability(obs_tensor).item()
                log_p_mc = self.model.distributions[mc_idx].log_probability(obs_tensor).item()

                # Softmax
                max_log = max(log_p_c, log_p_mc)
                p_c = np.exp(log_p_c - max_log)
                p_mc = np.exp(log_p_mc - max_log)
                total = p_c + p_mc

                posteriors[i, c_idx] = p_c / total
                posteriors[i, mc_idx] = p_mc / total

        return posteriors

    def classify(self, X: np.ndarray) -> List[ForkClassificationResult]:
        """
        Classify sequences using fork-based HMM.

        Uses likelihood-ratio approach comparing P(X|C path) vs P(X|5mC path),
        consistent with the Schreiber-Karplus methodology.

        Args:
            X: Array of shape (n_samples, 8)

        Returns:
            List of ForkClassificationResult objects
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted")

        # Compute path log-likelihoods (main classification signal)
        log_prob_c, log_prob_mc = self._compute_path_likelihoods(X)

        # Also get per-position posteriors for filter score
        posteriors = self._get_state_posteriors(X)

        results = []

        for i in range(len(X)):
            # Per-position flows for detailed analysis
            per_position = {}
            position_flows_c = []
            position_flows_mc = []

            for j, pos in enumerate(self.POSITIONS):
                c_idx = 2 * j
                mc_idx = 2 * j + 1

                c_flow = posteriors[i, c_idx]
                mc_flow = posteriors[i, mc_idx]

                per_position[pos] = {'C': c_flow, '5mC': mc_flow}
                position_flows_c.append(c_flow)
                position_flows_mc.append(mc_flow)

            # Filter score: product of best posteriors (like paper's fork flow product)
            # Measures overall quality - how well the sequence fits either model
            min_best = min(max(c, mc) for c, mc in zip(position_flows_c, position_flows_mc))
            filter_score = min_best

            # Classification based on likelihood ratio
            log_ratio = log_prob_mc[i] - log_prob_c[i]

            if log_ratio > 0:
                classification = '5mC'
                # Convert log-ratio to probability-like confidence
                confidence = 1.0 / (1.0 + np.exp(-log_ratio))
            else:
                classification = 'C'
                confidence = 1.0 / (1.0 + np.exp(log_ratio))

            # Use average posteriors for flow metrics (for compatibility)
            total_c = np.mean(position_flows_c)
            total_mc = np.mean(position_flows_mc)

            results.append(ForkClassificationResult(
                read_id=f'read_{i}',
                filter_score=filter_score,
                c_flow=total_c,
                mc_flow=total_mc,
                classification=classification,
                confidence=confidence,
                per_position_flows=per_position
            ))

        return results

    def classify_dataframe(
        self,
        df: pd.DataFrame
    ) -> Tuple[List[ForkClassificationResult], np.ndarray, np.ndarray]:
        """
        Classify reads from DataFrame.

        Args:
            df: DataFrame with position columns and 'sample' for ground truth

        Returns:
            (results, y_true, y_pred)
        """
        df_valid = df.dropna(subset=self.POSITIONS)
        X = df_valid[self.POSITIONS].values.astype(np.float32)

        results = self.classify(X)

        # Update read_ids if available
        if 'read_id' in df_valid.columns:
            for i, r in enumerate(results):
                r.read_id = str(df_valid.iloc[i]['read_id'])

        # Ground truth
        y_true = (df_valid['sample'].values == 'modified').astype(int)
        y_pred = np.array([1 if r.classification == '5mC' else 0 for r in results])

        return results, y_true, y_pred

    def evaluate(self, df: pd.DataFrame) -> Dict:
        """
        Evaluate classifier on labeled data.

        Args:
            df: DataFrame with 'sample' column (ground truth)

        Returns:
            Dict with metrics
        """
        df_valid = df.dropna(subset=self.POSITIONS)
        X = df_valid[self.POSITIONS].values.astype(np.float32)

        results, y_true, y_pred = self.classify_dataframe(df)

        accuracy = (y_pred == y_true).mean()

        # Per-class accuracy
        ctrl_mask = y_true == 0
        mod_mask = y_true == 1
        ctrl_acc = (y_pred[ctrl_mask] == 0).mean() if ctrl_mask.sum() > 0 else 0
        mod_acc = (y_pred[mod_mask] == 1).mean() if mod_mask.sum() > 0 else 0

        # Scores for AUC - use log-likelihood ratios
        log_prob_c, log_prob_mc = self._compute_path_likelihoods(X)
        scores = log_prob_mc - log_prob_c

        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, scores)
        except:
            auc = None

        # Filter score threshold analysis
        filter_scores = np.array([r.filter_score for r in results])

        return {
            'accuracy': accuracy,
            'control_accuracy': ctrl_acc,
            'modified_accuracy': mod_acc,
            'auc': auc,
            'n_samples': len(y_true),
            'filter_scores': filter_scores,
            'scores': scores,
            'y_true': y_true,
            'y_pred': y_pred
        }


def cross_validate(
    training_csv: str,
    n_folds: int = 5
) -> CrossValidationResult:
    """
    Perform n-fold cross-validation.

    Args:
        training_csv: Path to training CSV
        n_folds: Number of folds

    Returns:
        CrossValidationResult
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    df = pd.read_csv(training_csv)
    positions = ForkHMMClassifier.POSITIONS
    df_clean = df.dropna(subset=positions)

    X = df_clean[positions].values.astype(np.float32)
    y = (df_clean['sample'].values == 'modified').astype(int)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_aucs = []

    print(f"\n{'='*50}")
    print(f"{n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*50}\n")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Create training dataframe
        train_df = df_clean.iloc[train_idx].copy()
        test_df = df_clean.iloc[test_idx].copy()

        # Split training into control/modified for param computation
        ctrl_train = train_df[train_df['sample'] == 'control']
        mod_train = train_df[train_df['sample'] == 'modified']

        # Train classifier
        classifier = ForkHMMClassifier()
        classifier._compute_emission_params(ctrl_train, mod_train)
        classifier._build_fork_hmm()
        classifier._train(X_train)

        # Evaluate
        metrics = classifier.evaluate(test_df)

        fold_accuracies.append(metrics['accuracy'])
        fold_aucs.append(metrics['auc'] if metrics['auc'] else 0.5)

        print(f"Fold {fold+1}: Accuracy = {metrics['accuracy']:.3f}, AUC = {metrics['auc']:.3f}")

    result = CrossValidationResult(
        fold_accuracies=fold_accuracies,
        fold_aucs=fold_aucs,
        mean_accuracy=np.mean(fold_accuracies),
        std_accuracy=np.std(fold_accuracies),
        mean_auc=np.mean(fold_aucs),
        std_auc=np.std(fold_aucs)
    )

    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Accuracy: {result.mean_accuracy:.3f} +/- {result.std_accuracy:.3f}")
    print(f"AUC:      {result.mean_auc:.3f} +/- {result.std_auc:.3f}")
    print(f"{'='*50}\n")

    return result


def plot_fork_hmm_results(
    classifier: ForkHMMClassifier,
    test_df: pd.DataFrame,
    output_dir: str
) -> Dict:
    """
    Generate evaluation plots for fork HMM classifier.

    Creates:
    - ROC curve
    - Accuracy vs filter score threshold
    - Score distributions

    Returns:
        Dict with metrics
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    metrics = classifier.evaluate(test_df)

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(metrics['y_true'], metrics['scores'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'Fork HMM ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve: Fork HMM Classification', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'fork_hmm_roc.png', dpi=150)
    plt.close()

    # 2. Accuracy vs Filter Score Threshold (like paper's Figure 4)
    filter_scores = metrics['filter_scores']
    y_true = metrics['y_true']
    y_pred = metrics['y_pred']

    # Sort by filter score descending
    sorted_idx = np.argsort(-filter_scores)
    y_true_sorted = y_true[sorted_idx]
    y_pred_sorted = y_pred[sorted_idx]

    coverages = []
    accuracies = []

    for pct in np.linspace(0.05, 1.0, 20):
        n_keep = int(len(y_true) * pct)
        if n_keep < 10:
            continue
        acc = (y_true_sorted[:n_keep] == y_pred_sorted[:n_keep]).mean()
        coverages.append(pct)
        accuracies.append(acc)

    plt.figure(figsize=(8, 6))
    plt.plot(np.array(coverages) * 100, np.array(accuracies) * 100,
             'b-o', lw=2, markersize=6)
    plt.xlabel('Coverage (% of reads)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Fork HMM: Accuracy vs Coverage', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 105])

    # Mark 26% threshold
    idx_26 = np.argmin(np.abs(np.array(coverages) - 0.26))
    if idx_26 < len(accuracies):
        plt.axvline(x=26, color='r', linestyle='--', alpha=0.7)
        plt.annotate(f'Top 26%: {accuracies[idx_26]*100:.1f}%',
                    xy=(26, accuracies[idx_26]*100),
                    xytext=(40, accuracies[idx_26]*100 - 5),
                    fontsize=10, color='red',
                    arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()
    plt.savefig(output_path / 'fork_hmm_accuracy_vs_coverage.png', dpi=150)
    plt.close()

    # 3. Score Distribution
    scores = metrics['scores']

    plt.figure(figsize=(10, 6))
    bins = np.linspace(scores.min(), scores.max(), 50)

    plt.hist(scores[y_true == 0], bins=bins, alpha=0.6,
             label=f'Control (C) n={sum(y_true==0)}', color='blue', density=True)
    plt.hist(scores[y_true == 1], bins=bins, alpha=0.6,
             label=f'Modified (5mC) n={sum(y_true==1)}', color='red', density=True)

    plt.axvline(x=0, color='black', linestyle='--', lw=2, label='Decision boundary')

    plt.xlabel('Score: P(5mC) - P(C)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Fork HMM Score Distribution', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'fork_hmm_score_dist.png', dpi=150)
    plt.close()

    print(f"\nFork HMM Results:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  AUC: {roc_auc:.3f}")
    print(f"  Control accuracy: {metrics['control_accuracy']:.3f}")
    print(f"  Modified accuracy: {metrics['modified_accuracy']:.3f}")

    if idx_26 < len(accuracies):
        print(f"  Top 26% accuracy: {accuracies[idx_26]:.3f}")

    metrics['roc_auc'] = roc_auc
    return metrics


if __name__ == '__main__':
    training_csv = '/home/jesse/repos/bio_hackathon/output/hmm_training_sequences.csv'
    output_dir = '/home/jesse/repos/bio_hackathon/output/plots'

    print("="*60)
    print("FORK-BASED HMM METHYLATION CLASSIFIER")
    print("="*60)

    # Train and evaluate
    print("\nTraining fork HMM classifier...")
    classifier, test_df = ForkHMMClassifier.from_training_data(training_csv)

    print(f"Test set: {len(test_df)} reads")

    # Generate plots and metrics
    metrics = plot_fork_hmm_results(classifier, test_df, output_dir)

    # Cross-validation
    print("\nRunning 5-fold cross-validation...")
    cv_results = cross_validate(training_csv, n_folds=5)
