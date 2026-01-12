"""
Tests for the simplified HMM pipeline module.
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestSimplifiedPipeline:
    """Test the SimplifiedMethylationClassifier."""

    @pytest.fixture
    def training_csv(self):
        return str(PROJECT_ROOT / "output" / "hmm_training_sequences.csv")

    @pytest.fixture
    def emission_params_json(self):
        return str(PROJECT_ROOT / "output" / "hmm_emission_params_pomegranate.json")

    def test_from_json(self, emission_params_json):
        """Test creating classifier from pre-computed params."""
        from methylation_hmm.simplified_pipeline import SimplifiedMethylationClassifier

        classifier = SimplifiedMethylationClassifier.from_json(emission_params_json)

        assert classifier.is_fitted
        assert classifier.control_model is not None
        assert classifier.modified_model is not None

    def test_from_training_data(self, training_csv):
        """Test creating and training classifier from CSV."""
        from methylation_hmm.simplified_pipeline import SimplifiedMethylationClassifier

        classifier, test_df = SimplifiedMethylationClassifier.from_training_data(
            training_csv, test_split=0.2
        )

        assert classifier.is_fitted
        assert len(test_df) > 0

    def test_predict(self, emission_params_json, training_csv):
        """Test prediction on test data."""
        import numpy as np
        import pandas as pd
        from methylation_hmm.simplified_pipeline import SimplifiedMethylationClassifier

        classifier = SimplifiedMethylationClassifier.from_json(emission_params_json)

        # Load some test data
        df = pd.read_csv(training_csv)
        df = df.dropna(subset=classifier.POSITIONS)

        X = df[classifier.POSITIONS].values[:100].astype(np.float32)

        predictions = classifier.predict(X)

        assert len(predictions) == 100
        assert set(predictions).issubset({0, 1})

    def test_evaluate(self, training_csv):
        """Test evaluation metrics."""
        from methylation_hmm.simplified_pipeline import SimplifiedMethylationClassifier

        classifier, test_df = SimplifiedMethylationClassifier.from_training_data(
            training_csv, test_split=0.2
        )

        metrics = classifier.evaluate(test_df)

        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.control_accuracy <= 1
        assert 0 <= metrics.modified_accuracy <= 1
        assert metrics.n_samples > 0
        assert metrics.n_correct >= 0

        # Should achieve >65% accuracy
        assert metrics.accuracy > 0.65, f"Accuracy {metrics.accuracy} too low"

        print(f"Accuracy: {metrics.accuracy:.3f}")

    def test_full_pipeline(self, training_csv, emission_params_json):
        """Test the full pipeline function."""
        from methylation_hmm.simplified_pipeline import run_full_pipeline

        classifier, metrics = run_full_pipeline(
            training_csv=training_csv,
            emission_params_json=emission_params_json
        )

        assert classifier.is_fitted
        assert metrics.accuracy > 0.65

    def test_classify_dataframe(self, training_csv, emission_params_json):
        """Test DataFrame classification."""
        import pandas as pd
        from methylation_hmm.simplified_pipeline import SimplifiedMethylationClassifier

        classifier = SimplifiedMethylationClassifier.from_json(emission_params_json)

        df = pd.read_csv(training_csv)
        df = df.head(50)  # Small sample

        results = classifier.classify_dataframe(df)

        assert len(results) > 0
        for r in results:
            assert r.prediction in ['control', 'modified']
            assert r.confidence >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
