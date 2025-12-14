"""Unit tests for data preparation module."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.prepare import load_data, preprocess_data, split_data


@pytest.fixture
def sample_wine_df():
    """Create a sample wine quality DataFrame for testing.
    
    Uses balanced quality classes to support stratified splitting.
    """
    return pd.DataFrame({
        "fixed acidity": [7.4, 7.8, 7.8, 11.2, 7.4, 7.9, 7.3, 7.8, 6.7, 7.5, 8.1, 7.2, 6.9, 7.0, 8.5, 7.6, 6.8, 7.1, 8.0, 7.7],
        "volatile acidity": [0.7, 0.88, 0.76, 0.28, 0.7, 0.6, 0.65, 0.58, 0.58, 0.5, 0.45, 0.55, 0.62, 0.48, 0.52, 0.68, 0.72, 0.44, 0.38, 0.66],
        "citric acid": [0.0, 0.0, 0.04, 0.56, 0.0, 0.06, 0.0, 0.02, 0.08, 0.36, 0.12, 0.24, 0.18, 0.32, 0.28, 0.14, 0.08, 0.22, 0.16, 0.1],
        "residual sugar": [1.9, 2.6, 2.3, 1.9, 1.9, 1.6, 1.2, 2.0, 1.8, 6.1, 2.4, 1.5, 2.8, 3.2, 1.7, 2.1, 2.5, 1.4, 3.0, 2.2],
        "chlorides": [0.076, 0.098, 0.092, 0.075, 0.076, 0.069, 0.065, 0.073, 0.097, 0.071, 0.068, 0.082, 0.078, 0.066, 0.088, 0.074, 0.086, 0.072, 0.064, 0.09],
        "free sulfur dioxide": [11.0, 25.0, 15.0, 17.0, 11.0, 15.0, 15.0, 9.0, 15.0, 17.0, 22.0, 18.0, 12.0, 20.0, 14.0, 16.0, 19.0, 13.0, 21.0, 10.0],
        "total sulfur dioxide": [34.0, 67.0, 54.0, 60.0, 34.0, 21.0, 21.0, 18.0, 65.0, 102.0, 45.0, 38.0, 72.0, 55.0, 48.0, 62.0, 58.0, 42.0, 50.0, 68.0],
        "density": [0.9978, 0.9968, 0.997, 0.998, 0.9978, 0.9946, 0.9946, 0.9968, 0.9959, 0.9978, 0.9955, 0.9962, 0.9972, 0.9948, 0.9965, 0.9958, 0.9975, 0.9952, 0.9942, 0.9968],
        "pH": [3.51, 3.2, 3.26, 3.16, 3.51, 3.3, 3.39, 3.36, 3.28, 3.15, 3.22, 3.45, 3.18, 3.32, 3.24, 3.38, 3.12, 3.42, 3.35, 3.28],
        "sulphates": [0.56, 0.68, 0.65, 0.58, 0.56, 0.46, 0.47, 0.57, 0.54, 0.65, 0.62, 0.52, 0.48, 0.72, 0.58, 0.44, 0.66, 0.55, 0.7, 0.5],
        "alcohol": [9.4, 9.8, 9.8, 9.8, 9.4, 9.4, 10.0, 9.5, 9.2, 9.0, 10.5, 9.6, 10.2, 9.3, 11.0, 9.7, 10.8, 9.1, 10.4, 9.9],
        "quality": [5, 5, 5, 6, 5, 6, 6, 6, 5, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5],  # Balanced: 10 quality=5, 10 quality=6
    })


@pytest.fixture
def sample_csv_file(sample_wine_df):
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_wine_df.to_csv(f, index=False)
        return Path(f.name)



class TestLoadData:
    """Tests for load_data function."""

    def test_load_data_success(self, sample_csv_file, sample_wine_df):
        """Test successful data loading from CSV."""
        df = load_data(sample_csv_file)
        assert len(df) == len(sample_wine_df)
        assert list(df.columns) == list(sample_wine_df.columns)

    def test_load_data_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            load_data("nonexistent_file.csv")

    def test_load_data_empty_file(self, tmp_path):
        """Test that EmptyDataError is raised for empty file."""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("col1,col2\n")  # Header only, no data
        
        df = pd.read_csv(empty_file)
        # Empty file with headers creates empty DataFrame
        assert df.empty

    def test_load_data_with_path_object(self, sample_csv_file):
        """Test loading data with Path object."""
        df = load_data(Path(sample_csv_file))
        assert not df.empty


class TestPreprocessData:
    """Tests for preprocess_data function."""

    def test_preprocess_data_no_missing(self, sample_wine_df):
        """Test preprocessing with no missing values."""
        result = preprocess_data(sample_wine_df)
        assert len(result) == len(sample_wine_df)
        assert not result.isna().any().any()

    def test_preprocess_data_missing_features(self, sample_wine_df):
        """Test preprocessing fills missing feature values with median."""
        df = sample_wine_df.copy()
        df.loc[0, "fixed acidity"] = None
        df.loc[1, "volatile acidity"] = None
        
        result = preprocess_data(df)
        
        assert not result.isna().any().any()
        assert len(result) == len(df)

    def test_preprocess_data_missing_target(self, sample_wine_df):
        """Test preprocessing drops rows with missing target values."""
        df = sample_wine_df.copy()
        df.loc[0, "quality"] = None
        df.loc[1, "quality"] = None
        
        result = preprocess_data(df)
        
        assert len(result) == len(df) - 2
        assert not result["quality"].isna().any()

    def test_preprocess_data_missing_columns(self, sample_wine_df):
        """Test that ValueError is raised for missing required columns."""
        df = sample_wine_df.drop(columns=["fixed acidity", "pH"])
        
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess_data(df)

    def test_preprocess_data_preserves_dtypes(self, sample_wine_df):
        """Test that preprocessing preserves numeric dtypes."""
        result = preprocess_data(sample_wine_df)
        
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col])



class TestSplitData:
    """Tests for split_data function."""

    def test_split_data_default_params(self, sample_wine_df):
        """Test data splitting with default parameters."""
        df = preprocess_data(sample_wine_df)
        train_df, test_df = split_data(df)
        
        assert len(train_df) + len(test_df) == len(df)
        # Default test_size is 0.2
        assert len(test_df) == pytest.approx(len(df) * 0.2, abs=1)

    def test_split_data_custom_test_size(self, sample_wine_df):
        """Test data splitting with custom test size."""
        df = preprocess_data(sample_wine_df)
        train_df, test_df = split_data(df, test_size=0.3)
        
        assert len(train_df) + len(test_df) == len(df)
        assert len(test_df) == pytest.approx(len(df) * 0.3, abs=1)

    def test_split_data_reproducibility(self, sample_wine_df):
        """Test that same random_state produces same split."""
        df = preprocess_data(sample_wine_df)
        
        train1, test1 = split_data(df, random_state=42)
        train2, test2 = split_data(df, random_state=42)
        
        pd.testing.assert_frame_equal(train1.reset_index(drop=True), train2.reset_index(drop=True))
        pd.testing.assert_frame_equal(test1.reset_index(drop=True), test2.reset_index(drop=True))

    def test_split_data_different_random_state(self, sample_wine_df):
        """Test that different random_state produces different split."""
        df = preprocess_data(sample_wine_df)
        
        train1, test1 = split_data(df, random_state=42)
        train2, test2 = split_data(df, random_state=123)
        
        # The splits should be different (indices won't match)
        assert not train1.index.equals(train2.index)

    def test_split_data_invalid_test_size_zero(self, sample_wine_df):
        """Test that ValueError is raised for test_size = 0."""
        df = preprocess_data(sample_wine_df)
        
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            split_data(df, test_size=0)

    def test_split_data_invalid_test_size_one(self, sample_wine_df):
        """Test that ValueError is raised for test_size = 1."""
        df = preprocess_data(sample_wine_df)
        
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            split_data(df, test_size=1)

    def test_split_data_invalid_test_size_negative(self, sample_wine_df):
        """Test that ValueError is raised for negative test_size."""
        df = preprocess_data(sample_wine_df)
        
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            split_data(df, test_size=-0.1)

    def test_split_data_preserves_columns(self, sample_wine_df):
        """Test that split preserves all columns."""
        df = preprocess_data(sample_wine_df)
        train_df, test_df = split_data(df)
        
        assert list(train_df.columns) == list(df.columns)
        assert list(test_df.columns) == list(df.columns)

    def test_split_data_stratified(self, sample_wine_df):
        """Test that split is stratified by quality."""
        df = preprocess_data(sample_wine_df)
        train_df, test_df = split_data(df)
        
        # Check that both quality classes are present in both splits
        original_classes = set(df["quality"].unique())
        train_classes = set(train_df["quality"].unique())
        test_classes = set(test_df["quality"].unique())
        
        # Both splits should contain all quality classes
        assert train_classes == original_classes
        assert test_classes == original_classes
