"""Data loading for Playground Series S6E4."""
import pandas as pd
from pathlib import Path

from src.config import DATA_RAW, ID_COL, TARGET_COL


def load_data():
    """Load train, test, and sample submission.

    Returns:
        tuple: (train_df, test_df, sample_submission_df)
    """
    train = pd.read_csv(DATA_RAW / "train.csv")
    test = pd.read_csv(DATA_RAW / "sample_submission.csv")  # just for IDs
    test_full = pd.read_csv(DATA_RAW / "test.csv")

    # Validate
    assert ID_COL in train.columns, f"Missing {ID_COL} in train"
    assert ID_COL in test_full.columns, f"Missing {ID_COL} in test"
    assert TARGET_COL in train.columns, f"Missing {TARGET_COL} in train"
    assert TARGET_COL not in test_full.columns, f"{TARGET_COL} should not be in test"

    # No ID overlap
    overlap = set(train[ID_COL]) & set(test_full[ID_COL])
    assert len(overlap) == 0, f"ID overlap: {len(overlap)} rows"

    print(f"Train: {train.shape}, Test: {test_full.shape}")
    print(f"Target distribution:\n{train[TARGET_COL].value_counts(normalize=True)}")

    return train, test_full
