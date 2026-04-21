"""Feature engineering pipeline for Playground Series S6E4.

Baseline version: Label Encoding for categorical features, keep numerical as-is.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL, ID_COL, CLASSES


def build_features(train_df, test_df):
    """Build features for train and test data.

    Args:
        train_df: Training DataFrame with target column
        test_df: Test DataFrame without target column

    Returns:
        tuple: (train_processed, test_processed, feature_cols, label_encoders)
    """
    train = train_df.copy()
    test = test_df.copy()

    # Encode target: High=0, Low=1, Medium=2 (alphabetical)
    target_le = LabelEncoder()
    target_le.fit(CLASSES)
    train["_target_encoded"] = target_le.transform(train[TARGET_COL])

    # Label encode categorical features
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        # Fit on combined train+test to handle unseen categories
        combined = pd.concat([train[col], test[col]], axis=0)
        le.fit(combined)
        train[f"{col}_encoded"] = le.transform(train[col])
        test[f"{col}_encoded"] = le.transform(test[col])
        label_encoders[col] = le

    # Build feature column list
    encoded_cat_cols = [f"{col}_encoded" for col in CATEGORICAL_COLS]
    feature_cols = NUMERICAL_COLS + encoded_cat_cols

    # Verify no missing values in features
    for col in feature_cols:
        if train[col].isna().any():
            print(f"  WARNING: {col} has {train[col].isna().sum()} NaN in train")
            train[col] = train[col].fillna(train[col].median())
        if test[col].isna().any():
            print(f"  WARNING: {col} has {test[col].isna().sum()} NaN in test")
            test[col] = test[col].fillna(train[col].median())

    print(f"  Features: {len(feature_cols)} ({len(NUMERICAL_COLS)} numeric + {len(encoded_cat_cols)} encoded categorical)")

    return train, test, feature_cols, label_encoders, target_le
