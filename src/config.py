"""Project configuration: paths, constants, hyperparameter defaults."""
from pathlib import Path
from dataclasses import dataclass, field

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"
SUBMISSIONS = OUTPUTS / "submissions"

# Competition info
TARGET_COL = "Irrigation_Need"
CLASSES = ["High", "Low", "Medium"]  # alphabetically sorted for label encoding
NUM_CLASSES = 3
ID_COL = "id"

# Feature groups
CATEGORICAL_COLS = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Region", "Mulching_Used",
]
NUMERICAL_COLS = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]


@dataclass
class ModelConfig:
    n_folds: int = 5
    random_state: int = 42
    lgb_params: dict = field(default_factory=lambda: {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "n_estimators": 3000,
        "num_leaves": 64,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
        "class_weight": "balanced",
    })
    xgb_params: dict = field(default_factory=lambda: {
        "objective": "multi:softprob",
        "num_class": 3,
        "learning_rate": 0.05,
        "n_estimators": 3000,
        "max_depth": 6,
        "min_child_weight": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": -1,
    })
