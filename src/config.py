"""Project configuration placeholders."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA = DATA_ROOT / "raw"
PROCESSED_DATA = DATA_ROOT / "processed"
SPLITS_DIR = DATA_ROOT / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"

# TODO: define hyperparameters and thresholds for descriptor/matching.
DESCRIPTOR_PARAMS = {}
MATCHING_PARAMS = {}
EVAL_PARAMS = {}
