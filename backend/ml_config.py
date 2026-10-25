from pathlib import Path


# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

RAW_FILE = DATA_DIR / "c1.csv"
FEATURES_FILE = DATA_DIR / "c1_features.csv"
CLASSIFIED_FILE = DATA_DIR / "c1_classified.csv"
SHAP_FILE = DATA_DIR / "c1_shap.csv"
COUNTERFACTUAL_FILE = DATA_DIR / "c1_counterfactuals.csv"
EXPLANATIONS_FILE = DATA_DIR / "c1_explanations.csv"
LATEST_PREDICTION_FILE = DATA_DIR / "latest_prediction.json"

MODEL_BUNDLE_FILE = MODEL_DIR / "model_bundle.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"


# =====================================================
# CLASSES
# =====================================================

# NORMAL = expected use
# IE     = idle equipment (load on, nobody present)
# OC     = overconsumption (occupied, power above limit)
# UP     = unusual pattern (occupied, sudden power spike)

CLASSES = ["NORMAL", "IE", "OC", "UP"]

CLASS_NAMES = {
    "NORMAL": "Normal",
    "IE": "Idle equipment",
    "OC": "Overconsumption",
    "UP": "Unusual pattern"
}

# Firmware status (main.py on the ESP32) -> training class

STATUS_TO_CLASS = {
    "IDLE_WASTE": "IE",
    "OVERCONSUMPTION": "OC",
    "NORMAL": "NORMAL",
    "NO_LOAD": "NORMAL"
}


# =====================================================
# FIRMWARE CONSTANTS (must match main.py)
# =====================================================

LOAD_VOLTAGE = 5.0
CURRENT_NOISE_LIMIT = 0.02
OVERCONSUMPTION_LIMIT_W = 1.5


# =====================================================
# FEATURE SETTINGS
# =====================================================

ROLLING_WINDOW = 10

# A gap longer than this starts a new recording session,
# so rolling windows never mix separate sessions.
SESSION_GAP_SECONDS = 60


# =====================================================
# UNUSUAL PATTERN RULE
# =====================================================

# Occupied room and power jumps (up or down) away from the
# previous window by more than
# max(SPIKE_STD_MULTIPLIER * previous std, SPIKE_MIN_W).

SPIKE_STD_MULTIPLIER = 3.0
SPIKE_MIN_W = 1.0


# =====================================================
# MODEL FEATURES
# =====================================================

# hour / day_type / temperature / humidity are deliberately
# excluded: the data covers only six days, so they identify
# the recording session rather than the energy behaviour.

MODEL_FEATURES = [
    "power_W",
    "current_A",
    "presence_binary",
    "rolling_mean_power",
    "rolling_std_power",
    "prev_rolling_std_power",
    "power_jump",
    "power_jump_score",
    "deviation_from_baseline"
]


# =====================================================
# PIPELINE LIMITS
# =====================================================

# Most recent non-NORMAL rows explained per pipeline run
MAX_EXPLAIN_ROWS = 2000

# Rows sent to DiCE (slow); others use the model-based search
MAX_DICE_ROWS = 50
