import subprocess
import sys
from pathlib import Path


# =====================================================
# PROJECT DIRECTORY
# =====================================================

BASE_DIR = Path(__file__).resolve().parent


# =====================================================
# PIPELINE MODULES
# =====================================================

PIPELINE = [
    "feature_engineer.py",
    "predict.py",
    "xai_module.py",
    "dice_module.py",
    "nlg_module.py"
]

MODEL_BUNDLE_FILE = BASE_DIR / "models" / "model_bundle.joblib"


# =====================================================
# RUN MODULE
# =====================================================

def run_module(filename):

    print()
    print("=" * 50)
    print(f"RUNNING: {filename}")
    print("=" * 50)

    script_path = BASE_DIR / filename

    if not script_path.exists():

        raise FileNotFoundError(
            f"File not found: {script_path}"
        )

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR
    )

    if result.returncode != 0:

        print()
        print(f"FAILED: {filename}")

        raise SystemExit(
            result.returncode
        )

    print()
    print(f"COMPLETED: {filename}")


# =====================================================
# MAIN
# =====================================================

print("======================================")
print("XAI-EnergyGuard PIPELINE RUNNER")
print("======================================")

if not MODEL_BUNDLE_FILE.exists():

    print()
    print("No trained model found:", MODEL_BUNDLE_FILE)
    print("Train it first:  python train.py")

    raise SystemExit(1)

for module in PIPELINE:

    run_module(module)


print()
print("======================================")
print("FULL PIPELINE COMPLETE")
print("======================================")