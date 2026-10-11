import subprocess
import sys
import time
from pathlib import Path


# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "c1.csv"
PIPELINE_FILE = BASE_DIR / "pipeline_runner.py"


# =====================================================
# SETTINGS
# =====================================================

CHECK_INTERVAL = 30


# =====================================================
# MAIN
# =====================================================

print("======================================")
print("XAI-EnergyGuard LIVE PIPELINE")
print("======================================")

print("Watching:", DATA_FILE)
print("Check interval:", CHECK_INTERVAL, "seconds")
print()


last_size = DATA_FILE.stat().st_size if DATA_FILE.exists() else 0


while True:

    try:

        if not DATA_FILE.exists():

            print("Waiting for:", DATA_FILE)

            time.sleep(CHECK_INTERVAL)

            continue


        current_size = DATA_FILE.stat().st_size


        # -------------------------------------------------
        # NEW DATA DETECTED
        # -------------------------------------------------

        if current_size > last_size:

            print()
            print("--------------------------------------")
            print("New sensor data detected.")
            print("Running pipeline...")
            print("--------------------------------------")

            result = subprocess.run(
                [
                    sys.executable,
                    str(PIPELINE_FILE)
                ],
                cwd=BASE_DIR
            )


            if result.returncode == 0:

                print()
                print("Pipeline completed successfully.")

            else:

                print()
                print(
                    "Pipeline failed with exit code:",
                    result.returncode
                )


            last_size = current_size


        time.sleep(CHECK_INTERVAL)


    except KeyboardInterrupt:

        print()
        print("Live pipeline stopped.")

        break


    except Exception as error:

        print()
        print("Live pipeline error:")
        print(error)

        time.sleep(CHECK_INTERVAL)