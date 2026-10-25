import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(BACKEND_DIR))


def reading(timestamp, power, presence="PRESENT", status=None):

    return {
        "timestamp": timestamp,
        "circuit_id": "c1",
        "distance_cm": 50.0 if presence == "PRESENT" else None,
        "presence": presence,
        "temperature_C": 25,
        "humidity_percent": 50,
        "voltage_V": 5.0,
        "current_A": power / 5.0,
        "power_W": power,
        "status": status
    }


@pytest.fixture
def raw_df():

    """Two sessions: steady occupied use with one spike, then a
    gap and an empty room with load on."""

    rows = [
        reading(1_000 + i, 0.8, "PRESENT", "NORMAL")
        for i in range(15)
    ]

    rows.append(reading(1_015, 6.0, "PRESENT", "OVERCONSUMPTION"))

    rows += [
        reading(5_000 + i, 9.0, "ABSENT", "IDLE_WASTE")
        for i in range(10)
    ]

    return pd.DataFrame(rows)
