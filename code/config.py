from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"

OUTPUT_CSV = DATA_DIR / "full_results_v2.csv"
CHECKPOINT_JSON = DATA_DIR / "checkpoint_v2.json"
LOG_FILE = DATA_DIR / "simulation_execution.log"

T = 500
WINDOW = 20
DELTA_T_VAR = 100
JSR_W = 100
BATCH_SIZE = 2000
N_JOBS = -1
N_SEEDS = 30

VAR_TEMP_CYCLE_THRESH = 1e-3
SPATIAL_VAR_CONSENSUS_THRESH = 1e-4
VAR_TEMP_DISSENSUS_THRESH = 1e-4

PENALTY_SHAPES = ["Linear", "Step", "Tanh"]
MODELS = [
    "proposed",
    "symmetric",
    "fixed_suppression",
    "anti-expert",
    "inverse-reputation",
    "stubborn-mix",
]
TOPOLOGIES = ["ER", "BA", "WS", "k-regular"]
N_VALS = [20, 50, 100, 200]
P_VALS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
GAMMA_VALS = [0.0, 0.1, 0.2, 0.5]
K_FRACTIONS = ["N", "N/2", "N/5", "N/10"]
