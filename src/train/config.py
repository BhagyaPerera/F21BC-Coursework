from pathlib import Path

# where you will drop the CSV
ROOT_DIR = Path(__file__).resolve().parents[2]  # F21BC-Coursework
DATASET_PATH = ROOT_DIR / "src" / "data" / "concrete_data.csv"

# splits
TRAIN_RATIO: float = 0.7
RANDOM_SEED: int = 42

# scaling
USE_MINMAX: bool = True
