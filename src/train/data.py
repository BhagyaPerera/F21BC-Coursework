import csv
import random
from pathlib import Path
from typing import List, Tuple

from src.train import config

#region load dataset
def load_data(path: Path) -> Tuple[List[List[float]], List[float]]:

    if not path.exists():
        raise FileNotFoundError(f"Invalid Path: {path}")                     #check dataset path

    with path.open("r", newline="") as f:
        rows = list(csv.reader(f))                                           #read dataset


    row_start = 0                                                            #detect header
    if rows and not is_number(rows[0][0]):
        row_start = 1

    x_input: List[List[float]] = []                                               #initialize x,y lists
    y_output: List[float] = []

    for row in rows[row_start:]:
        numbers = [float(value) for value in row]
        x_input.append(numbers[:-1])  # 8 features
        y_output.append(numbers[-1])   # strength-output

    return x_input, y_output

#endregion


#region train_test_split

def train_test_split(
    x_input: List[List[float]],
    y_output: List[float],
    train_ratio: float = None,
    seed: int = None,
) -> Tuple[List[List[float]], List[List[float]], List[float], List[float]]:
    if train_ratio is None:                               #check train ratio is available
        train_ratio = config.TRAIN_RATIO
    if seed is None:
        seed = config.RANDOM_SEED                         #if seed none set random seed

    indices = list(range(len(x_input)))                   #get indices
    random.Random(seed).shuffle(indices)

    split = int(len(x_input) * train_ratio)              #compute split index for train test
    train_idx = indices[:split]                          #compute train indices
    test_idx = indices[split:]                           #compute test indices

    x_train = [x_input[i] for i in train_idx]            #get x_train list
    y_train = [y_output[i] for i in train_idx]           #get y_train list
    x_test = [x_input[i] for i in test_idx]              #get x_test list
    y_test = [y_output[i] for i in test_idx]             #get y_test list

    return x_train, x_test, y_train, y_test              #return x_train,x_test,y_train,y_test

#endregion

#region isNumber
#private method to check values are numbers
def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False

#endregion
