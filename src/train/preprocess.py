from typing import Sequence, List, Tuple

"""This functions learn min maxes for each column"""
#region fit_minmax
def fit_minmax(x: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    number_of_features = len(x[0])                   #number of features
    mins = [float("inf")] * number_of_features       #mins = [inf, inf, inf, inf, inf, inf, inf, inf]
    maxs = [float("-inf")] * number_of_features      #maxs = [-inf,-inf,-inf,-inf,-inf,-inf,-inf,-inf]

    for row in x:
        for column_index, value in enumerate(row):
            if value < mins[column_index]:
                mins[column_index] = value
            if value > maxs[column_index]:
                maxs[column_index] = value
    return mins, maxs
#endregion


"""This functions scale value for each column"""
#region transform minmax
def transform_minmax(
    x: Sequence[Sequence[float]],
    mins: Sequence[float],
    maxs: Sequence[float],
) -> List[List[float]]:
    out: List[List[float]] = []
    for row in x:
        new_row = []
        for index, value in enumerate(row):
            diff = maxs[index] - mins[index]
            if diff == 0:
                new_row.append(0.0)
            else:
                new_row.append((value - mins[index]) / diff)
        out.append(new_row)
    return out

#endregion
