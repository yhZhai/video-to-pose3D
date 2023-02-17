# modified from https://towardsdatascience.com/dynamic-time-warping-3933f25fcdd

import math
from tabulate import tabulate


def cost_fn(v1, v2):
    return abs(v1 - v2)


def dtw(x, y):
    # find the length of the strings
    n, m = len(x), len(y)
    # declaring the array for storing the dp values and initializing
    dtw_matrix = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            dtw_matrix[i][j] = math.inf
    dtw_matrix[0][0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cost_fn(x[i - 1], y[j - 1])
            # take last min from a square box
            last_min = min(
                dtw_matrix[i - 1][j],  # insertion
                dtw_matrix[i][j - 1],  # deletion
                dtw_matrix[i - 1][j - 1],  # match
            )
            dtw_matrix[i][j] = cost + last_min
    return dtw_matrix[n][m], dtw_matrix


if __name__ == "__main__":
    x = [1, 2, 3, 7]
    y = [0, 2, 2, 2, 3, 4, 6]

    cost, table = dtw(x, y)
    print("the matching cost is", cost)

    for i, c in enumerate([" "] + x):
        table[i].insert(0, c)
    table = tabulate(table, headers=y, tablefmt="simple_grid")
    print(table)
