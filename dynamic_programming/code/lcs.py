# modified from https://www.geeksforgeeks.org/longest-common-subsequence-dp-4/

from tabulate import tabulate


def lcs(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in range(m + 1)]

    """Following steps build L[m+1][n+1] in bottom up fashion
	Note: L[i][j] contains length of LCS of X[0..i-1]
	and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:  # initialization
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:  # if match
                L[i][j] = L[i - 1][j - 1] + 1
            else:  # if not match
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n], L


if __name__ == "__main__":
    x = "AGGTAB"
    y = "GXTXAYB"
    length, table = lcs(x, y)
    print("Length of LCS is", length)

    for i, c in enumerate(" " + x):
        table[i].insert(0, c)
    table = tabulate(table, headers=y, tablefmt="simple_grid")
    print(table)
