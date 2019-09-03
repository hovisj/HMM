import numpy as np


def get_hmm_parameters(stay_background, stay_bead, lambda_0_analysis, lambda_1_analysis, bin_width, num_for_bead_geometric, max_num_for_bead_geometric):
    n = 1 + num_for_bead_geometric
    max_n = 1 + max_num_for_bead_geometric

    arrayPi = get_arrayPi(n, stay_background)
    arrayMeanCounts = get_arrayMeanCounts(n, lambda_0_analysis, lambda_1_analysis, bin_width)
    matrixA = get_matrixA(n, num_for_bead_geometric, stay_background, stay_bead)

    return arrayPi, arrayMeanCounts, matrixA, max_n, n

def get_arrayPi(n, stay_background):
    arrayPi = np.zeros(n)
    arrayPi.astype(np.float64)

    arrayPi[0] = stay_background
    for i in range(n - 1):
        arrayPi[i + 1] = (1 - stay_background)/(n - 1)
    print(arrayPi)

    return arrayPi

def get_arrayMeanCounts(n, lambda_0_analysis, lambda_1_analysis, bin_width):
    arrayMeanCounts = np.zeros(n)
    arrayMeanCounts[0] = lambda_0_analysis

    for i in range(n - 1):
        arrayMeanCounts[i + 1] = lambda_1_analysis
    arrayMeanCounts = arrayMeanCounts * bin_width
    print(arrayMeanCounts)

    return arrayMeanCounts

def get_matrixA(n, num_for_bead_geometric, stay_background, stay_bead):
    matrixA = np.zeros((n, n))
    matrixA.astype(np.float64)

    matrixA[0, 0] = stay_background
    matrixA[0, 1] = 1 - stay_background

    for k in range(num_for_bead_geometric - 1):
        i = k + 1
        j = k + 2
        matrixA[i, i] = stay_bead
        matrixA[i, j] = 1 - stay_bead

    matrixA[-1, -1] = stay_bead
    matrixA[-1, 0] = 1 - stay_bead

    print(matrixA)

    return matrixA
