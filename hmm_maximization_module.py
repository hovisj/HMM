import numpy as np
from numba import jit
import math

@jit(nopython=True)
def gammas(forward_likelihood, backwards_likelihood, data, matrixA, arrayMeanCounts):
    number_of_states = len(arrayMeanCounts)
    number_time_points = len(data)

    gamma_ki = np.zeros((number_time_points, number_of_states))
    gamma_ijk = np.zeros((number_of_states, number_of_states, number_time_points))
    gamma_ki.astype(np.float64)
    gamma_ijk.astype(np.float64)
    for k in range(number_time_points - 1):
        fact = math.gamma(data[k + 1] + 1)
        for i in range(number_of_states):
            gamma_ki[k, i] = 0
            for j in range(number_of_states):
                alphak_ofi = forward_likelihood[k, i]
                a_ij = matrixA[i, j]
                bj_ofdata_kplus1 = ((arrayMeanCounts[j])**data[k + 1] * np.exp(-arrayMeanCounts[j])/fact)
                betakplus1_ofj = backwards_likelihood[k + 1, j]

                gamma_ijk[i, j, k] = alphak_ofi * a_ij * bj_ofdata_kplus1 * betakplus1_ofj
                gamma_ki[k, i] = gamma_ki[k, i] + gamma_ijk[i, j, k]

    # Special Case, last row of gamma_ki
    gamma_ki[-1, :] = forward_likelihood[-1, :]

    return gamma_ijk, gamma_ki

@jit(nopython=True)
def re_estimate(data, gamma_ijk, gamma_ki):
    number_of_states = len(gamma_ki[0, :])
    number_time_points = len(gamma_ijk[0, 0, :])

    # arrayPi
    new_arrayPi = gamma_ki[0, :]

    # matrixA
    new_matrixA = np.zeros((number_of_states, number_of_states))
    new_matrixA.astype(np.float64)
    for i in range(number_of_states):
        denom = 0
        for k in range(number_time_points - 1):
            denom = denom + gamma_ki[k, i]

        for j in range(number_of_states):
            numer = 0
            for k in range(number_time_points - 1):
                numer = numer + gamma_ijk[i, j, k]

            new_matrixA[i, j] = numer/denom

    # B
    new_arrayMeanCounts = np.zeros(number_of_states)
    new_arrayMeanCounts.astype(np.float64)
    for i in range(number_of_states):
        denom = 0
        numer = 0
        for k in range(number_time_points - 1):
            denom = denom + gamma_ki[k, i]
            numer = numer + data[k] * gamma_ki[k, i]

        new_arrayMeanCounts[i] = numer/denom

    return new_arrayPi, new_matrixA, new_arrayMeanCounts

def maximization(forward_likelihood, backwards_likelihood, data, matrixA, arrayMeanCounts):
    print("Baum-Welch: Maximization")
    gamma_ijk, gamma_ki = gammas(forward_likelihood, backwards_likelihood, data, matrixA, arrayMeanCounts)
    arrayPi, matrixA, arrayMeanCounts = re_estimate(data, gamma_ijk, gamma_ki)

    return arrayPi, matrixA, arrayMeanCounts