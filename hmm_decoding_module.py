import numpy as np
import hmm_forward_backwards_module
import hmm_changepoints_module
import hmm_outputs_module
from numba import jit
import math


def decode_by_smoothers_plus_changepoints_compositebead_statistics(n, forward_likelihood, backwards_likelihood, num_for_bead_geometric, time, data, max_num_bins_bead, background_mean_counts, num_iteration, bin_width):
    decoded = by_smoothers(n, forward_likelihood, backwards_likelihood)
    changepoint_index = hmm_changepoints_module.get_changepoint_time_index_direction(decoded, time)
    all_beads_plus_dwell, composite_bead_array, num_beads_array = hmm_outputs_module.composite_bead(data, max_num_bins_bead, background_mean_counts, changepoint_index, num_iteration, num_for_bead_geometric)
    stats_all_transitions = hmm_outputs_module.statistics_all_transitions(changepoint_index, bin_width)

    return all_beads_plus_dwell, composite_bead_array, num_beads_array, stats_all_transitions

@jit(nopython=True)
def by_smoothers(n, forward_likelihood, backwards_likelihood):
    print("Smoothers")
    decoded = np.zeros(len(forward_likelihood))
    decoded.astype(np.int8)

    number_of_iterations = range(len(forward_likelihood) - 1)
    for k in number_of_iterations:
        array = np.zeros(n)
        for i in range(n):
            array[i] = forward_likelihood[k, i] * backwards_likelihood[k, i]
        decoded[k] = np.argmax(array)

    return decoded

@jit(nopython=True)
def by_viterbi(data, matrixA, arrayMeanCounts, arrayPi):
    print("Viterbi")
    n = len(arrayMeanCounts)
    delta, psi, decoded = get_empty_arrays(len(data), n)

    ####  Find delta and psi  #######
    #  Initialization   #
    delta[0, :] = get_first_row_delta(data[0], arrayMeanCounts, arrayPi)

    #  Loop   #
    for k in range(len(data) - 1):
        k = k + 1

        fact = math.gamma(data[k] + 1)
        arrayEP = np.zeros(n)
        for p in range(n):
            arrayEP[p] = (arrayMeanCounts[p])**data[k] * np.exp(-arrayMeanCounts[p])/fact

        temp_matrix = np.zeros((n, n))
        for j in range(n):
            for i in range(n):
                if matrixA[j, i] != 0:
                    temp_matrix[j, i] = delta[k - 1, i] + np.log(matrixA[j, i]) + np.log(arrayEP[j])
                else:
                    temp_matrix[j, i] = np.nan

        for j in range(n):
            delta[k, j] = np.nanmax(temp_matrix[j, :])
            psi[k, j] = get_nanargmax_usingnumba(temp_matrix[j, :])

    ####  Decode    ######
    #  Initilization  #
    decoded[-1] = get_nanargmax_usingnumba(delta[-1, :])

    #  Loop  #
    for m in range(len(data) - 1):
        k = len(data) - 2 - m
        decoded[k] = psi[k + 1, decoded[k + 1]]

    return decoded

@jit(nopython=True)
def get_empty_arrays(len_data, n):
    delta = np.zeros((len_data, n))
    psi = np.zeros((len_data, n), dtype=np.int64)
    decoded = np.zeros(len_data, dtype=np.int64)

    return delta, psi, decoded

@jit(nopython=True)
def get_first_row_delta(data, arrayMeanCounts, arrayPi):
    first_row_delta = np.zeros(len(arrayMeanCounts))
    for i in range(len(arrayMeanCounts)):
        b = ((arrayMeanCounts[i]) ** data) * math.exp(-(arrayMeanCounts[i])) / math.gamma(data)
        first_row_delta[i] = np.log(arrayPi[i]) + np.log(b)

    return first_row_delta

@jit(nopython=True)
def get_nanargmax_usingnumba(data):
    # Unlike np.nanargmax this algorithm randomly chooses between indices when they have the same max value
    # np.nanargmax in contrast always picks the smallest index
    num_iters = len(data)
    max_index = np.random.randint(0, num_iters - 1)
    max_value = np.NINF
    for i in range(num_iters):
        if data[i] != np.nan and data[i] > max_value:
            max_index = i
            max_value = data[i]
        elif data[i] != np.nan and data[i] == max_value:
            flip = np.random.randint(0, 1)
            if flip == 0:
                max_index = i

    return max_index

def get_nanargmax_withoutnumba(data):
    # Not sure you should use this even if you can, if an array has two or more values that are equal to one another
    # np.nanargmax returns the index of the first instance of that value. One would think that that adds a bias to
    # what you are doing and that instead you would want to randomly choose between the possible indices
    num_iters = len(data)
    if np.count_nonzero(~np.isnan(data)) == 0:
        max_index = np.random.randint(0, num_iters - 1)
    else:
        max_index = np.nanargmax(data)

    return max_index