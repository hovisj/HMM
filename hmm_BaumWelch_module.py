import numpy as np
import hmm_decoding_module
import hmm_forward_backwards_module
import hmm_maximization_module
import hmm_changepoints_module
import hmm_analysis_module

def for_plotting(time, data, arrayPi, matrixA, arrayMeanCounts, max_iterations):
    print("Starting Baum-Welch")

    # Initialize Arrays
    changepoints_smoothers_list = []
    changepoints_viterbi_list = []
    arrayPi_list = []
    matrixA_list = []
    arrayMeanCounts_list = []

    #### HMM: EM / Baum-Welch Algorithm  #######
    # Initialization
    num_iterations = 0
    old_loglikelihood = np.NINF

    loop = True
    while loop == True:
        # Estimation Step
        forward_likelihood, backwards_likelihood, loglikelihood = hmm_forward_backwards_module.run_algorithms_scaled_for_numba(len(arrayPi), len(data), data, arrayPi, matrixA, arrayMeanCounts)

        # Decided Whether to Continue Maximizing or Stop
        num_iterations = num_iterations + 1
        if (num_iterations < max_iterations and loglikelihood > old_loglikelihood):
            old_loglikelihood = loglikelihood

            # Decode for Plotting
            # Smoothers
            decoded = hmm_decoding_module.by_smoothers(len(arrayMeanCounts), forward_likelihood, backwards_likelihood)
            changepoints_smoothers = hmm_changepoints_module.get_changepoint_time_index_direction(decoded, time)
            # Viterbi
            decoded = hmm_decoding_module.by_viterbi(data, matrixA, arrayMeanCounts, arrayPi)
            changepoints_viterbi = hmm_changepoints_module.get_changepoint_time_index_direction(decoded, time)

            # Save Info for Each Iteration
            changepoints_smoothers_list.append(changepoints_smoothers)
            changepoints_viterbi_list.append(changepoints_viterbi)
            arrayPi_list, arrayMeanCounts_list, matrixA_list = save_arrays_each_iter(arrayPi, arrayMeanCounts, matrixA, arrayPi_list, arrayMeanCounts_list, matrixA_list)

            # Maximization
            arrayPi, matrixA, arrayMeanCounts = hmm_maximization_module.maximization(forward_likelihood, backwards_likelihood, data, matrixA, arrayMeanCounts)
        else:
            loop = False

    if num_iterations == max_iterations:
        decoded = hmm_decoding_module.by_smoothers(len(arrayMeanCounts), forward_likelihood, backwards_likelihood)
        changepoints_smoothers = hmm_changepoints_module.get_changepoint_time_index_direction(decoded, time)
        decoded = hmm_decoding_module.by_viterbi(data, matrixA, arrayMeanCounts, arrayPi)
        changepoints_viterbi = hmm_changepoints_module.get_changepoint_time_index_direction(decoded, time)

        changepoints_smoothers_list.append(changepoints_smoothers)
        changepoints_viterbi_list.append(changepoints_viterbi)
        arrayPi_list, arrayMeanCounts_list, matrixA_list = save_arrays_each_iter(arrayPi, arrayMeanCounts, matrixA, arrayPi_list, arrayMeanCounts_list, matrixA_list)

    changepoints_smoothers = np.asarray(changepoints_smoothers_list)
    changepoints_viterbi = np.asarray(changepoints_viterbi_list)
    arrayPi = np.asarray(arrayPi_list)
    matrixA = np.asarray(matrixA_list)
    arrayMeanCounts = np.asarray(arrayMeanCounts_list)

    return arrayPi, matrixA, arrayMeanCounts, changepoints_smoothers, changepoints_viterbi, num_iterations


def with_decoding(smoothers, viterbi, time, data, max_num_bins_bead, max_iterations, concentration, bin_width, num_photons_subset, arrayPi, arrayMeanCounts, matrixA, row, final_info_array, array_of_composite_bead_arrays, n, max_n, num_for_bead_geometric):
    print("Starting Baum-Welch")

    # Initialize Arrays
    arrayPi_list = []
    matrixA_list = []
    arrayMeanCounts_list = []

    #############################
    ####  Main Algorithm  #######
    #############################

    # Initialization
    num_iterations = 0
    old_loglikelihood = np.NINF

    loop = True
    while loop == True:
        # Estimation Step
        forward_likelihood, backwards_likelihood, loglikelihood = hmm_forward_backwards_module.run_algorithms_scaled_for_numba(len(arrayPi), len(data), data, arrayPi, matrixA, arrayMeanCounts)

        # Decided Whether to Maximize or Stop
        num_iterations = num_iterations + 1
        if (num_iterations < max_iterations and loglikelihood > old_loglikelihood):
            old_loglikelihood = loglikelihood

            # Save arrayPi, arrayMeanCounts, matrixA for Each Iteration
            arrayPi_list, arrayMeanCounts_list, matrixA_list = save_arrays_each_iter(arrayPi, arrayMeanCounts, matrixA, arrayPi_list, arrayMeanCounts_list, matrixA_list)

            # Decode, Find ChangePoints, Get Composite Bead, Calculate Statistics, Add to Arrays
            row, final_info_array, array_of_composite_bead_arrays = hmm_analysis_module.decode_changepoints_compositebead_statistics_savingarrays(smoothers, viterbi, n, forward_likelihood,
                                                                                                                                                  backwards_likelihood, time, data,
                                                                                                                                                  max_num_bins_bead, arrayMeanCounts[0], row,
                                                                                                                                                  final_info_array, array_of_composite_bead_arrays,
                                                                                                                                                  bin_width, concentration, num_photons_subset,
                                                                                                                                                  loglikelihood, matrixA, arrayMeanCounts, arrayPi,
                                                                                                                                                  num_iterations, max_iterations,
                                                                                                                                                  num_for_bead_geometric)

            # Maximization
            arrayPi, matrixA, arrayMeanCounts = hmm_maximization_module.maximization(forward_likelihood, backwards_likelihood, data, matrixA, arrayMeanCounts)
        else:
            loop = False

    if num_iterations == max_iterations:
        # Save arrayPi, arrayMeanCounts, matrixA for Last Iteration
        arrayPi_list, arrayMeanCounts_list, matrixA_list = save_arrays_each_iter(arrayPi, arrayMeanCounts, matrixA, arrayPi_list, arrayMeanCounts_list, matrixA_list)

        # Decode, Find ChangePoints, Get Composite Bead, Calculate Statistics, Add to Arrays
        row, final_info_array, array_of_composite_bead_arrays = hmm_analysis_module.decode_changepoints_compositebead_statistics_savingarrays(
            smoothers, viterbi, n, forward_likelihood, backwards_likelihood, time, data, max_num_bins_bead, arrayMeanCounts[0], row, final_info_array, array_of_composite_bead_arrays,
            bin_width, concentration, num_photons_subset, loglikelihood, matrixA, arrayMeanCounts, arrayPi, num_iterations, max_iterations, num_for_bead_geometric)

    arrayPi_alliters = np.asarray(arrayPi_list)
    arrayMeanCounts_alliters = np.asarray(arrayMeanCounts_list)
    matrixA_alliters = np.asarray(matrixA_list)

    return final_info_array, array_of_composite_bead_arrays, row, arrayPi_alliters, arrayMeanCounts_alliters, matrixA_alliters


def save_arrays_each_iter(arrayPi, arrayMeanCounts, matrixA, arrayPi_list, arrayMeanCounts_list, matrixA_list):
    arrayPi_list.append(arrayPi)
    arrayMeanCounts_list.append(arrayMeanCounts)
    matrixA_list.append(matrixA)

    return arrayPi_list, arrayMeanCounts_list, matrixA_list