import hmm_decoding_module
import hmm_changepoints_module
import hmm_outputs_module
import hmm_simulation_module
import hmm_forward_backwards_module
import hmm_BaumWelch_module
import numpy as np

def baum_welch_plus_analysis(smoothers, viterbi, time, data, max_num_bins_composite_bead, max_iterations, concentration, bin_width,
        fraction_of_photons_for_subset, arrayPi, arrayMeanCounts, matrixA, row, final_info_array,
        array_of_composite_bead_arrays, n, max_n, num_for_bead_geometric, max_num_for_bead_geometric, arrayPi_arraytosave, arrayMeanCounts_arraytosave, matrixA_arraytosave):
    print("row0 ", row)
    final_info_array, array_of_composite_bead_arrays, row, arrayPi_alliters, arrayMeanCounts_alliters, matrixA_alliters, loglikelihood_alliters = hmm_BaumWelch_module.with_decoding(
        smoothers, viterbi, time, data, max_num_bins_composite_bead, max_iterations, concentration, bin_width,
        fraction_of_photons_for_subset, arrayPi, arrayMeanCounts, matrixA, row, final_info_array,
        array_of_composite_bead_arrays, n, max_n, num_for_bead_geometric)
    matrixA_arraytosave, arrayMeanCounts_arraytosave, arrayPi_arraytosave = hmm_outputs_module.add_hmm_inputs_to_array(
        row, arrayPi_arraytosave, arrayMeanCounts_arraytosave, matrixA_arraytosave, arrayPi_alliters,
        arrayMeanCounts_alliters, matrixA_alliters, num_for_bead_geometric, max_num_for_bead_geometric)
    print("row1 ", row)

    return arrayPi_alliters, arrayMeanCounts_alliters, matrixA_alliters, loglikelihood_alliters, final_info_array, array_of_composite_bead_arrays, row, arrayPi_arraytosave, arrayMeanCounts_arraytosave, matrixA_arraytosave


def decode_changepoints_compositebead_statistics_savingarrays(smoothers, viterbi, n, forward_likelihood, backwards_likelihood, time, data, max_num_bins_bead, background_mean_counts, row, final_info_array, array_of_composite_bead_arrays, bin_width, concentration, fraction_of_photons_for_subset, loglikelihood, matrixA, arrayMeanCounts, arrayPi, num_iterations, max_iterations, num_for_bead_geometric):
    print("Decode, Changepoints, Composite Bead, Statistics, Add to Arrays")
    if smoothers == 1:
        decoded = hmm_decoding_module.by_smoothers(n, forward_likelihood, backwards_likelihood)
        row, final_info_array, array_of_composite_bead_arrays = changepoints_compositebead_statistics_savingarrays(1, 0, decoded, time, data, max_num_bins_bead,
                                                           background_mean_counts, row, final_info_array,
                                                           array_of_composite_bead_arrays, bin_width, concentration,
                                                           fraction_of_photons_for_subset, loglikelihood, matrixA, arrayMeanCounts,
                                                           num_iterations, max_iterations, num_for_bead_geometric)

    if viterbi == 1:
        decoded = hmm_decoding_module.by_viterbi(data, matrixA, arrayMeanCounts, arrayPi)
        row, final_info_array, array_of_composite_bead_arrays = changepoints_compositebead_statistics_savingarrays(0, 1, decoded, time, data, max_num_bins_bead,
                                                           background_mean_counts, row, final_info_array,
                                                           array_of_composite_bead_arrays, bin_width, concentration,
                                                           fraction_of_photons_for_subset, loglikelihood, matrixA, arrayMeanCounts,
                                                           num_iterations, max_iterations, num_for_bead_geometric)

    return row, final_info_array, array_of_composite_bead_arrays

def changepoints_compositebead_statistics_savingarrays(smoothers, viterbi, decoded, time, data, max_num_bins_bead, background_mean_counts, row, final_info_array, array_of_composite_bead_arrays, bin_width, concentration, num_photons_subset, loglikelihood, matrixA, arrayMeanCounts, num_iterations, max_iterations, num_for_bead_geometric):
    changepoint_index = hmm_changepoints_module.get_changepoint_time_index_direction(decoded, time)
    all_beads_plus_dwell, composite_bead_array, num_beads_array = hmm_outputs_module.composite_bead(data,
                                                                                                    max_num_bins_bead,
                                                                                                    background_mean_counts,
                                                                                                    changepoint_index,
                                                                                                    num_iterations,
                                                                                                    num_for_bead_geometric)
    stats_all_transitions = hmm_outputs_module.statistics_all_transitions(changepoint_index, bin_width)
    row, final_info_array, array_of_composite_bead_arrays = hmm_outputs_module.add_to_holding_arrays(smoothers, viterbi, row,
                                                                                                     final_info_array,
                                                                                                     array_of_composite_bead_arrays,
                                                                                                     stats_all_transitions,
                                                                                                     bin_width,
                                                                                                     concentration,
                                                                                                     num_photons_subset,
                                                                                                     loglikelihood,
                                                                                                     matrixA,
                                                                                                     arrayMeanCounts,
                                                                                                     num_beads_array,
                                                                                                     num_iterations,
                                                                                                     max_iterations,
                                                                                                     num_for_bead_geometric,
                                                                                                     composite_bead_array)

    return row, final_info_array, array_of_composite_bead_arrays

def find_beads_detected_in_background_simulation(num_data_points, bin_width, row_b, arrayPi_alliters, arrayMeanCounts_alliters, matrixA_alliters, smoothers, viterbi, n, max_num_bins_bead, final_info_array, array_of_composite_bead_arrays, num_for_bead_geometric, concentration, fraction_of_photons_for_subset, max_iterations):
    print("Find Beads Detected in Background Simulation")
    total_iterations = len(arrayMeanCounts_alliters)
    for i in range(total_iterations):
        arrayPi = arrayPi_alliters[i]
        arrayMeanCounts = arrayMeanCounts_alliters[i]
        matrixA = matrixA_alliters[i]

        num_iterations = i + 1

        time, background = hmm_simulation_module.simulate_background(arrayMeanCounts[0], num_data_points, bin_width)

        forward_likelihood, backwards_likelihood, loglikelihood = hmm_forward_backwards_module.run_algorithms_scaled_for_numba(len(arrayMeanCounts), len(background), background, arrayPi, matrixA, arrayMeanCounts)

        row_b, final_info_array, array_of_composite_bead_arrays = decode_changepoints_compositebead_statistics_savingarrays(smoothers, viterbi, n, forward_likelihood,
                                                                  backwards_likelihood, time, background, max_num_bins_bead,
                                                                  arrayMeanCounts[0], row_b, final_info_array,
                                                                  array_of_composite_bead_arrays, bin_width,
                                                                  concentration, fraction_of_photons_for_subset, loglikelihood,
                                                                  matrixA, arrayMeanCounts, arrayPi,
                                                                  num_iterations, max_iterations,
                                                                  num_for_bead_geometric)

    return final_info_array, array_of_composite_bead_arrays, background, row_b

def get_composite_bead_from_simulation(row, array_of_composite_bead_arrays_sim, array_of_num_beads_array, matrixA, arrayMeanCounts, time, num_for_bead_geometric, max_num_bins_bead):
    print("Composite Bead from Simulation")
    total_iterations = len(arrayMeanCounts)
    for i in range(total_iterations):
        num_iterations = i + 1
        row1 = int(row/2) - total_iterations + i
        state, data = hmm_simulation_module.simulation(matrixA[i], arrayMeanCounts[i], len(time))
        changepoint_index = hmm_changepoints_module.get_changepoint_time_index_direction(state, time)
        all_beads_plus_dwell, composite_bead_array, num_beads_array = hmm_outputs_module.composite_bead(data, max_num_bins_bead, arrayMeanCounts[i, 0], changepoint_index, num_iterations, num_for_bead_geometric)
        array_of_composite_bead_arrays_sim = hmm_outputs_module.add_composite_bead_to_array(0, 0, row1, array_of_composite_bead_arrays_sim, composite_bead_array)
        num_beads_array = np.append(num_beads_array, np.array([num_iterations, num_for_bead_geometric]))
        array_of_num_beads_array[row1, :] = num_beads_array

    return array_of_num_beads_array, array_of_composite_bead_arrays_sim, data
