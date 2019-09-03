
import datetime
import numpy as np
import pandas as pd
import hmm_changepoints_module
from numba import jit

##################################################
#####          Statistics                #########
##################################################

#####       Every Transition - HMM           ##########
#@jit(nopython=True)
def statistics_every_transition_HMM(exponential_or_poisson, time_index_direction, time_data, bin_width):
    time = time_data[:, 0]
    data = time_data[:, 1]

    changepoint_index = time_index_direction[:, 1].astype(np.int64)

    changepoint_time = np.zeros(len(changepoint_index), dtype=np.float64)
    mean = np.zeros(len(changepoint_index), dtype=np.float64)
    stdev = np.zeros(len(changepoint_index), dtype=np.float64)
    var = np.zeros(len(changepoint_index), dtype=np.float64)
    number_in_state = np.zeros(len(changepoint_index), dtype=np.float64)
    duration_in_state = np.zeros(len(changepoint_index), dtype=np.float64)
    duration_in_both_states = np.zeros(len(changepoint_index), dtype=np.float64)

    # All transitions except the last two
    for i in range(len(changepoint_index) - 2):
        changepoint_time[i] = time[changepoint_index[i]]
        number_in_state[i] = changepoint_index[i + 1] - changepoint_index[i]
        duration_in_state[i] = time[changepoint_index[i + 1]] - time[changepoint_index[i]]
        duration_in_both_states[i] = time[changepoint_index[i + 2]] - time[changepoint_index[i]]

        if changepoint_index[i + 1] >= changepoint_index[i] + 1:
            mean[i] = np.mean(data[changepoint_index[i]:changepoint_index[i + 1]])
            stdev[i] = np.std(data[changepoint_index[i]:changepoint_index[i + 1]], ddof=1)
            var[i] = np.var(data[changepoint_index[i]:changepoint_index[i + 1]], ddof=1)
        elif changepoint_index[i + 1] == changepoint_index[i]:
            mean[i] = np.mean(data[changepoint_index[i]])
            stdev[i] = np.std(data[changepoint_index[i]], ddof=1)
            var[i] = np.var(data[changepoint_index[i]], ddof=1)

    # Second to last transition
    changepoint_time[-2] = time[changepoint_index[-2]]
    number_in_state[-2] = changepoint_index[-1] - changepoint_index[-2]
    duration_in_state[-2] = time[changepoint_index[-1]] - time[changepoint_index[-2]]
    duration_in_both_states[-2] = np.nan

    if changepoint_index[-1] >= changepoint_index[-2] + 1:
        mean[-2] = np.mean(data[changepoint_index[-2]:changepoint_index[-1]])
        stdev[-2] = np.std(data[changepoint_index[-2]:changepoint_index[-1]], ddof=1)
        var[-2] = np.var(data[changepoint_index[-2]:changepoint_index[-1]], ddof=1)
    elif changepoint_index[-1] == changepoint_index[-2]:
        mean[-2] = np.mean(data[changepoint_index[-2]])
        stdev[-2] = np.std(data[changepoint_index[-2]], ddof=1)
        var[-2] = np.var(data[changepoint_index[-2]], ddof=1)

    # Last transition
    changepoint_time[-1] = time[changepoint_index[-1]]
    number_in_state[-1] = np.nan
    duration_in_state[-1] = np.nan
    duration_in_both_states[-1] = np.nan
    mean[-1] = np.nan
    stdev[-1] = np.nan
    var[-1] = np.nan

    if exponential_or_poisson == 0:
        rate = 1 / mean
    else:
        rate = mean / bin_width

    return changepoint_time, mean, stdev, var, number_in_state, duration_in_state, duration_in_both_states, rate

#####        All Transitions             ##########
def statistics_all_transitions(changepoint_index, bin_width):
    # Calculate Statistics
    dwells = np.subtract(changepoint_index[1::2], changepoint_index[::2])
    dwells.astype(np.float64)
    dwells = dwells * bin_width
    dwell_mean = np.mean(dwells)
    dwell_median = np.median(dwells)
    dwell_var = np.var(dwells, ddof=1)

    bead_arrival_index = changepoint_index[::2]
    bead_arrival_index.astype(np.float64)
    bead_interarrival_times = np.ediff1d(bead_arrival_index) * bin_width
    bead_interarrival_times_mean = np.mean(bead_interarrival_times)
    bead_interarrival_index_coeff_var = np.std(bead_interarrival_times, ddof=1) / bead_interarrival_times_mean
    bead_arrival_rate = 1 / bead_interarrival_times_mean

    number_of_beads = len(changepoint_index[::2])

    return dwell_mean, dwell_median, dwell_var, bead_arrival_rate, bead_interarrival_index_coeff_var, number_of_beads


##################################################
#####        Composite Bead              #########
##################################################
def composite_bead(data, max_num_bins_bead, background_mean_counts, changepoint_index, num_iteration, num_for_bead_geometric):
    print("Composite Bead")
    if changepoint_index.size > 0:
        all_beads_plus_dwell = get_bead_traces_padwithrandom(data, max_num_bins_bead, background_mean_counts, changepoint_index)
        composite_bead_array, num_beads_array = find_mean(all_beads_plus_dwell, max_num_bins_bead)
        composite_bead_array = add_iter_geo(composite_bead_array, num_iteration, num_for_bead_geometric)
    else:
        all_beads_plus_dwell = np.zeros(max_num_bins_bead + 1)
        composite_bead_array = np.zeros((max_num_bins_bead, 7))
        composite_bead_array[:] = np.nan
        composite_bead_array[:, 0] = np.arange(1, (max_num_bins_bead + 1))
        composite_bead_array = add_iter_geo(composite_bead_array, num_iteration, num_for_bead_geometric)
        num_beads_array = np.zeros(8)
    all_beads_plus_dwell = None
    all_beads_plus_dwell = []
    return all_beads_plus_dwell, composite_bead_array, num_beads_array

def get_bead_traces_padwithrandom(data, max_num_bins_bead, background_mean_counts, changepoint_index):
    # This module grabs the bead, centers the bead in the array and pads the rest out with
    # numbers randomly generated from the poisson distribution with lambda = background_mean_counts

    bead_arrival_index = changepoint_index[::2]
    bead_depature_index = changepoint_index[1::2]

    num_beads = len(bead_arrival_index)
    all_beads = np.random.poisson(background_mean_counts, size=(len(bead_arrival_index), max_num_bins_bead))
    bead_dwell_array = np.subtract(bead_depature_index, bead_arrival_index)
    data.astype(np.float16)

    for i in range(num_beads):
        num_bins = bead_dwell_array[i]
        if num_bins <= max_num_bins_bead:
            bead = data[bead_arrival_index[i]:bead_depature_index[i]]
            if (num_bins/2) != np.floor(num_bins/2):
                rand = np.random.random_sample()
                if rand < 0.5:
                    start = int(np.floor((max_num_bins_bead - num_bins)/2))
                    all_beads[i, start:(start + num_bins)] = bead
                else:
                    start = int(np.ceil((max_num_bins_bead - num_bins)/2))
                    all_beads[i, start:(start + num_bins)] = bead
            else:
                start = int((max_num_bins_bead - num_bins)/2)
                all_beads[i, start:(start + num_bins)] = bead

    all_beads_plus_dwell = np.column_stack((all_beads, bead_dwell_array))

    return all_beads_plus_dwell

def find_mean(all_beads_plus_dwell, max_num_bins_bead):
    num_beads_array = np.zeros(8)
    num_beads_array[0] = len(all_beads_plus_dwell)

    # Filter out rows with length < 5 bins
    all_beads_plus_dwell = all_beads_plus_dwell[all_beads_plus_dwell[:, -1] >= 5]
    num_beads_array[1] = num_beads_array[0] - len(all_beads_plus_dwell)

    #Filter out blank rows (those beads that exceeded max_num_bins_bead)
    all_beads_plus_dwell = all_beads_plus_dwell[all_beads_plus_dwell[:, -1] <= max_num_bins_bead]
    num_beads_array[2] = (num_beads_array[0] - num_beads_array[1]) - len(all_beads_plus_dwell)

    # Find Composite Bead, All lengths from 1 to max_num_bins_bead
    composite_bead_array = np.zeros((max_num_bins_bead, 7))
    composite_bead_array[:, 0] = np.arange(1, (max_num_bins_bead + 1))
    composite_bead_array[:, 1] = np.mean(all_beads_plus_dwell[:, 0:-1], axis = 0)
    for i in range(5):
        j = i + 2
        temp = all_beads_plus_dwell[(all_beads_plus_dwell[:, -1] > (i * 10)) & (all_beads_plus_dwell[:, -1] <= (i * 10 + 10))]
        num_beads_array[i + 3] = len(temp)
        composite_bead_array[:, j] = np.mean(temp[:, 0:-1], axis = 0)

    if np.sum(num_beads_array[1::]) != num_beads_array[0]:
        print("mistake in outputs.composite_bead: ", print(num_beads_array))

    return composite_bead_array, num_beads_array

def add_iter_geo(composite_bead_array, num_iteration, num_for_bead_geometric):
    num_iters_array = num_iteration * np.ones(len(composite_bead_array))
    composite_bead_array = np.column_stack((composite_bead_array, num_iters_array))

    num_geo_array = num_for_bead_geometric * np.ones(len(composite_bead_array))
    composite_bead_array = np.column_stack((composite_bead_array, num_geo_array))

    return composite_bead_array

# Alt Version, Not Used
def get_bead_traces(data, max_num_bins_bead, changepoint_index):
    #This module grabs the bead and enough of the data trace on either side to equal max_num_bins_bead
    #If beads are closely spaced it could be mis-leading
    bead_arrival_index = changepoint_index[::2]
    bead_depature_index = changepoint_index[1::2]

    num_beads = len(bead_arrival_index)

    all_beads = -1 * np.ones((num_beads, max_num_bins_bead), dtype=np.float16)
    bead_dwell_array = np.subtract(bead_depature_index, bead_arrival_index)
    data.astype(np.float16)

    for i in range(num_beads):
        num_bins = bead_dwell_array[i]
        if num_bins <= max_num_bins_bead:
            if (num_bins/2) != np.floor(num_bins/2):
                rand = np.random.random_sample()
                if rand < 0.5:
                    start = bead_arrival_index[i] - int(np.floor((max_num_bins_bead - num_bins)/2))
                else:
                    start = bead_arrival_index[i] - int(np.ceil((max_num_bins_bead - num_bins)/2))
            else:
                start = bead_arrival_index[i] - int((max_num_bins_bead - num_bins)/2)
            all_beads[i, :] = data[start:(start + max_num_bins_bead)]

    all_beads_plus_dwell = np.column_stack((all_beads, bead_dwell_array))

    if np.min(all_beads) == -1:
        print("mistake in outputs.get_bead_traces")

    return all_beads_plus_dwell
##################################################
####      Create Holding Arrays   ################
##################################################

def empty_array(num_col_final_info, max_iterations, length_num_for_bead_geometric, max_num_for_bead_geometric, max_num_bins_bead):
    final_info_array = empty_final_info_array(num_col_final_info, max_iterations, length_num_for_bead_geometric)
    arrayPi_arraytosave, arrayMeanCounts_arraytosave, matrixA_arraytosave = empty_hmm_inputs_arrays(max_iterations, length_num_for_bead_geometric, max_num_for_bead_geometric)
    array_of_composite_bead_arrays = empty_composite_bead_array(max_num_bins_bead, max_iterations, length_num_for_bead_geometric)

    final_info_array_background = np.copy(final_info_array)
    array_of_composite_bead_arrays_background = np.copy(array_of_composite_bead_arrays)

    array_of_composite_bead_arrays_sim = np.copy(array_of_composite_bead_arrays)
    array_of_num_beads_array = np.zeros((max_iterations * length_num_for_bead_geometric, 10))

    row = 0
    row_b = 0

    return final_info_array, arrayPi_arraytosave, arrayMeanCounts_arraytosave, matrixA_arraytosave, array_of_composite_bead_arrays, final_info_array_background, array_of_composite_bead_arrays_background, array_of_composite_bead_arrays_sim, array_of_num_beads_array, row, row_b

def empty_final_info_array(num_columns, max_iterations, length_num_for_bead_geometric):
    num_rows = max_iterations * length_num_for_bead_geometric * 2
    empty_final_info_array = np.zeros((num_rows, num_columns))

    return empty_final_info_array

def empty_hmm_inputs_arrays(max_iterations, length_num_for_bead_geometric, max_num_for_bead_geometric):
    num_rows = (1 + max_num_for_bead_geometric) * max_iterations * length_num_for_bead_geometric
    arrayPi_arraytosave = np.zeros((num_rows, 4))
    arrayMeanCounts_arraytosave = np.zeros((num_rows, 4))
    matrixA_arraytosave = np.zeros((num_rows * 2, 5))

    return arrayPi_arraytosave, arrayMeanCounts_arraytosave, matrixA_arraytosave

def empty_composite_bead_array(max_num_bins_bead, max_iterations, length_num_for_bead_geometric):
    num_rows = max_num_bins_bead * max_iterations * length_num_for_bead_geometric * 2
    empty_composite_bead_array = np.zeros((num_rows, 11))

    return empty_composite_bead_array

##################################################
####      Add to Holding Arrays   ################
##################################################

def add_to_holding_arrays(smoothers, viterbi, row, final_info_array, array_of_composite_bead_arrays, stats_all_transitions, bin_width, concentration, num_photons_subset, loglikelihood, matrixA, arrayMeanCounts, num_beads_array, num_iterations, max_iterations, num_for_bead_geometric, composite_bead_array):
    final_info_array = add_to_final_info_array_data(smoothers, viterbi, row, final_info_array, stats_all_transitions, bin_width, concentration, num_photons_subset, loglikelihood, matrixA, arrayMeanCounts[0], num_beads_array, num_iterations, max_iterations, num_for_bead_geometric)
    array_of_composite_bead_arrays = add_composite_bead_to_array(smoothers, viterbi, row, array_of_composite_bead_arrays, composite_bead_array)
    row = row + 1

    return row, final_info_array, array_of_composite_bead_arrays

##### Statistics: One Row Per Data Trace ##########
def add_to_final_info_array_data(smoothers, viterbi, row, final_info_array, stats_all_transitions, bin_width, concentration, num_photons_subset, loglikelihood, matrixA, background_counts, num_beads_array, num_iterations, max_iterations, num_for_bead_geometric):

    final_info_array[row, 0] = smoothers
    final_info_array[row, 1] = viterbi
    final_info_array[row, 2:8] = stats_all_transitions
    final_info_array[row, 8] = -np.log(matrixA[0, 0])/bin_width
    final_info_array[row, 9] = bin_width
    final_info_array[row, 10] = concentration
    final_info_array[row, 11] = num_photons_subset
    final_info_array[row, 12] = loglikelihood
    final_info_array[row, 13] = num_iterations
    final_info_array[row, 14] = max_iterations
    final_info_array[row, 15] = num_for_bead_geometric
    final_info_array[row, 16] = background_counts
    final_info_array[row, 17::] = num_beads_array

    return final_info_array

#####         HMM Inputs                ##########
def add_hmm_inputs_to_array(row, arrayPi_arraytosave, arrayMeanCounts_arraytosave, matrixA_arraytosave, arrayPi_alliters, arrayMeanCounts_alliters, matrixA_alliters, num_for_bead_geometric, max_number):
    total_iterations = len(arrayPi_alliters)
    for m in range(total_iterations):
        number_of_iterations = (m + 1)
        arrayPi = arrayPi_alliters[m]
        arrayMeanCounts = arrayMeanCounts_alliters[m]
        matrixA = matrixA_alliters[m]
        n = len(matrixA)

        # MatrixA Non-Zero Elements
        matrixA_i = np.zeros(n * 2)
        matrixA_j = np.zeros(n * 2)
        matrixA_value = np.zeros(n * 2)

        p = 0
        for i in range(n - 1):
            matrixA_value[p] = matrixA[i, i]
            matrixA_i[p] = i
            matrixA_j[p] = i
            matrixA_value[p + 1] = matrixA[i, i + 1]
            matrixA_i[p + 1] = i
            matrixA_j[p + 1] = i + 1
            p = p + 2
        matrixA_value[p] = matrixA[-1, -1]
        matrixA_i[p] = n - 1
        matrixA_j[p] = n - 1
        matrixA_value[p + 1] = matrixA[-1, 0]
        matrixA_i[p + 1] = n - 1
        matrixA_j[p + 1] = 0

        start = (int(row/2) - total_iterations + m) * ((max_number + 1) * 2)
        matrixA_arraytosave[start:(start + n * 2), 0] = matrixA_i
        matrixA_arraytosave[start:(start + n * 2), 1] = matrixA_j
        matrixA_arraytosave[start:(start + n * 2), 2] = matrixA_value
        matrixA_arraytosave[start:(start + n * 2), 3] = number_of_iterations * np.ones(n * 2)
        matrixA_arraytosave[start:(start + n * 2), 4] = num_for_bead_geometric * np.ones(n * 2)

        # arrayMean Counts
        start = (int(row/2) - total_iterations + m) * (max_number + 1)
        arrayMeanCounts_arraytosave[start:(start + n), 0] = np.arange(n)
        arrayMeanCounts_arraytosave[start:(start + n), 1] = arrayMeanCounts
        arrayMeanCounts_arraytosave[start:(start + n), 2] = number_of_iterations * np.ones(n)
        arrayMeanCounts_arraytosave[start:(start + n), 3] = num_for_bead_geometric * np.ones(n)

        # arrayPi
        start = (int(row/2) - total_iterations + m) * (max_number + 1)
        arrayPi_arraytosave[start:(start + n), 0] = np.arange(n)
        arrayPi_arraytosave[start:(start + n), 1] = arrayPi
        arrayPi_arraytosave[start:(start + n), 2] = number_of_iterations * np.ones(n)
        arrayPi_arraytosave[start:(start + n), 3] = num_for_bead_geometric * np.ones(n)

    return matrixA_arraytosave, arrayMeanCounts_arraytosave, arrayPi_arraytosave

#####     Composite Bead Array         ##########
def add_composite_bead_to_array(smoothers, viterbi, row, array_of_composite_bead_arrays, composite_bead_array):
    composite_bead_array = np.column_stack((viterbi * np.ones(len(composite_bead_array)), composite_bead_array))
    composite_bead_array = np.column_stack((smoothers * np.ones(len(composite_bead_array)), composite_bead_array))
    start = row * len(composite_bead_array)
    stop = start + len(composite_bead_array)
    array_of_composite_bead_arrays[start:stop, :] = composite_bead_array

    return array_of_composite_bead_arrays

##################################################
#####          Save                ###############
##################################################

def save_info(file_save_location, file_save_names, open_filename, final_info_array, final_info_array_background, array_of_composite_bead_arrays, array_of_composite_bead_arrays_background, arrayPi_arraytosave, arrayMeanCounts_arraytosave, matrixA_arraytosave):
    source = '_data'
    save_statistics_all_transitions_data(final_info_array, file_save_location, file_save_names, open_filename, source)
    save_composite_bead(array_of_composite_bead_arrays, file_save_location, file_save_names, open_filename, source)
    save_hmm_inputs(arrayPi_arraytosave, arrayMeanCounts_arraytosave, matrixA_arraytosave, file_save_location, file_save_names, open_filename)

    source = '_background'
    save_statistics_all_transitions_data(final_info_array_background, file_save_location, file_save_names, open_filename, source)
    save_composite_bead(array_of_composite_bead_arrays_background, file_save_location, file_save_names, open_filename, source)

def remove_empty_rows_save_info(file_save_location, file_save_names, open_filename, final_info_array, final_info_array_background, array_of_composite_bead_arrays, array_of_composite_bead_arrays_background, arrayPi_arraytosave, arrayMeanCounts_arraytosave, matrixA_arraytosave):
    final_info_array = final_info_array[final_info_array[:, 11] > 0]
    final_info_array_background = final_info_array_background[final_info_array_background[:, 11] > 0]

    array_of_composite_bead_arrays = array_of_composite_bead_arrays[array_of_composite_bead_arrays[:, 9] > 0]
    array_of_composite_bead_arrays_background = array_of_composite_bead_arrays_background[array_of_composite_bead_arrays_background[:, 9] > 0]

    arrayPi_arraytosave = arrayPi_arraytosave[arrayPi_arraytosave[:, 2] > 0]
    arrayMeanCounts_arraytosave = arrayMeanCounts_arraytosave[arrayMeanCounts_arraytosave[:, 2] > 0]
    matrixA_arraytosave = matrixA_arraytosave[matrixA_arraytosave[:, 3] > 0]

    source = '_data'
    save_statistics_all_transitions_data(final_info_array, file_save_location, file_save_names, open_filename, source)
    save_composite_bead(array_of_composite_bead_arrays, file_save_location, file_save_names, open_filename, source)
    save_hmm_inputs(arrayPi_arraytosave, arrayMeanCounts_arraytosave, matrixA_arraytosave, file_save_location, file_save_names, open_filename)

    source = '_background'
    save_statistics_all_transitions_data(final_info_array_background, file_save_location, file_save_names, open_filename, source)
    save_composite_bead(array_of_composite_bead_arrays_background, file_save_location, file_save_names, open_filename, source)

#####  Every Transition  ##########
def save_statistics_every_transition_data_HMM(min_num_for_bead, j, exponential_or_poisson, time_index_direction, changepoint_time, mean, stdev,
                                          var, number_in_state, duration_in_state, duration_in_both_states, rate,
                                          bin_width, open_filename, concentration,
                                          file_save_location, save_name_extended, smoothers_or_viterbi):

    iteration_array = j * np.ones(len(changepoint_time))
    min_num_for_bead_array = min_num_for_bead * np.ones(len(changepoint_time))
    bin_width_array = bin_width * np.ones(len(changepoint_time))
    smoothers_or_viterbi_array = smoothers_or_viterbi * np.ones(len(changepoint_time))
    exponential_or_poisson_array = exponential_or_poisson * np.ones(len(changepoint_time))
    concentration_array = concentration * np.ones(len(changepoint_time))

    df = pd.DataFrame({'ChangepointTime': changepoint_time,
                               'DurationinOneState_Time': duration_in_state,
                               'DurationinBothStates_Time': duration_in_both_states,
                               'TransitionDirection': time_index_direction[:, 2],
                               'Mean_Counts': mean,
                               'Std_Counts': stdev,
                               'Varience_Counts': var,
                               'Rate_pertimeunit': rate,
                               'CoefficientofDispersion': var / mean,
                               'DurationinOneState_ArrayIndex': number_in_state,
                               'Exponential_or_Poisson': exponential_or_poisson_array,
                               'smoothers_or_viterbi': smoothers_or_viterbi_array,
                               'BinWidth': bin_width_array,
                               'Concentration': concentration_array,
                               'min_num_bead': min_num_for_bead_array,
                               'iteration': iteration_array
                               })
    df.to_csv(file_save_location + open_filename + '_' + save_name_extended + '_' + 'iteration_' + str(j) + '_' + 'minnumbead_' + str(min_num_for_bead) + '_smoothers_or_viterbi_' +str(smoothers_or_viterbi) + '_2HMM.csv')


#####  All Transitions   ##########
def save_statistics_all_transitions_data(final_info_array, file_save_location, file_random_number_to_prevent_overwritting_files, open_filename, source):
    if source == '_data':
        df = pd.DataFrame({'Smoothers': final_info_array[:, 0],
                           'Viterbi': final_info_array[:, 1],
                           'BeadTransitTime_Mean': final_info_array[:, 2],
                           'BeadTransitTime_Median': final_info_array[:, 3],
                           'BeadTransitTime_Variance': final_info_array[:, 4],
                           'BeadArrivalRate_Decoded': final_info_array[:, 5],
                           'BeadArrivalRate_CoefficientofVariation_Decoded': final_info_array[:, 6],
                           'Number_of_Beads': final_info_array[:, 7],
                           'BeadArrivalRate_MatrixA': final_info_array[:, 8],
                           'BinWidth': final_info_array[:, 9],
                           'Concentration': final_info_array[:, 10],
                           'FractionDataSubset': final_info_array[:, 11],
                           'loglikelihood': final_info_array[:, 12],
                           'IterationNumber': final_info_array[:, 13],
                           'MaximumNumberIterations': final_info_array[:, 14],
                           'NumGeo': final_info_array[:, 15],
                           'MeanBackgroundCounts': final_info_array[:, 16],
                           'NumberBeads': final_info_array[:, 17],
                           'NumberBeads_LessThan5Bins': final_info_array[:, 18],
                           'NumberBeads_GreaterThanMaxNum': final_info_array[:, 19],
                           'NumberBeads_Over4BinsLessThanEqual10': final_info_array[:, 20],
                           'NumberBeads_Over10BinsLessThanEqual20': final_info_array[:, 21],
                           'NumberBeads_Over20BinsLessThanEqual30': final_info_array[:, 22],
                           'NumberBeads_Over30BinsLessThanEqual40': final_info_array[:, 23],
                           'NumberBeads_Over40BinsLessThanEqual50': final_info_array[:, 24],
                           })
    else:
        df = pd.DataFrame({'Smoothers': final_info_array[:, 0],
                           'Viterbi': final_info_array[:, 1],
                           'BeadTransitTime_Mean': final_info_array[:, 2],
                           'BeadTransitTime_Median': final_info_array[:, 3],
                           'BeadTransitTime_Variance': final_info_array[:, 4],
                           'BeadArrivalRate_Decoded': final_info_array[:, 5],
                           'BeadArrivalRate_CoefficientofVariation_Decoded': final_info_array[:, 6],
                           'Number_of_Beads': final_info_array[:, 7],
                           'BeadArrivalRate_MatrixA': np.nan,
                           'BinWidth': final_info_array[:, 9],
                           'Concentration': final_info_array[:, 10],
                           'FractionDataSubset': final_info_array[:, 11],
                           'loglikelihood': final_info_array[:, 12],
                           'IterationNumber': final_info_array[:, 13],
                           'MaximumNumberIterations': final_info_array[:, 14],
                           'NumGeo': final_info_array[:, 15],
                           'MeanBackgroundCounts': final_info_array[:, 16],
                           'NumberBeads': final_info_array[:, 17],
                           'NumberBeads_LessThan5Bins': final_info_array[:, 18],
                           'NumberBeads_GreaterThanMaxNum': final_info_array[:, 19],
                           'NumberBeads_Over4BinsLessThanEqual10': final_info_array[:, 20],
                           'NumberBeads_Over10BinsLessThanEqual20': final_info_array[:, 21],
                           'NumberBeads_Over20BinsLessThanEqual30': final_info_array[:, 22],
                           'NumberBeads_Over30BinsLessThanEqual40': final_info_array[:, 23],
                           'NumberBeads_Over40BinsLessThanEqual50': final_info_array[:, 24],
                           })
    df.to_csv(file_save_location + file_random_number_to_prevent_overwritting_files + '_' + open_filename + '_HMM_AllEventsStatistics' + source + '.csv')

#####  Composite Bead   ######
def save_composite_bead(array_of_composite_bead_arrays, file_save_location, file_random_number_to_prevent_overwritting_files, open_filename, source):
    df = pd.DataFrame({
        'Smoothers': array_of_composite_bead_arrays[:, 0],
        'Viterbi': array_of_composite_bead_arrays[:, 1],
        'BinNumber': array_of_composite_bead_arrays[:, 2],
        'Mean_Over4BinsLessThanEqualMax': array_of_composite_bead_arrays[:, 3],
        'Mean_Over4BinsLessThanEqual10': array_of_composite_bead_arrays[:, 4],
        'Mean_Over10BinsLessThanEqual20': array_of_composite_bead_arrays[:, 5],
        'Mean_Over20BinsLessThanEqual30': array_of_composite_bead_arrays[:, 6],
        'Mean_Over30BinsLessThanEqual40': array_of_composite_bead_arrays[:, 7],
        'Mean_Over40BinsLessThanEqual50': array_of_composite_bead_arrays[:, 8],
        'IterationNumber': array_of_composite_bead_arrays[:, 9],
        'NumGeo': array_of_composite_bead_arrays[:, 10]
    })
    df.to_csv(file_save_location + file_random_number_to_prevent_overwritting_files + '_' + open_filename + '_HMM_CompositeBead'+ source + '.csv')

#####   HMM Inputs  #########
def save_hmm_inputs(arrayPi_arraytosave, arrayMeanCounts_arraytosave, matrixA_arraytosave, file_save_location, file_random_number_to_prevent_overwritting_files, open_filename):
    df = pd.DataFrame({"State": arrayPi_arraytosave[:, 0],
                       "ArrayPi": arrayPi_arraytosave[:, 1],
                       "IterationNumber": arrayPi_arraytosave[:, 2],
                       "NumGeo": arrayPi_arraytosave[:, 3]})
    df.to_csv(file_save_location + file_random_number_to_prevent_overwritting_files + '_' + open_filename + '_HMM_arrayPi.csv')

    df = pd.DataFrame({"State": arrayMeanCounts_arraytosave[:, 0],
                       "ArrayMeanCounts": arrayMeanCounts_arraytosave[:, 1],
                       "IterationNumber": arrayMeanCounts_arraytosave[:, 2],
                       "NumGeo": arrayMeanCounts_arraytosave[:, 3]})
    df.to_csv(file_save_location + file_random_number_to_prevent_overwritting_files + '_' + open_filename + '_HMM_arrayMeanCounts.csv')

    df = pd.DataFrame({"Row": matrixA_arraytosave[:, 0],
                       "Column": matrixA_arraytosave[:, 1],
                       "MatrixA": matrixA_arraytosave[:, 2],
                       "IterationNumber": matrixA_arraytosave[:, 3],
                       "NumGeo": matrixA_arraytosave[:, 4]})
    df.to_csv(file_save_location + file_random_number_to_prevent_overwritting_files + '_' + open_filename + '_HMM_MatrixANonZeroValues.csv')

#####   Composite Bead & Number Beads from Simulation  #########
def save_sim_analysis(file_save_location, file_random_number_to_prevent_overwritting_files, open_filename, array_of_num_beads_array, array_of_composite_bead_arrays):
    df = pd.DataFrame({
        'NumberBeads': array_of_num_beads_array[:, 0],
        'NumberBeads_LessThan5Bins': array_of_num_beads_array[:, 1],
        'NumberBeads_GreaterThanMaxNum': array_of_num_beads_array[:, 2],
        'NumberBeads_Over4BinsLessThanEqual10': array_of_num_beads_array[:, 3],
        'NumberBeads_Over10BinsLessThanEqual20': array_of_num_beads_array[:, 4],
        'NumberBeads_Over20BinsLessThanEqual30': array_of_num_beads_array[:, 5],
        'NumberBeads_Over30BinsLessThanEqual40': array_of_num_beads_array[:, 6],
        'NumberBeads_Over40BinsLessThanEqual50': array_of_num_beads_array[:, 7],
        'IterationNumber': array_of_num_beads_array[:, 8],
        'NumGeo': array_of_num_beads_array[:, 9]
    })
    df.to_csv(file_save_location + file_random_number_to_prevent_overwritting_files + '_' + open_filename + '_HMM_NumBeads_Simulation.csv')

    df = pd.DataFrame({'BinNumber': array_of_composite_bead_arrays[:, 2],
        'Mean_Over4BinsLessThanEqualMax': array_of_composite_bead_arrays[:, 3],
        'Mean_Over4BinsLessThanEqual10': array_of_composite_bead_arrays[:, 4],
        'Mean_Over10BinsLessThanEqual20': array_of_composite_bead_arrays[:, 5],
        'Mean_Over20BinsLessThanEqual30': array_of_composite_bead_arrays[:, 6],
        'Mean_Over30BinsLessThanEqual40': array_of_composite_bead_arrays[:, 7],
        'Mean_Over40BinsLessThanEqual50': array_of_composite_bead_arrays[:, 8],
        'NumberIterations': array_of_composite_bead_arrays[:, 9],
        'NumGeo': array_of_composite_bead_arrays[:, 10]})
    df.to_csv(file_save_location + file_random_number_to_prevent_overwritting_files + '_' + open_filename + '_HMM_CompositeBead_Simulation.csv')

def remove_empty_rows_save_sim_analysis(file_save_location, file_random_number_to_prevent_overwritting_files, open_filename, array_of_num_beads_array, array_of_composite_bead_arrays):
    array_of_num_beads_array = array_of_num_beads_array[array_of_num_beads_array[:, 8] > 0]
    df = pd.DataFrame({
        'NumberBeads': array_of_num_beads_array[:, 0],
        'NumberBeads_LessThan5Bins': array_of_num_beads_array[:, 1],
        'NumberBeads_GreaterThanMaxNum': array_of_num_beads_array[:, 2],
        'NumberBeads_Over4BinsLessThanEqual10': array_of_num_beads_array[:, 3],
        'NumberBeads_Over10BinsLessThanEqual20': array_of_num_beads_array[:, 4],
        'NumberBeads_Over20BinsLessThanEqual30': array_of_num_beads_array[:, 5],
        'NumberBeads_Over30BinsLessThanEqual40': array_of_num_beads_array[:, 6],
        'NumberBeads_Over40BinsLessThanEqual50': array_of_num_beads_array[:, 7],
        'IterationNumber': array_of_num_beads_array[:, 8],
        'NumGeo': array_of_num_beads_array[:, 9]
    })
    df.to_csv(file_save_location + file_random_number_to_prevent_overwritting_files + '_' + open_filename + '_HMM_NumBeads_Simulation.csv')

    array_of_composite_bead_arrays = array_of_composite_bead_arrays[:, 2:]
    array_of_composite_bead_arrays = array_of_composite_bead_arrays[array_of_composite_bead_arrays[:, 7] > 0]
    df = pd.DataFrame({'BinNumber': array_of_composite_bead_arrays[:, 0],
        'Mean_Over4BinsLessThanEqualMax': array_of_composite_bead_arrays[:, 1],
        'Mean_Over4BinsLessThanEqual10': array_of_composite_bead_arrays[:, 2],
        'Mean_Over10BinsLessThanEqual20': array_of_composite_bead_arrays[:, 3],
        'Mean_Over20BinsLessThanEqual30': array_of_composite_bead_arrays[:, 4],
        'Mean_Over30BinsLessThanEqual40': array_of_composite_bead_arrays[:, 5],
        'Mean_Over40BinsLessThanEqual50': array_of_composite_bead_arrays[:, 6],
        'IterationNumber': array_of_composite_bead_arrays[:, 7],
        'NumGeo': array_of_composite_bead_arrays[:, 8]})
    df.to_csv(file_save_location + file_random_number_to_prevent_overwritting_files + '_' + open_filename + '_HMM_CompositeBead_Simulation.csv')

#####  Simulations of Background & Data  ##############
def save_simulation(file_save_location, file_random_number_to_prevent_overwritting_files, open_filename, source, data, num_for_bead_geometric, number_beads):
   np.save(file_save_location + file_random_number_to_prevent_overwritting_files + '_' + open_filename + '_simulated' + source + '_geo_' + str(num_for_bead_geometric) + '_numberbeads_' + str(int(number_beads)), data)
