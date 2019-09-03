

import hmm_BaumWelch_module
import numpy as np
import hmm_get_data_module
import hmm_plotting_module
import cusum_analysis_core_module
import hmm_arrayPi_arrayMeanCounts_matrixA_module

def variables():
    ####       HMM  & CUSUM      #################
    bin_width = 2e-5   # scalar
    lambda_0_analysis = 0.4e5  # scalar
    lambda_1_analysis = 2e5  # scalar

    ####       HMM        #################
    num_for_bead_geometric = np.array([5])   # array
    stay_background = (1 - 1e-7)   # scalar
    stay_bead = 0.8    # scalar
    max_iterations = 5   # scalar

    ####       CUSUM        #################
    ####  Type of Analysis  ##################
    oneway_or_twoway_cusum = 2  # 1 for one-way, 2 for two-way
    exponential_or_poisson = 1  # 0 for exponential, 1 for Poisson

    ####  Variables  ##################
    delay = 0     # scalar
    h_increasing = 12    # scalar
    h_decreasing = 5     # scalar

    ####  File Open Information  ########
    ####  Data (or Simulation)   ########
    subset_data = 1   # 0 no, 1 yes
    num_photons_subset = 500000    #Number of photons in subset - subset will be picked randomly from data
    file_open_location = '/Users/jenniferhovis/documents/fluxus_josh/data/LargeParticle3June24/'
    open_filename = '100fM'
    open_filename_extension = '.npy'  # Choices are '.npy' and '.bin'
    data_convert_to_seconds = 1e-12

    ####  Peak Locations   #######
    include_peaks = 1   # 0 for no 1 for yes
    file_open_location_peaks = '/Users/jenniferhovis/documents/fluxus_josh/data/LargeParticle3June24/'
    open_filename_peaks = '100fM_Peak_Location.csv'  #assumes file saved as csv with a single column of time stamps in units of seconds


    ####End Inputs ##################
    ##############################
    #####  Behind the scences ####
    hmm_parameters = [num_for_bead_geometric, stay_background, stay_bead, max_iterations]
    bin_width = np.array([bin_width])
    lambda0_lambda1_hi_hd_delay = np.array([lambda_0_analysis, lambda_1_analysis, h_increasing, h_decreasing, delay])
    peak_locations = []
    if include_peaks == 1:
        peak_locations = np.loadtxt(file_open_location_peaks + open_filename_peaks, delimiter=",")
        peak_locations = np.array(peak_locations)
    peaks = [include_peaks, peak_locations]

    return peaks, subset_data, num_photons_subset, bin_width, lambda0_lambda1_hi_hd_delay, oneway_or_twoway_cusum, exponential_or_poisson, file_open_location, open_filename, open_filename_extension, data_convert_to_seconds, hmm_parameters


def main():
    # Variables
    peaks, subset_data, num_photons_subset, bin_width, lambda0_lambda1_hi_hd_delay, oneway_or_twoway_cusum, exponential_or_poisson, file_open_location, open_filename, open_filename_extension, data_convert_to_seconds, hmm_paramters = variables()
    num_for_bead_geometric = hmm_paramters[0]
    stay_background = hmm_paramters[1]
    stay_bead = hmm_paramters[2]
    max_iterations = hmm_paramters[3]

    # Get Time and Data
    time, data = hmm_get_data_module.main(file_open_location, open_filename, open_filename_extension, data_convert_to_seconds, bin_width[0], subset_data, num_photons_subset)

    # CUSUM
    time_data = np.stack((time, data), axis=-1)
    trip_changpoint_numtrips_direction_nummerge_2way = cusum_analysis_core_module.run_analysis_core(oneway_or_twoway_cusum, exponential_or_poisson, time_data, bin_width[0], lambda0_lambda1_hi_hd_delay)
    print("CUSUM Finished")

    for j in range(len(num_for_bead_geometric)):
        # Get arrayPi, arrayMeanCounts and matrixA
        arrayPi, arrayMeanCounts, matrixA, max_n, n = hmm_arrayPi_arrayMeanCounts_matrixA_module.get_hmm_parameters(stay_background, stay_bead, lambda0_lambda1_hi_hd_delay[0], lambda0_lambda1_hi_hd_delay[1], bin_width, num_for_bead_geometric[j], np.max(num_for_bead_geometric))

        # HMM
        arrayPi, matrixA, arrayMeanCounts, changepoints_smoothers, changepoints_viterbi, num_iterations = hmm_BaumWelch_module.for_plotting(time, data, arrayPi, matrixA, arrayMeanCounts, max_iterations)
        print("HMM Finished")

        # Plot CUSUM and For Each Iteration Smoothers & Viterbi
        for k in range(num_iterations):
            if peaks[0] == 0:
                hmm_plotting_module.plot_HMM_CUSUM(changepoints_smoothers[k], changepoints_viterbi[k], trip_changpoint_numtrips_direction_nummerge_2way, time_data, bin_width[0])
            else:
                hmm_plotting_module.plot_HMM_CUSUM_Peaks(peaks[1], changepoints_smoothers[k], changepoints_viterbi[k], trip_changpoint_numtrips_direction_nummerge_2way, time_data, bin_width[0])

            print("Sum ArrayPi: ", np.sum(arrayPi[k]), "  ArrayPi ", arrayPi[k])
            print("Array Mean Counts ", arrayMeanCounts[k])
            temp = matrixA[k]
            for q in range(len(temp)):
                print("Matrix A, Row=", q, temp[q, :], "Sum=", np.sum(temp[q, :]))

main()
