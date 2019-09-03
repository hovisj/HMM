
import numpy as np
import hmm_get_data_module
import hmm_outputs_module
import hmm_arrayPi_arrayMeanCounts_matrixA_module
import hmm_analysis_module
import hmm_simulation_module

#####  Things that are hard coded ######
# For the composite bead analysis beads between 5 and 50 bins are analyzed; those beads are further broken down into 5-10, 11-20, 21-30, 31-40, 41-50
# The number of columns in the final_info_array (this contains all the statistics that can be pulled from analyzing every transition)
# The number of columns in the array_of_composite_bead_arrays (this contains the mean of the composite bead, 5-50 & 5-10, 11-20, 21-30, 31-40, 41-50
########################################

def variables():
    ####   HMM - Inputs  #################
    bin_width = 1e-5
    lambda_0_analysis = 0.5e5  # scalar
    lambda_1_analysis = 3e5  # scalar
    num_for_bead_geometric = np.array([3])   #array
    stay_background = np.array([(1 - 1e-7)])  # array, !! must have same number of values as open_filename and concentration below
    stay_bead = 0.8
    max_iterations = 6

    ####  HMM - Analysis   ###########
    smoothers = 1  # 0 for no, 1 for yes
    viterbi = 1   # 0 for no, 1 for yes
    composite_bead_from_simulation = 1   # 0 for no, 1 for yes

    ####  Save Simulated Data/Background Using HMM Inputs from Final Iteration  ####
    save_simulation_data = 1  # 0 for no, 1 for yes
    save_simulation_background = 1  # 0 for no, 1 for yes

    ####  File Open Information  ########
    subset_data = 1   # 0 no, 1 yes
    fraction_of_photons_for_subset = 0.4  # value must be <= 1

    file_open_location = '/Users/jenniferhovis/documents/fluxus_josh/data/LargeParticle3June24/'
    open_filename = np.array(['100fM']) #array
    concentration = np.array([1e-13]) #This needs to match open_filename, e.g. if open_filname = ['100fM', '10fM'] then concentration = [1e-13, 1e-14]

    open_filename_extension = '.npy'  # Choices are '.npy' and '.bin'
    data_convert_to_seconds = 1e-12

    ####  File Save Information   ##################
    file_save_location = '/Users/jenniferhovis/Documents/fluxus_josh/data/CodeTesting_HMM/'



    ####End Inputs ##################
    ####################################################################
    #####  Behind the scences ####
    hmm_parameters = [num_for_bead_geometric, stay_background, stay_bead, max_iterations, lambda_0_analysis, lambda_1_analysis]
    hmm_analysis = [smoothers, viterbi, composite_bead_from_simulation]
    save_simulation = [save_simulation_data, save_simulation_background]

    if len(stay_background) != len(open_filename) or len(open_filename) != len(concentration):
        print("Array Lengths do Not Match: ", "Stay Background: ", stay_background, "Open Filename: ", open_filename, "Concentration: ", concentration)
        exit()

    return subset_data, fraction_of_photons_for_subset, bin_width, file_open_location, open_filename, concentration, open_filename_extension, data_convert_to_seconds, file_save_location, hmm_parameters, hmm_analysis, save_simulation


def main():
    ####   Import Variables  ##################
    subset_data, fraction_of_photons_for_subset, bin_width, file_open_location, open_filename, concentration, open_filename_extension, data_convert_to_seconds, file_save_location, hmm_parameters, hmm_analysis, save_simulation = variables()
    num_for_bead_geometric = hmm_parameters[0]
    stay_background = hmm_parameters[1]

    ####   Open Data File  ########################
    for i in range(len(open_filename)):
        time, data = hmm_get_data_module.main(file_open_location, open_filename[i], open_filename_extension, data_convert_to_seconds, bin_width, subset_data, fraction_of_photons_for_subset)

        ####   Arrays For Saving  ######
        final_info_array, arrayPi_arraytosave, arrayMeanCounts_arraytosave, matrixA_arraytosave, array_of_composite_bead_arrays, final_info_array_background, array_of_composite_bead_arrays_background, array_of_composite_bead_arrays_sim, array_of_num_beads_array, row, row_b = hmm_outputs_module.empty_array(25, hmm_parameters[3], len(num_for_bead_geometric), np.max(num_for_bead_geometric), 50)

        file_random_number_to_prevent_overwritting_files = str(np.random.randint(0, 100000 + 1))
        for j in range(len(num_for_bead_geometric)):
            ####   Create arrayPi, arrayMeanCounts, matrixA  ######
            arrayPi, arrayMeanCounts, matrixA, max_n, n = hmm_arrayPi_arrayMeanCounts_matrixA_module.get_hmm_parameters(stay_background[i], hmm_parameters[2], hmm_parameters[4], hmm_parameters[5], bin_width, num_for_bead_geometric[j], np.max(num_for_bead_geometric))

            ####   Baum-Welch  &   Analysis (Decoding, Changepoints, Composite Bead, Statistics, Add to Arrays for Saving)     ######
            arrayPi_alliters, arrayMeanCounts_alliters, matrixA_alliters, final_info_array, array_of_composite_bead_arrays, row, arrayPi_arraytosave, arrayMeanCounts_arraytosave, matrixA_arraytosave = hmm_analysis_module.baum_welch_plus_analysis(hmm_analysis[0], hmm_analysis[1], time, data, 50, hmm_parameters[3], concentration[i], bin_width, fraction_of_photons_for_subset, arrayPi, arrayMeanCounts, matrixA, row, final_info_array, array_of_composite_bead_arrays, n, max_n, num_for_bead_geometric[j], np.max(num_for_bead_geometric), arrayPi_arraytosave, arrayMeanCounts_arraytosave, matrixA_arraytosave)

            #### Get Number of Beads & Composite Bead Detected in Simulated Background  ######
            final_info_array_background, array_of_composite_bead_arrays_background, background, row_b = hmm_analysis_module.find_beads_detected_in_background_simulation(len(data), bin_width, row_b, arrayPi_alliters, arrayMeanCounts_alliters, matrixA_alliters, hmm_analysis[0], hmm_analysis[1], n, 50, final_info_array_background, array_of_composite_bead_arrays_background, num_for_bead_geometric[j], concentration[i], fraction_of_photons_for_subset, hmm_parameters[3])

            ####   Save (Data & Background): (1) Statistics (Every Transition), (2) HMM arrays, (3) Composite Bead   ######
            hmm_outputs_module.save_info(file_save_location, file_random_number_to_prevent_overwritting_files, open_filename[i], final_info_array, final_info_array_background, array_of_composite_bead_arrays, array_of_composite_bead_arrays_background, arrayPi_arraytosave, arrayMeanCounts_arraytosave, matrixA_arraytosave)

            ####   Get Composite Bead From Simulation  ######
            if hmm_analysis[2] == 1:
                array_of_num_beads_array, array_of_composite_bead_arrays_sim, simulated_data = hmm_analysis_module.get_composite_bead_from_simulation(row, array_of_composite_bead_arrays_sim, array_of_num_beads_array, matrixA_alliters, arrayMeanCounts_alliters, time, num_for_bead_geometric[j], 50)
                hmm_outputs_module.save_sim_analysis(file_save_location, file_random_number_to_prevent_overwritting_files, open_filename[i], array_of_num_beads_array, array_of_composite_bead_arrays_sim)

            #### Save Simulation Using Final Iteration Parameters  ######
            if save_simulation[0] == 1:
                if hmm_analysis[2] == 0:
                    simulated_data = hmm_simulation_module.simulation(matrixA_alliters[i], arrayMeanCounts_alliters[i], len(data))
                hmm_outputs_module.save_simulation(file_save_location, file_random_number_to_prevent_overwritting_files, open_filename[i], '_data', simulated_data, num_for_bead_geometric[j], array_of_num_beads_array[(int(row / 2) - 1), 0])
            if save_simulation[1] == 1:
                hmm_outputs_module.save_simulation(file_save_location, file_random_number_to_prevent_overwritting_files, open_filename[i], '_background', background, num_for_bead_geometric[j], 0)


        hmm_outputs_module.remove_empty_rows_save_info(file_save_location, file_random_number_to_prevent_overwritting_files, open_filename[i], final_info_array, final_info_array_background, array_of_composite_bead_arrays, array_of_composite_bead_arrays_background, arrayPi_arraytosave, arrayMeanCounts_arraytosave, matrixA_arraytosave)
        hmm_outputs_module.remove_empty_rows_save_sim_analysis(file_save_location, file_random_number_to_prevent_overwritting_files,
                                                               open_filename[i], array_of_num_beads_array,
                                                               array_of_composite_bead_arrays_sim)


main()
