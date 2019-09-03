import numpy as np
import matplotlib.pyplot as plt

def plot_HMM_CUSUM(changepoints_smoothers, changepoints_viterbi, trip_changpoint_numtrips_direction_nummerge_2way, time_data, bin_width):
    counts = time_data[:, 1]
    time = time_data[:, 0]

    #changepoints from cusum
    cp_merged_forward_time_2, cp_merged_backward_time_2 = get_time_of_changepoint_for_plotting(trip_changpoint_numtrips_direction_nummerge_2way, time_data)

    #changepoints from hmm
    #smoothers
    changepoints_smoothers = changepoints_smoothers * bin_width + time[0]
    bead_start_smoothers = changepoints_smoothers[::2]
    bead_stop_smoothers = changepoints_smoothers[1::2]

    #viterbi
    changepoints_viterbi = changepoints_viterbi * bin_width + time[0]
    bead_start_viterbi = changepoints_viterbi[::2]
    bead_stop_viterbi = changepoints_viterbi[1::2]

    plt.figure(figsize=(14, 6), dpi=80)

    plt.plot(time, counts, marker='', color='black', linewidth=1, label='Smoothers - Purple/Yellow; Viterbi - Blue/Orange; CUSUM - Green/Red')
    plt.legend(loc='lower center')
    plt.vlines(bead_start_smoothers, ymin = -2, ymax = 0, color='purple', linestyle='solid')
    plt.vlines(bead_stop_smoothers, ymin=-2, ymax=0, color='yellow', linestyle='solid')
    plt.vlines(bead_start_viterbi, ymin = -4, ymax = -2, color='blue', linestyle='solid')
    plt.vlines(bead_stop_viterbi, ymin=-4, ymax=-2, color='orange', linestyle='solid')
    plt.vlines(cp_merged_forward_time_2, ymin = -6, ymax = -4, color='green', linestyle='solid')
    plt.vlines(cp_merged_backward_time_2, ymin=-6, ymax=-4, color='red', linestyle='solid')
    plt.ylim([-10.1, np.amax(counts)*1.2])
    plt.xlim([time[0], time[-1]])

    plt.tight_layout()
    plt.show()

def plot_HMM_CUSUM_Peaks(peak_locations, changepoints_smoothers, changepoints_viterbi, trip_changpoint_numtrips_direction_nummerge_2way, time_data, bin_width):
    counts = time_data[:, 1]
    time = time_data[:, 0]

    #changepoints from cusum
    cp_merged_forward_time_2, cp_merged_backward_time_2 = get_time_of_changepoint_for_plotting(trip_changpoint_numtrips_direction_nummerge_2way, time_data)

    #changepoints from hmm
    #smoothers
    changepoints_smoothers = changepoints_smoothers * bin_width + time[0]
    bead_start_smoothers = changepoints_smoothers[::2]
    bead_stop_smoothers = changepoints_smoothers[1::2]

    #viterbi
    changepoints_viterbi = changepoints_viterbi * bin_width + time[0]
    bead_start_viterbi = changepoints_viterbi[::2]
    bead_stop_viterbi = changepoints_viterbi[1::2]

    plt.figure(figsize=(14, 6), dpi=80)

    plt.plot(time, counts, marker='', color='black', linewidth=1, label='Smoothers - Purple/Yellow; Viterbi - Blue/Orange; CUSUM - Green/Red; Peaks - Black')
    plt.legend(loc='lower center')
#    for i in range(len(bead_start_smoothers)):
#        plt.axvspan(bead_start_smoothers[i], bead_stop_smoothers[i], color='purple', alpha=0.5, lw=0)
    plt.vlines(peak_locations, ymin=-8, ymax=-6, color='black', linestyle='solid')
    plt.vlines(bead_start_smoothers, ymin = -2, ymax = 0, color='purple', linestyle='solid')
    plt.vlines(bead_stop_smoothers, ymin=-2, ymax=0, color='yellow', linestyle='solid')
    plt.vlines(bead_start_viterbi, ymin = -4, ymax = -2, color='blue', linestyle='solid')
    plt.vlines(bead_stop_viterbi, ymin=-4, ymax=-2, color='orange', linestyle='solid')
    plt.vlines(cp_merged_forward_time_2, ymin = -6, ymax = -4, color='green', linestyle='solid')
    plt.vlines(cp_merged_backward_time_2, ymin=-6, ymax=-4, color='red', linestyle='solid')
    plt.ylim([-12.1, np.amax(counts)*1.2])
    plt.xlim([time[0], time[-1]])

    plt.tight_layout()
    plt.show()

def get_time_of_changepoint_for_plotting(trip_changepoint_numtrips_direction, time_data):
    time = time_data[:, 0]

    forward = trip_changepoint_numtrips_direction[trip_changepoint_numtrips_direction[:, 3] == 1]
    cp_forward = forward[:, 1]
    cp_forward_time = np.zeros(len(cp_forward))
    for i in range(len(cp_forward)):
        cp_forward_time[i] = time[int(cp_forward[i])]

    backward = trip_changepoint_numtrips_direction[trip_changepoint_numtrips_direction[:, 3] == -1]
    cp_backward = backward[:, 1]
    cp_backward_time = np.zeros(len(cp_backward))
    for i in range(len(cp_backward)):
        cp_backward_time[i] = time[int(cp_backward[i])]

    return cp_forward_time, cp_backward_time


