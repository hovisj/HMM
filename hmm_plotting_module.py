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
    plt.vlines(peak_locations, ymin=-8, ymax=-6, color='black', linestyle='solid')
    plt.vlines(bead_start_smoothers, ymin = -2, ymax = 0, color='purple', linestyle='solid')
    plt.vlines(bead_stop_smoothers, ymin=-2, ymax=0, color='yellow', linestyle='solid')
    plt.vlines(bead_start_viterbi, ymin = -4, ymax = -2, color='blue', linestyle='solid')
    plt.vlines(bead_stop_viterbi, ymin=-4, ymax=-2, color='orange', linestyle='solid')
    plt.vlines(cp_merged_forward_time_2, ymin = -6, ymax = -4, color='green', linestyle='solid')
    plt.vlines(cp_merged_backward_time_2, ymin=-6, ymax=-4, color='red', linestyle='solid')
    plt.ylim([-12.1, np.amax(counts)*1.2])
    plt.xlim([time[0], time[-1]])
    plt.xlabel('Time (s)')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.show()


def plot_HMM_CUSUM_Peaks_rectangles(peak_locations, changepoints_smoothers, changepoints_viterbi, trip_changpoint_numtrips_direction_nummerge_2way, time_data, bin_width):
    counts = time_data[:, 1]
    time = time_data[:, 0]

    peak_locations = peak_locations[peak_locations > time[0]]
    peak_locations = peak_locations[peak_locations < time[-1]]
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

    #fig = plt.figure(figsize=(14, 6), dpi=80)
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True, figsize=(14, 6), dpi=80)
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    ax1.plot(time, counts, marker='', color='black', linewidth=1)
    ax1.annotate('HMM: Smoothers', xy=(1, 0.7), xycoords='axes fraction', fontsize=14, color='purple',
                xytext=(-5, 5), textcoords='offset points',
                ha='right', va='bottom')

    ax1.set_ylabel('Counts')
    ax2.plot(time, counts, marker='', color='black', linewidth=1)
    ax2.annotate('HMM: Viterbi', xy=(1, 0.7), xycoords='axes fraction', fontsize=14, color='blue',
                xytext=(-5, 5), textcoords='offset points',
                ha='right', va='bottom')
    ax2.set_ylabel('Counts')
    ax3.plot(time, counts, marker='', color='black', linewidth=1)
    ax3.annotate('CUSUM', xy=(1, 0.7), xycoords='axes fraction', fontsize=14, color='orange',
                xytext=(-5, 5), textcoords='offset points',
                ha='right', va='bottom')
    ax3.set_ylabel('Counts')
    ax4.plot(time, counts, marker='', color='black', linewidth=1)
    ax4.annotate('Peak Fitting - Dashed', xy=(1, 0.7), xycoords='axes fraction', fontsize=14, color='black',
                xytext=(-5, 5), textcoords='offset points',
                ha='right', va='bottom')
    ax4.vlines(peak_locations, -0.1, np.amax(counts) * 1.2, color='black', linestyle='dashed')
    ax4.set_ylabel('Counts')

    ax4.axvspan(time[0], time[-1], facecolor='white', alpha=0.1, lw=0)
    ax3.axvspan(time[0], time[-1], facecolor='white', alpha=0.1, lw=0)
    ax2.axvspan(time[0], time[-1], facecolor='white', alpha=0.1, lw=0)
    ax1.axvspan(time[0], time[-1], facecolor='white', alpha=0.1, lw=0)

    for i in range(len(bead_start_smoothers)):
        ax1.axvspan(bead_start_smoothers[i], bead_stop_smoothers[i], facecolor='darkorchid', alpha=0.2, lw=0)
    #for i in range(len(bead_start_smoothers)):
    #    ax1.axvspan(bead_stop_smoothers[i], bead_start_smoothers[i], facecolor='grey', alpha=0.1, lw=0)

    for i in range(len(bead_start_viterbi)):
        ax2.axvspan(bead_start_viterbi[i], bead_stop_viterbi[i], facecolor='royalblue', alpha=0.2, lw=0)
    #for i in range(len(bead_start_viterbi)):
    #    ax2.axvspan(bead_stop_viterbi[i], bead_start_viterbi[i], facecolor='grey', alpha=0.1, lw=0)

    for i in range(len(cp_merged_forward_time_2)):
        ax3.axvspan(cp_merged_forward_time_2[i], cp_merged_backward_time_2[i], facecolor='tab:orange', alpha=0.2, lw=0)
    #for i in range(len(cp_merged_forward_time_2)):
    #    ax3.axvspan(cp_merged_backward_time_2[i], cp_merged_forward_time_2[i], facecolor='grey', alpha=0.1, lw=0)

    plt.xlabel('Time (s)')

    plt.show()


def get_time_of_changepoint_for_plotting(trip_changepoint_numtrips_direction, time_data):
    time = time_data[:, 0]

    if trip_changepoint_numtrips_direction[0, 3] == -1:
        trip_changepoint_numtrips_direction = trip_changepoint_numtrips_direction[1:-1, :]

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


