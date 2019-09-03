import numpy as np
import math
from numba import jit

# Opens the file
def experiment_get_photon_arrival_times_from_file(file_open_location, open_filename, open_filename_extension, data_convert_to_seconds):

    if open_filename_extension == '.bin':
        photon_arrival_times = np.fromfile(file_open_location + open_filename + open_filename_extension, dtype='uint64', sep='')

    elif open_filename_extension == '.npy':
        photon_arrival_times = np.load(file_open_location + open_filename + open_filename_extension)

    else:
        photon_arrival_times = np.loadtxt(file_open_location + open_filename + open_filename_extension, delimiter=",", usecols=[0])
        photon_arrival_times = np.array(photon_arrival_times)

    print("Opened: ", file_open_location + open_filename)

    photon_arrival_times = photon_arrival_times * data_convert_to_seconds

    print("Number of Photons: ", len(photon_arrival_times))
    print("Start Time of Analysis: ", photon_arrival_times[0])
    print("Stop Time of Analysis: ", photon_arrival_times[-1])

    return photon_arrival_times

@jit(nopython=True)
def make_bins(photon_arrival_times, bin_width):
    duration = photon_arrival_times[-1] - photon_arrival_times[0]
    number_bins = np.floor((duration+bin_width)/bin_width)
    number_bins = int(number_bins)

    bins = np.arange(number_bins) * bin_width + photon_arrival_times[0]

    return bins

@jit(nopython=True)
def find_number_counts_per_bin(photon_arrival_times, bins):
    counts, edge = np.histogram(photon_arrival_times, bins)
    edge = edge[:-1]

    return edge, counts

def main(file_open_location, open_filename, open_filename_extension, data_convert_to_seconds, bin_width, subset_data, photons_subset):
    photon_arrival_times = experiment_get_photon_arrival_times_from_file(file_open_location, open_filename, open_filename_extension, data_convert_to_seconds)
    if subset_data == 1:
        if photons_subset > 1:
            fit = False
            while fit == False:
                start = int(len(photon_arrival_times) * np.random.uniform())
                end = start + photons_subset
                if end <= len(photon_arrival_times):
                    fit = True
        else:
            start = 0
            end = int(photons_subset * len(photon_arrival_times))
        photon_arrival_times = photon_arrival_times[start:end]

    bins = make_bins(photon_arrival_times, bin_width)
    time, data = find_number_counts_per_bin(photon_arrival_times, bins)

    return time, data