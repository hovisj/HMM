
import numpy as np
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

# Makes 2D array of time and data
def get_time_and_data(photon_arrival_times, bin_width, exponential_or_poisson):

    if exponential_or_poisson == 0:
        time = photon_arrival_times
        photon_interarrival_times = get_photon_interarrival_times(time)
        data = np.append(photon_arrival_times[0], photon_interarrival_times)

    else:
        bins = make_bins(photon_arrival_times, bin_width)
        time, data = find_number_counts_per_bin(photon_arrival_times, bins)

    time_data = np.stack((time, data), axis=-1)

    return time_data

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

@jit(nopython=True)
def get_photon_interarrival_times(photon_arrival_times):
    photon_interarrival_times = np.ediff1d(photon_arrival_times)

    return photon_interarrival_times

# Reverses time for the one-way analysis
@jit(nopython=True)
def reverse_the_direction_of_time(time_data_forward):
    time_data_backwards = time_data_forward[np.argsort(-time_data_forward[:, 0])]

    return time_data_backwards





