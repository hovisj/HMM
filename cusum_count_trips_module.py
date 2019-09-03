import numpy as np
from numba import jit

############################
#CUSUM One-Way (Forward and Backwards):Count trips and return trip_changepoint_numtrips_direction matrix
def count_trips_and_return_matrix_oneway(trip_index_forward, trip_index_backwards, changepoint_index_forward, changepoint_index_backwards, num_photons_or_bins):
#   Forwards
    direction = 1
    trip_index_condensed_forward, changepoint_index_condensed_forward, number_of_trips_forward = find_number_of_trips_each_changepoint(trip_index_forward, changepoint_index_forward)
    matrix_forward = make_index_matrix_oneway(trip_index_condensed_forward, changepoint_index_condensed_forward, number_of_trips_forward, direction, num_photons_or_bins)

#   Backwards
    direction = -1
    trip_index_condensed_backwards, changepoint_index_condensed_backwards, number_of_trips_backwards = find_number_of_trips_each_changepoint(trip_index_backwards, changepoint_index_backwards)
    matrix_backwards = make_index_matrix_oneway(trip_index_condensed_backwards, changepoint_index_condensed_backwards, number_of_trips_backwards, direction, num_photons_or_bins)

#   Put Forward and Backwards Information Together into Single Matrix
    trip_changepoint_numtrips_direction = np.vstack((matrix_forward, matrix_backwards))

#   Sort Matrix by Changepoint (Ascending)
    trip_changepoint_numtrips_direction = sort(trip_changepoint_numtrips_direction)

    return trip_changepoint_numtrips_direction

############################
#CUSUM Two-Way: Count trips and return trip_changepoint_numtrips_direction matrix
def count_trips_and_return_matrix_twoway(trip_index, changepoint_index, direction_array):
    trip_changepoint_direction = np.stack((trip_index, changepoint_index, direction_array), axis = -1)

#Increasing
    direction = 1
    trip_changepoint_direction_increasing = trip_changepoint_direction[np.where(trip_changepoint_direction[:, 2] == direction)]
    trip_index_condensed_increasing, changepoint_index_condensed_increasing, number_of_trips_increasing = find_number_of_trips_each_changepoint(trip_changepoint_direction_increasing[:, 0], trip_changepoint_direction_increasing[:, 1])
    matrix_increasing = make_index_matrix_twoway(trip_index_condensed_increasing, changepoint_index_condensed_increasing, number_of_trips_increasing, direction)

#Decreasing
    direction = -1
    trip_changepoint_direction_decreasing = trip_changepoint_direction[np.where(trip_changepoint_direction[:, 2] == direction)]
    trip_index_condensed_decreasing, changepoint_index_condensed_decreasing, number_of_trips_decreasing = find_number_of_trips_each_changepoint(trip_changepoint_direction_decreasing[:, 0], trip_changepoint_direction_decreasing[:, 1])
    matrix_decreasing = make_index_matrix_twoway(trip_index_condensed_decreasing, changepoint_index_condensed_decreasing, number_of_trips_decreasing, direction)

#   Put Increasing and Decreasing Information Together into Single Matrix
    trip_changepoint_numtrips_direction = np.vstack((matrix_increasing, matrix_decreasing))

#   Sort Matrix by Changepoint (Ascending)
    trip_changepoint_numtrips_direction = sort(trip_changepoint_numtrips_direction)

    return trip_changepoint_numtrips_direction


############################
#Used by both one-way and two-way
def find_number_of_trips_each_changepoint(trip_index, changepoint_index):
    changepoint_index = changepoint_index[np.nonzero(changepoint_index)]
    trip_index = trip_index[np.nonzero(trip_index)]

    changepoint_index_condensed, indices_list, number_of_trips = np.unique(changepoint_index, return_index=True, return_counts=True)
    trip_index_condensed = np.take(trip_index, indices_list)

    return trip_index_condensed, changepoint_index_condensed, number_of_trips

@jit(nopython=True)
def sort(trip_changepoint_numtrips_direction):
    trip_changepoint_numtrips_direction = trip_changepoint_numtrips_direction[np.argsort(trip_changepoint_numtrips_direction[:, 1])]

    return trip_changepoint_numtrips_direction

############################
#Used only by one-way
def make_index_matrix_oneway(trip_index, changepoint_index, number_of_trips, direction, num_photons_or_bins):
    direction_array = direction * np.ones(len(trip_index))
    if direction == 1:
        matrix = np.stack((trip_index, changepoint_index, number_of_trips, direction_array), axis=-1)
    else:
        backwards_flip = (num_photons_or_bins - 1)*np.ones(len(trip_index))
        trip_index = backwards_flip - trip_index
        changepoint_index = backwards_flip - changepoint_index
        matrix = np.stack((trip_index, changepoint_index, number_of_trips, direction_array), axis=-1)

    return matrix

############################
#Used only by two-way
def make_index_matrix_twoway(trip_index, changepoint_index, number_of_trips, direction):
    direction_array = direction * np.ones(len(trip_index))
    index_matrix = np.stack((trip_index, changepoint_index, number_of_trips, direction_array), axis = -1)

    return index_matrix

