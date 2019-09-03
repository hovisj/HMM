
import numpy as np
from numba import jit

def merge_consequtive_changepoint_in_same_direction(oneway_or_twoway_cusum, trip_changepoint_numtrips_direction):

    if oneway_or_twoway_cusum == 1:
        trip_index_reduced, changepoint_index_reduced,  direction_array_reduced, direction_rollup, direction_rolldown = merge_consequtive_changepoint_in_same_direction_part1_oneway(trip_changepoint_numtrips_direction[:, 0], trip_changepoint_numtrips_direction[:, 1], trip_changepoint_numtrips_direction[:, 3])
    else:
        trip_index_reduced, changepoint_index_reduced,  direction_array_reduced, direction_rollup, direction_rolldown = merge_consequtive_changepoint_in_same_direction_part1_twoway(trip_changepoint_numtrips_direction[:, 0], trip_changepoint_numtrips_direction[:, 1], trip_changepoint_numtrips_direction[:, 3])
    number_of_trips_reduced, onesformerge, number_of_merges = merge_consequtive_changepoint_in_same_direction_part2(len(changepoint_index_reduced), len(trip_changepoint_numtrips_direction))
    number_of_trips_reduced, number_of_merges = merge_consequtive_changepoint_in_same_direction_part3(trip_changepoint_numtrips_direction[:, 2], number_of_merges, number_of_trips_reduced, onesformerge, trip_changepoint_numtrips_direction[:, 3], direction_rollup, direction_rolldown)
    trip_changpoint_numtrips_direction_nummerge = merge_consequtive_changepoint_in_same_direction_part4(trip_index_reduced, changepoint_index_reduced, direction_array_reduced, number_of_trips_reduced, number_of_merges)

    return trip_changpoint_numtrips_direction_nummerge

@jit(nopython=True)
def merge_consequtive_changepoint_in_same_direction_part1_oneway(trip_index, changepoint_index, direction_array):
    direction_rollup = np.roll(direction_array, -1)
    direction_rolldown = np.roll(direction_array, 1)

    indices_for_merging = np.where((direction_array == 1) & (direction_rolldown == -1) | (direction_array == -1) & (direction_rollup == 1))[0]

    trip_index_reduced = np.take(trip_index, indices_for_merging)
    changepoint_index_reduced = np.take(changepoint_index, indices_for_merging)
    direction_array_reduced = np.take(direction_array, indices_for_merging)

    return trip_index_reduced, changepoint_index_reduced, direction_array_reduced, direction_rollup, direction_rolldown
@jit(nopython=True)
def merge_consequtive_changepoint_in_same_direction_part1_twoway(trip_index, changepoint_index, direction_array):
    direction_rollup = np.roll(direction_array, -1)
    direction_rolldown = np.roll(direction_array, 1)

    indices_for_merging = np.where((direction_array == 1) & (direction_rolldown == -1) | (direction_array == -1) & (direction_rolldown == 1))[0]

    trip_index_reduced = np.take(trip_index, indices_for_merging)
    changepoint_index_reduced = np.take(changepoint_index, indices_for_merging)
    direction_array_reduced = np.take(direction_array, indices_for_merging)

    return trip_index_reduced, changepoint_index_reduced, direction_array_reduced, direction_rollup, direction_rolldown

def merge_consequtive_changepoint_in_same_direction_part2(length_reduced, length):
    number_of_trips_reduced = np.zeros(length_reduced, dtype=int)
    onesformerge = np.ones(length, dtype=int)
    number_of_merges = np.zeros(length_reduced, dtype=int)

    return number_of_trips_reduced, onesformerge, number_of_merges
@jit(nopython=True)
def merge_consequtive_changepoint_in_same_direction_part3(number_of_trips, number_of_merges, number_of_trips_reduced, onesformerge, direction_array, direction_rollup, direction_rolldown):

    indices_to_sum_num_trips = np.where((direction_array == 1) & (direction_rolldown == -1) | (direction_array == -1) & (direction_rolldown == 1))[0]

    iteration_length = range(len(number_of_trips_reduced) - 1)
    for i in iteration_length:
        number_of_trips_reduced[i] = np.sum(number_of_trips[indices_to_sum_num_trips[i]:indices_to_sum_num_trips[i + 1]])
        number_of_merges[i] = np.sum(onesformerge[indices_to_sum_num_trips[i]:indices_to_sum_num_trips[i + 1]]) - 1
    number_of_trips_reduced[-1] = np.sum(number_of_trips[indices_to_sum_num_trips[-1]::])
    number_of_merges[-1] = np.sum(onesformerge[indices_to_sum_num_trips[-1]::]) - 1

    return number_of_trips_reduced, number_of_merges

def merge_consequtive_changepoint_in_same_direction_part4(trip_index_reduced, changepoint_index_reduced, direction_array_reduced, number_of_trips_reduced, number_of_merges):
    trip_changpoint_numtrips_direction_nummerge = np.stack((trip_index_reduced, changepoint_index_reduced, number_of_trips_reduced, direction_array_reduced, number_of_merges), axis=-1)

    return trip_changpoint_numtrips_direction_nummerge

