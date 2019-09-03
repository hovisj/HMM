
import cusum_time_data_module
import cusum_cusum_module
import cusum_count_trips_module
import cusum_merge_changepoints_module

def run_analysis_core(oneway_or_twoway_cusum, exponential_or_poisson, time_data, bin_width, lambda0_lambda1_hi_hd_delay):

    if oneway_or_twoway_cusum == 1:

        time_data_backwards = cusum_time_data_module.reverse_the_direction_of_time(time_data)

        trip_index_forward, trip_index_backwards, changepoint_index_forward, changepoint_index_backwards = cusum_cusum_module.run_cusum_oneway_wo_arrays(
                    exponential_or_poisson, bin_width, lambda0_lambda1_hi_hd_delay[4], time_data, time_data_backwards,
                    lambda0_lambda1_hi_hd_delay[0], lambda0_lambda1_hi_hd_delay[1], lambda0_lambda1_hi_hd_delay[2])

        trip_changepoint_numtrips_direction = cusum_count_trips_module.count_trips_and_return_matrix_oneway(
                    trip_index_forward, trip_index_backwards, changepoint_index_forward, changepoint_index_backwards,
                    len(time_data))

    else:
        trip_index, changepoint_index, direction_array = cusum_cusum_module.run_cusum_twoway_wo_arrays(
                    exponential_or_poisson, bin_width, lambda0_lambda1_hi_hd_delay[4], time_data,
                    lambda0_lambda1_hi_hd_delay[0], lambda0_lambda1_hi_hd_delay[1], lambda0_lambda1_hi_hd_delay[2],
                    lambda0_lambda1_hi_hd_delay[3])

        trip_changepoint_numtrips_direction = cusum_count_trips_module.count_trips_and_return_matrix_twoway(
                    trip_index, changepoint_index, direction_array)

    trip_changpoint_numtrips_direction_nummerge = cusum_merge_changepoints_module.merge_consequtive_changepoint_in_same_direction(
                oneway_or_twoway_cusum, trip_changepoint_numtrips_direction)

    return trip_changpoint_numtrips_direction_nummerge