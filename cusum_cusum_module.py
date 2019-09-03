
import numpy as np
from numba import jit
import math

####     CUSUM     ######################################
#########################################################
#One-Way: Looks only for increasing rate changes, run in both directions - forward detects bead arrival, backwards detects bead departure
#Two-Way: Looks for both increasing rate changes (bead arrival) and decreasing rate changes (bead departure)

#With Arrays: This version is used for plotting, it creates arrays of all the values, likelihood, cummulative sum, current minimun of cummulative sum, index of current minimum
#With Out Arrays: This version is much faster as it does not create arrays of cummulative sum, current minimun of cummulative sum, index of current minimum
#########################################################


####   Functions called by all CUSUM   ########################
#########################################################
def initial_zero_arrays_for_trip_and_changepoint_index(length):
    trip_index = np.zeros(length, dtype=int)
    changepoint_index = np.zeros(length, dtype=int)

    return trip_index, changepoint_index
#########################################################

#########################################################
####   One Way CUSUM     ########################################
#########################################################

#### Functions called only by w/ arrays one-way###################
#########################################################
def initial_zero_arrays_for_plotting_oneway(len_data):
    cummulative_sum = np.zeros(len_data, dtype=np.float64)
    cusum_curr_min = np.zeros(len_data, dtype=np.float64)
    x_current_min = np.zeros(len_data, dtype=np.int)
    decision_fn = np.zeros(len_data, dtype=np.float64)

    return cummulative_sum, cusum_curr_min, x_current_min, decision_fn

def tidy_output_cusum_oneway_w_array(exponential_or_poisson, delay, bin_width, lambda_0, lambda_1, h, time_data, trip_index, changepoint_index, cummulative_sum, cusum_curr_min, x_current_min, decision_fn):
    trip_index, changepoint_index, loglikelihood_ratio, cummulative_sum, cusum_curr_min, x_current_min, decision_fn = core_cusum_oneway_w_arrays(exponential_or_poisson, delay, bin_width, lambda_0, lambda_1, h, time_data, trip_index, changepoint_index, cummulative_sum, cusum_curr_min, x_current_min, decision_fn)
    matrix_for_plotting = np.stack((loglikelihood_ratio, decision_fn, cummulative_sum, cusum_curr_min, x_current_min), axis = -1)

    return trip_index, changepoint_index, matrix_for_plotting

@jit(nopython=True)
def core_cusum_oneway_w_arrays(exponential_or_poisson, delay, bin_width, lambda_0, lambda_1, h, time_data, trip_index, changepoint_index, cummulative_sum, cusum_curr_min, x_current_min, decision_fn):

    data = time_data[:, 1]

    if exponential_or_poisson == 0:
        loglikelihood_ratio = math.log(lambda_1 / lambda_0) * np.ones(len(data)) + data * (-lambda_1 + lambda_0)
    else:
        lambda_0 = lambda_0 * bin_width
        lambda_1 = lambda_1 * bin_width
        loglikelihood_ratio = data * math.log(lambda_1 / lambda_0) + (-lambda_1 + lambda_0) * np.ones(len(data))

    delay_counter = 0

    number_iterations = range(len(data) - 1)
    for x in number_iterations:
        delay_counter +=1
        cummulative_sum[x] = (cummulative_sum[x - 1] + loglikelihood_ratio[x])

        if decision_fn[x - 1] + loglikelihood_ratio[x] >= 0:
            decision_fn[x] = decision_fn[x - 1] + loglikelihood_ratio[x]
        else:
            decision_fn[x] = 0

        if delay_counter >= 1:
            if cummulative_sum[x] < cusum_curr_min[x - 1]:
                cusum_curr_min[x] = cummulative_sum[x]
                x_current_min[x] = x
            else:
                cusum_curr_min[x] = cusum_curr_min[x - 1]
                x_current_min[x] = x_current_min[x - 1]
        else:
            cusum_curr_min[x] = cusum_curr_min[x - 1]
            x_current_min[x] = x_current_min[x - 1]

        if decision_fn[x] > h:
            trip_index[x] = x + 1
            changepoint_index[x] = x_current_min[x] + 1

            decision_fn[x] = 0
            cummulative_sum[x] = 0
            cusum_curr_min[x] = 0

            delay_counter = -delay

    return trip_index, changepoint_index, loglikelihood_ratio, cummulative_sum, cusum_curr_min, x_current_min, decision_fn

def run_cusum_oneway_w_arrays(exponential_or_poisson, bin_width, delay, time_data_forward, time_data_backwards, lambda_0, lambda_1, h):
####Forwards#####################################
    trip_index, changepoint_index = initial_zero_arrays_for_trip_and_changepoint_index(len(time_data_forward))
    cummulative_sum, cusum_curr_min, x_current_min, decision_fn = initial_zero_arrays_for_plotting_oneway(len(time_data_forward))
    trip_index_forward, changepoint_index_forward, matrix_for_plotting_forward = tidy_output_cusum_oneway_w_array(exponential_or_poisson, delay, bin_width, lambda_0, lambda_1, h, time_data_forward, trip_index, changepoint_index, cummulative_sum, cusum_curr_min, x_current_min, decision_fn)

####Backwards#####################################
    trip_index, changepoint_index = initial_zero_arrays_for_trip_and_changepoint_index(len(time_data_backwards))
    cummulative_sum, cusum_curr_min, x_current_min, decision_fn = initial_zero_arrays_for_plotting_oneway(len(time_data_backwards))
    trip_index_backwards, changepoint_index_backwards, matrix_for_plotting_backwards = tidy_output_cusum_oneway_w_array(exponential_or_poisson, delay, bin_width, lambda_0, lambda_1, h, time_data_backwards, trip_index, changepoint_index, cummulative_sum, cusum_curr_min, x_current_min, decision_fn)

    return trip_index_forward, trip_index_backwards, changepoint_index_forward, changepoint_index_backwards, matrix_for_plotting_forward, matrix_for_plotting_backwards

#### Functions called only by w/o arrays one-way ################
#########################################################
@jit(nopython=True)
def core_cusum_oneway_wo_arrays_poisson(delay, lambda_0, lambda_1, h, data, trip_index, changepoint_index):

    decision_fn = 0
    cummulative_sum_prev = 0
    cusum_curr_min = 0
    cusum_curr_min_prev = 0
    x_current_min_prev = 0

    delay_counter = 0

    number_iterations = range(len(data) - 1)
    for x in number_iterations:

        delay_counter += 1

        loglikelihood_ratio = data[x] * math.log(lambda_1 / lambda_0) + (-lambda_1 + lambda_0)

        cummulative_sum = cummulative_sum_prev + loglikelihood_ratio

        if decision_fn + loglikelihood_ratio >= 0:
            decision_fn = decision_fn + loglikelihood_ratio
        else:
            decision_fn = 0

        if delay_counter >= 1:
            if cummulative_sum < cusum_curr_min_prev:
                cusum_curr_min = cummulative_sum
                x_current_min = x
            else:
                cusum_curr_min = cusum_curr_min_prev
                x_current_min = x_current_min_prev

        if decision_fn > h:
            trip_index[x] = x + 1
            changepoint_index[x] = x_current_min + 1

            decision_fn = 0
            cummulative_sum = 0
            cusum_curr_min = 0

            delay_counter = -delay

        cummulative_sum_prev = cummulative_sum
        cusum_curr_min_prev = cusum_curr_min
        x_current_min_prev = x_current_min

    return trip_index, changepoint_index

@jit(nopython=True)
def core_cusum_oneway_wo_arrays_exponential(delay, lambda_0, lambda_1, h, data, trip_index, changepoint_index):

    decision_fn = 0
    cummulative_sum_prev = 0
    cusum_curr_min = 0
    cusum_curr_min_prev = 0
    x_current_min_prev = 0

    delay_counter = 0

    number_iterations = range(len(data) - 1)
    for x in number_iterations:

        delay_counter += 1

        loglikelihood_ratio = math.log(lambda_1 / lambda_0) + data[x] * (-lambda_1 + lambda_0)

        cummulative_sum = cummulative_sum_prev + loglikelihood_ratio

        if decision_fn + loglikelihood_ratio >= 0:
            decision_fn = decision_fn + loglikelihood_ratio
        else:
            decision_fn = 0

        if delay_counter >= 1:
            if cummulative_sum < cusum_curr_min_prev:
                cusum_curr_min = cummulative_sum
                x_current_min = x
            else:
                cusum_curr_min = cusum_curr_min_prev
                x_current_min = x_current_min_prev

        if decision_fn > h:
            trip_index[x] = x + 1
            changepoint_index[x] = x_current_min + 1

            decision_fn = 0
            cummulative_sum = 0
            cusum_curr_min = 0

            delay_counter = -delay

        cummulative_sum_prev = cummulative_sum
        cusum_curr_min_prev = cusum_curr_min
        x_current_min_prev = x_current_min

    return trip_index, changepoint_index

def run_cusum_oneway_wo_arrays(exponential_or_poisson, bin_width, delay, time_data_forward, time_data_backwards, lambda_0, lambda_1, h):

    data_forward = time_data_forward[:, 1]
    data_backwards = time_data_backwards[:, 1]

    h = h.astype(np.float64)    #assigned a dtype b/c numba sometimes flips out and doesn't understand the type
    delay = delay.astype(np.int16)   #assigned dtype (numpy) rather than type (python) b/c they are 1x1 numpy arrays

    if exponential_or_poisson == 0:
        ####Forwards#####################################
        trip_index, changepoint_index = initial_zero_arrays_for_trip_and_changepoint_index(len(time_data_forward))
        trip_index_forward, changepoint_index_forward = core_cusum_oneway_wo_arrays_exponential(delay, lambda_0, lambda_1, h, data_forward, trip_index, changepoint_index)

        ####Backwards#####################################
        trip_index, changepoint_index = initial_zero_arrays_for_trip_and_changepoint_index(len(time_data_backwards))
        trip_index_backwards, changepoint_index_backwards = core_cusum_oneway_wo_arrays_exponential(delay, lambda_0, lambda_1, h, data_backwards, trip_index, changepoint_index)

    else:
        lambda_0 = lambda_0 * bin_width
        lambda_0 = lambda_0.astype(np.float64)      #see note above about assigning dtype

        lambda_1 = lambda_1 * bin_width
        lambda_1 = lambda_1.astype(np.float64)      #see note above about assigning dtype

        ####   Forwards   #####################################
        trip_index, changepoint_index = initial_zero_arrays_for_trip_and_changepoint_index(len(time_data_forward))
        trip_index_forward, changepoint_index_forward = core_cusum_oneway_wo_arrays_poisson(delay, lambda_0, lambda_1, h, data_forward, trip_index, changepoint_index)

        ####   Backwards   #####################################
        trip_index, changepoint_index = initial_zero_arrays_for_trip_and_changepoint_index(len(time_data_backwards))
        trip_index_backwards, changepoint_index_backwards = core_cusum_oneway_wo_arrays_poisson(delay, lambda_0, lambda_1, h, data_backwards, trip_index, changepoint_index)

    return trip_index_forward, trip_index_backwards, changepoint_index_forward, changepoint_index_backwards
#########################################################

#########################################################
####    Two Way CUSUM    ########################################
#########################################################

#### Functions called by w/ arrays & w/o arrays two-way ###########
#########################################################
def initial_zero_arrays_for_direction_array(length):
    direction_array = np.zeros(length)

    return direction_array

#### Functions called only by w/ arrays two-way #################
#########################################################
def initial_zero_arrays_for_plotting_twoway(len_data):
    cummulative_sum_i = np.zeros(len_data, dtype=np.float64)
    cummulative_sum_d = np.zeros(len_data, dtype=np.float64)

    cusum_curr_min_i = np.zeros(len_data, dtype=np.float64)
    cusum_curr_min_d = np.zeros(len_data, dtype=np.float64)

    x_current_min_i = np.zeros(len_data, dtype=np.int)
    x_current_min_d = np.zeros(len_data, dtype=np.int)

    decision_fn_i = np.zeros(len_data, dtype=np.float64)
    decision_fn_d = np.zeros(len_data, dtype=np.float64)

    return cummulative_sum_i, cummulative_sum_d, cusum_curr_min_i, cusum_curr_min_d, x_current_min_i, x_current_min_d, decision_fn_i, decision_fn_d

@jit(nopython=True)
def core_cusum_twoway_w_arrays(exponential_or_poisson, delay, bin_width, lambda_0, lambda_1, h_i, h_d, time_data, trip_index, changepoint_index, direction_array, cummulative_sum_i, cummulative_sum_d, cusum_curr_min_i, cusum_curr_min_d, x_current_min_i, x_current_min_d, decision_fn_i, decision_fn_d):

    data = time_data[:, 1]

    if exponential_or_poisson == 0:
        loglikelihood_ratio_i = math.log(lambda_1 / lambda_0) * np.ones(len(data)) + data * (-lambda_1 + lambda_0)
        loglikelihood_ratio_d = math.log(lambda_0 / lambda_1) * np.ones(len(data)) + data * (-lambda_0 + lambda_1)
    else:
        lambda_0 = lambda_0 * bin_width
        lambda_1 = lambda_1 * bin_width
        loglikelihood_ratio_i = data * math.log(lambda_1 / lambda_0) + (-lambda_1 + lambda_0) * np.ones(len(data))
        loglikelihood_ratio_d = data * math.log(lambda_0 / lambda_1) + (-lambda_0 + lambda_1) * np.ones(len(data))

    delay_counter = 0
    number_iterations = range(len(data) - 1)
    for x in number_iterations:

        delay_counter +=1

        cummulative_sum_i[x] = (cummulative_sum_i[x - 1] + loglikelihood_ratio_i[x])
        cummulative_sum_d[x] = (cummulative_sum_d[x - 1] + loglikelihood_ratio_d[x])

        if decision_fn_i[x - 1] + loglikelihood_ratio_i[x] >= 0:
            decision_fn_i[x] = decision_fn_i[x - 1] + loglikelihood_ratio_i[x]
        else:
            decision_fn_i[x] = 0

        if decision_fn_d[x - 1] + loglikelihood_ratio_d[x] >= 0:
            decision_fn_d[x] = decision_fn_d[x - 1] + loglikelihood_ratio_d[x]
        else:
            decision_fn_d[x] = 0

        if delay_counter >= 1:
            if cummulative_sum_i[x] < cusum_curr_min_i[x - 1]:
                cusum_curr_min_i[x] = cummulative_sum_i[x]
                x_current_min_i[x] = x
            else:
                cusum_curr_min_i[x] = cusum_curr_min_i[x - 1]
                x_current_min_i[x] = x_current_min_i[x - 1]

            if cummulative_sum_d[x] < cusum_curr_min_d[x - 1]:
                cusum_curr_min_d[x] = cummulative_sum_d[x]
                x_current_min_d[x] = x
            else:
                cusum_curr_min_d[x] = cusum_curr_min_d[x - 1]
                x_current_min_d[x] = x_current_min_d[x - 1]

        if decision_fn_i[x] > h_i:
            trip_index[x] = x + 1
            changepoint_index[x] = x_current_min_i[x] + 1
            direction_array[x] = 1

            decision_fn_i[x] = 0
            cummulative_sum_i[x] = 0
            cusum_curr_min_i[x] = 0

            decision_fn_d[x] = 0
            cummulative_sum_d[x] = 0
            cusum_curr_min_d[x] = 0

            delay_counter = -delay

        if decision_fn_d[x] > h_d:
            trip_index[x] = x + 1
            changepoint_index[x] = x_current_min_d[x] + 1
            direction_array[x] = -1

            decision_fn_i[x] = 0
            cummulative_sum_i[x] = 0
            cusum_curr_min_i[x] = 0

            decision_fn_d[x] = 0
            cummulative_sum_d[x] = 0
            cusum_curr_min_d[x] = 0

            delay_counter = -delay

    return trip_index, changepoint_index, direction_array, loglikelihood_ratio_i, loglikelihood_ratio_d, cummulative_sum_i, cummulative_sum_d, cusum_curr_min_i, cusum_curr_min_d, x_current_min_i, x_current_min_d, decision_fn_i, decision_fn_d

def run_cusum_twoway_w_arrays(exponential_or_poisson, bin_width, delay, time_data, lambda_0, lambda_1, h_i, h_d):

    #Create arrays for adding information
    direction_array = initial_zero_arrays_for_direction_array(len(time_data))
    trip_index, changepoint_index = initial_zero_arrays_for_trip_and_changepoint_index(len(time_data))
    cummulative_sum_i, cummulative_sum_d, cusum_curr_min_i, cusum_curr_min_d, x_current_min_i, x_current_min_d, decision_fn_i, decision_fn_d = initial_zero_arrays_for_plotting_twoway(len(time_data))

    #Run CUSUM Two-Way With Arrays
    trip_index, changepoint_index, direction_array, loglikelihood_ratio_i, loglikelihood_ratio_d, cummulative_sum_i, cummulative_sum_d, cusum_curr_min_i, cusum_curr_min_d, x_current_min_i, x_current_min_d, decision_fn_i, decision_fn_d = core_cusum_twoway_w_arrays(exponential_or_poisson, delay, bin_width, lambda_0, lambda_1, h_i, h_d, time_data, trip_index, changepoint_index, direction_array, cummulative_sum_i, cummulative_sum_d, cusum_curr_min_i, cusum_curr_min_d, x_current_min_i, x_current_min_d, decision_fn_i, decision_fn_d)

    #Create matrix of arrays used in plotting (just so things are tidier)
    matrix_for_plotting = np.stack((loglikelihood_ratio_i, loglikelihood_ratio_d, decision_fn_i, decision_fn_d, cummulative_sum_i, cummulative_sum_d, cusum_curr_min_i, cusum_curr_min_d, x_current_min_i, x_current_min_d), axis = -1)

    return trip_index, changepoint_index, direction_array, matrix_for_plotting


#### Functions called only by w/o arrays two-way ###########
########################################################
@jit(nopython=True)
def core_cusum_twoway_wo_arrays_poisson(delay, lambda_0, lambda_1, h_i, h_d, data, trip_index, changepoint_index, direction_array):

    decision_fn_i = 0
    decision_fn_d = 0

    cummulative_sum_prev_i = 0
    cummulative_sum_prev_d = 0

    cusum_curr_min_i = 0
    cusum_curr_min_d = 0
    cusum_curr_min_prev_i = 0
    cusum_curr_min_prev_d = 0

    x_current_min_i = 0
    x_current_min_d = 0
    x_current_min_prev_i = 0
    x_current_min_prev_d = 0

    delay_counter = 0

    number_iterations = range(len(data) - 1)
    for x in number_iterations:

        delay_counter += 1

        loglikelihood_ratio_i = data[x] * math.log(lambda_1 / lambda_0) + (-lambda_1 + lambda_0)
        loglikelihood_ratio_d = data[x] * math.log(lambda_0 / lambda_1) + (-lambda_0 + lambda_1)

        cummulative_sum_i = cummulative_sum_prev_i + loglikelihood_ratio_i
        cummulative_sum_d = cummulative_sum_prev_d + loglikelihood_ratio_d

        if decision_fn_i + loglikelihood_ratio_i >= 0:
            decision_fn_i = decision_fn_i + loglikelihood_ratio_i
        else:
            decision_fn_i = 0

        if decision_fn_d + loglikelihood_ratio_d >= 0:
            decision_fn_d = decision_fn_d + loglikelihood_ratio_d
        else:
            decision_fn_d = 0

        if delay_counter >= 1:
            if cummulative_sum_i < cusum_curr_min_prev_i:
                cusum_curr_min_i = cummulative_sum_i
                x_current_min_i = x
            else:
                cusum_curr_min_i = cusum_curr_min_prev_i
                x_current_min_i = x_current_min_prev_i

            if cummulative_sum_d < cusum_curr_min_prev_d:
                cusum_curr_min_d = cummulative_sum_d
                x_current_min_d = x
            else:
                cusum_curr_min_d = cusum_curr_min_prev_d
                x_current_min_d = x_current_min_prev_d

        if decision_fn_i > h_i:
            trip_index[x] = x + 1
            changepoint_index[x] = x_current_min_i + 1
            direction_array[x] = 1

            decision_fn_i = 0
            cummulative_sum_i = 0
            cusum_curr_min_i = 0

            decision_fn_d = 0
            cummulative_sum_d = 0
            cusum_curr_min_d = 0

            delay_counter = -delay

        if decision_fn_d > h_d:
            trip_index[x] = x + 1
            changepoint_index[x] = x_current_min_d + 1
            direction_array[x] = -1

            decision_fn_i = 0
            cummulative_sum_i = 0
            cusum_curr_min_i = 0

            decision_fn_d = 0
            cummulative_sum_d = 0
            cusum_curr_min_d = 0

            delay_counter = -delay

        cummulative_sum_prev_i = cummulative_sum_i
        cummulative_sum_prev_d = cummulative_sum_d

        cusum_curr_min_prev_i = cusum_curr_min_i
        cusum_curr_min_prev_d = cusum_curr_min_d

        x_current_min_prev_i = x_current_min_i
        x_current_min_prev_d = x_current_min_d

    return trip_index, changepoint_index, direction_array

@jit(nopython=True)
def core_cusum_twoway_wo_arrays_exponential(delay, lambda_0, lambda_1, h_i, h_d, data, trip_index, changepoint_index, direction_array):

    decision_fn_i = 0
    decision_fn_d = 0

    cummulative_sum_prev_i = 0
    cummulative_sum_prev_d = 0

    cusum_curr_min_i = 0
    cusum_curr_min_d = 0
    cusum_curr_min_prev_i = 0
    cusum_curr_min_prev_d = 0

    x_current_min_i = 0
    x_current_min_d = 0
    x_current_min_prev_i = 0
    x_current_min_prev_d = 0

    delay_counter = 0

    number_iterations = range(len(data) - 1)
    for x in number_iterations:

        delay_counter += 1

        loglikelihood_ratio_i = math.log(lambda_1 / lambda_0) + data[x] * (-lambda_1 + lambda_0)
        loglikelihood_ratio_d = math.log(lambda_0 / lambda_1) + data[x] * (-lambda_0 + lambda_1)

        cummulative_sum_i = cummulative_sum_prev_i + loglikelihood_ratio_i
        cummulative_sum_d = cummulative_sum_prev_d + loglikelihood_ratio_d

        if decision_fn_i + loglikelihood_ratio_i >= 0:
            decision_fn_i = decision_fn_i + loglikelihood_ratio_i
        else:
            decision_fn_i = 0

        if decision_fn_d + loglikelihood_ratio_d >= 0:
            decision_fn_d = decision_fn_d + loglikelihood_ratio_d
        else:
            decision_fn_d = 0

        if delay_counter >= 1:
            if cummulative_sum_i < cusum_curr_min_prev_i:
                cusum_curr_min_i = cummulative_sum_i
                x_current_min_i = x
            else:
                cusum_curr_min_i = cusum_curr_min_prev_i
                x_current_min_i = x_current_min_prev_i

            if cummulative_sum_d < cusum_curr_min_prev_d:
                cusum_curr_min_d = cummulative_sum_d
                x_current_min_d = x
            else:
                cusum_curr_min_d = cusum_curr_min_prev_d
                x_current_min_d = x_current_min_prev_d

        if decision_fn_i > h_i:
            trip_index[x] = x + 1
            changepoint_index[x] = x_current_min_i + 1
            direction_array[x] = 1

            decision_fn_i = 0
            cummulative_sum_i = 0
            cusum_curr_min_i = 0

            decision_fn_d = 0
            cummulative_sum_d = 0
            cusum_curr_min_d = 0

            delay_counter = -delay

        if decision_fn_d > h_d:
            trip_index[x] = x + 1
            changepoint_index[x] = x_current_min_d + 1
            direction_array[x] = -1

            decision_fn_i = 0
            cummulative_sum_i = 0
            cusum_curr_min_i = 0

            decision_fn_d = 0
            cummulative_sum_d = 0
            cusum_curr_min_d = 0

            delay_counter = -delay

        cummulative_sum_prev_i = cummulative_sum_i
        cummulative_sum_prev_d = cummulative_sum_d

        cusum_curr_min_prev_i = cusum_curr_min_i
        cusum_curr_min_prev_d = cusum_curr_min_d

        x_current_min_prev_i = x_current_min_i
        x_current_min_prev_d = x_current_min_d

    return trip_index, changepoint_index, direction_array

def run_cusum_twoway_wo_arrays(exponential_or_poisson, bin_width, delay, time_data, lambda_0, lambda_1, h_i, h_d):

    data = time_data[:, 1]

    h_i = h_i.astype(np.float64)    #assigned a dtype b/c numba sometimes flips out and doesn't understand the type
    h_d = h_d.astype(np.float64)    #assigned dtype (numpy) rather than type (python) b/c they are 1x1 numpy arrays
    delay = delay.astype(np.int16)

    #Create arrays for adding information
    direction_array = initial_zero_arrays_for_direction_array(len(time_data))
    trip_index, changepoint_index = initial_zero_arrays_for_trip_and_changepoint_index(len(time_data))

    #Run CUSUM Two-Way With Out Arrays
    if exponential_or_poisson == 0:
        trip_index, changepoint_index, direction_array = core_cusum_twoway_wo_arrays_exponential(delay, lambda_0, lambda_1, h_i, h_d, data, trip_index, changepoint_index, direction_array)
    else:
        lambda_0 = lambda_0 * bin_width
        lambda_0 = lambda_0.astype(np.float64)   #see note above about assigning dtype

        lambda_1 = lambda_1 * bin_width
        lambda_1 = lambda_1.astype(np.float64)    #see note above about assigning dtype

        trip_index, changepoint_index, direction_array = core_cusum_twoway_wo_arrays_poisson(delay, lambda_0, lambda_1, h_i, h_d, data, trip_index, changepoint_index, direction_array)

    return trip_index, changepoint_index, direction_array
#########################################################
