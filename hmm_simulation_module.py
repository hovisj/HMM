import numpy as np
from numba import jit

@jit(nopython=True)
def simulation(matrixA, arrayMeanCounts, num_data_points):
    state = np.zeros(num_data_points, dtype=np.int64)
    data = np.zeros(num_data_points, dtype=np.int64)

    number_of_iterations = range(num_data_points)
    for m in number_of_iterations:
        shift = 0
        rand = np.random.random_sample()
        i = state[m - 1]
        for j in range(len(matrixA)):
            if shift <= rand < matrixA[i, j] + shift:
                state[m] = j
                break
            shift = shift + matrixA[i, j]
        curr_state = state[m]
        meancounts = arrayMeanCounts[curr_state]
        data[m] = np.random.poisson(meancounts)

    return state, data

@jit(nopython=True)
def simulate_background(lambda0, num_data_points, bin_width):
    background = np.random.poisson(lambda0, num_data_points)
    time = np.arange(num_data_points) * bin_width

    return time, background

