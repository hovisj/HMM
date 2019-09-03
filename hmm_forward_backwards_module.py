import numpy as np
import math
from numba import jit
from numpy.linalg import multi_dot


#@jit(nopython=True)
def run_algorithms_notscaled(data, arrayPi, matrixA, arrayMeanCounts):
    forward_likelihood = np.zeros((len(data), len(matrixA)))
    backwards_likelihood = np.zeros((len(data), len(matrixA)))

    #  Forward
    #  Initialization
    matrixB = get_matrixB(data[0], arrayMeanCounts)
    forward_likelihood[0, :] = np.dot(matrixB, arrayPi)

    #  Recursion
    number_of_iterations = range(len(data) - 1)
    for k in number_of_iterations:
        current_index = k
        matrixB =  get_matrixB(data[current_index + 1], arrayMeanCounts)
        forward_likelihood[current_index + 1, :] = np.linalg.multi_dot([matrixB, matrixA, forward_likelihood[current_index, :]])

    #  Backwards
    #  Initialization
    backwards_likelihood[-1, :] = np.ones(len(arrayMeanCounts))

    #  Recursion
    number_of_iterations = range(len(data) - 1)
    for k in number_of_iterations:
        current_index = len(data) - 2 - k
        matrixB = get_matrixB(data[current_index + 1], arrayMeanCounts)
        backwards_likelihood[current_index, :] = np.linalg.multi_dot([np.transpose(matrixA), matrixB, backwards_likelihood[current_index + 1, :]])

    #  Compare Forward and Reverse
    final_forward_likelihood = np.sum(forward_likelihood[-1, :])

    matrixB = get_matrixB(data[0], arrayMeanCounts)
    final_backwards_likelihood = np.linalg.multi_dot([np.transpose(arrayPi), matrixB, backwards_likelihood[0, :]])

    print(final_forward_likelihood, final_backwards_likelihood)

    return forward_likelihood, backwards_likelihood

#@jit(nopython=True)
def run_algorithms_scaled(data, arrayPi, matrixA, arrayMeanCounts):
    forward_likelihood = np.zeros((len(data), len(matrixA)), dtype=np.float64)
    backwards_likelihood = np.zeros((len(data), len(matrixA)), dtype=np.float64)
    scale_factor = np.zeros(len(data), dtype=np.float64)

    #  Forward
    #  Initialization
    matrixB = get_matrixB(data[0], arrayMeanCounts)
    forward_likelihood[0, :] = np.dot(matrixB, arrayPi)
    scale_factor[0] = 1/np.sum(forward_likelihood[0, :])
    forward_likelihood[0, :] = forward_likelihood[0, :] * scale_factor[0]

    #  Loop from 1 to T - 1 by +1
    number_of_iterations = range(len(data) - 1)
    for k in number_of_iterations:
        current_index = k
        matrixB = get_matrixB(data[current_index + 1], arrayMeanCounts)
        forward_likelihood[current_index + 1, :] = multi_dot([matrixB, matrixA, forward_likelihood[current_index, :]])
        #forward_likelihood[current_index + 1, :] = np.dot(np.dot(matrixB, matrixA), forward_likelihood[current_index, :])
        scale_factor[current_index + 1] = 1 / np.sum(forward_likelihood[current_index + 1, :])
        forward_likelihood[current_index + 1, :] = forward_likelihood[current_index + 1, :] * scale_factor[current_index + 1]

    #  Backwards
    #  Initialization
    backwards_likelihood[-1, :] = np.ones(len(arrayMeanCounts)) * scale_factor[-1]

    #  Loop from T - 2 to 0 by -1
    number_of_iterations = range(len(data) - 1)
    for k in number_of_iterations:
        current_index = len(data) - 2 - k
        matrixB = get_matrixB(data[current_index + 1], arrayMeanCounts)
        backwards_likelihood[current_index, :] = multi_dot([np.transpose(matrixA), matrixB, backwards_likelihood[current_index + 1, :]])
        #backwards_likelihood[current_index, :] = np.dot(np.dot(np.transpose(matrixA), matrixB), backwards_likelihood[current_index + 1, :])
        backwards_likelihood[current_index, :] = backwards_likelihood[current_index, :] * scale_factor[current_index]

    #  Compare Forward and Reverse
    final_forward_likelihood = np.sum(forward_likelihood[-1, :])
    matrixB = get_matrixB(data[0], arrayMeanCounts)
    final_backwards_likelihood = np.linalg.multi_dot([np.transpose(arrayPi), matrixB, backwards_likelihood[0, :]])
    print(final_forward_likelihood, final_backwards_likelihood)

    return forward_likelihood, backwards_likelihood, scale_factor

def get_poisson(meancounts: float, data: int) -> float:
    return meancounts ** data * math.exp(-meancounts) / math.factorial(data)

@jit(nopython=True)
def run_algorithms_scaled_for_numba(number_of_states, number_of_data_points, data, arrayPi, matrixA, arrayMeanCounts):
    forward_likelihood = np.zeros((number_of_data_points, number_of_states), dtype=np.float64)
    backwards_likelihood = np.zeros((number_of_data_points, number_of_states), dtype=np.float64)
    scale_factor = np.zeros(number_of_data_points, dtype=np.float64)

    #  Forward
    #  Initialization
    fact = math.gamma(data[0] + 1)
    for i in range(number_of_states):
        forward_likelihood[0, i] = arrayPi[i] * (arrayMeanCounts[i]**data[0] * np.exp(-arrayMeanCounts[i])/fact)
    scale_factor[0] = 1/np.sum(forward_likelihood[0, :])
    forward_likelihood[0, :] = forward_likelihood[0, :] * scale_factor[0]

    #  Loop from 1 to T - 1 by +1
    number_of_iterations = range(number_of_data_points - 1)
    for k in number_of_iterations:
        t = k + 1
        fact = math.gamma(data[t] + 1)
        for i in range(number_of_states):
            for j in range(number_of_states):
                forward_likelihood[t, i] = forward_likelihood[t, i] + forward_likelihood[t - 1, j] * matrixA[j, i]
            forward_likelihood[t, i] = forward_likelihood[t, i] * (arrayMeanCounts[i]**data[t] * np.exp(-arrayMeanCounts[i])/fact)
        scale_factor[t] = 1 / np.sum(forward_likelihood[t, :])
        forward_likelihood[t, :] = forward_likelihood[t, :] * scale_factor[t]

    #  Backwards
    #  Initialization
    backwards_likelihood[-1, :] = np.ones(number_of_states) * scale_factor[-1]

    #  Loop from T - 2 to 0 by -1
    number_of_iterations = range(number_of_data_points - 1)
    for k in number_of_iterations:
        t = number_of_data_points - 2 - k
        fact = math.gamma(data[t + 1] + 1)
        for i in range(number_of_states):
            for j in range(number_of_states):
                backwards_likelihood[t, i] = backwards_likelihood[t, i] + matrixA[i, j] * (arrayMeanCounts[j]**data[t + 1] * np.exp(-arrayMeanCounts[j])/fact) * backwards_likelihood[t + 1, j]
        backwards_likelihood[t, :] = backwards_likelihood[t, :] * scale_factor[t]

    loglikelihood = -np.sum(np.log(scale_factor))
    print("loglikelihood: ", loglikelihood)

    return forward_likelihood, backwards_likelihood, loglikelihood

