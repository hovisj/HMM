
import numpy as np

def get_changepoint_time_index_direction(decoded, time):
    decoded = reduce_decoded_to_two_states(decoded)
    time_index_direction = find_changepoints(decoded, time)
    changepoint_index = time_index_direction[:, 1].astype(np.int64)

    return changepoint_index

def reduce_decoded_to_two_states(decoded):
    decoded[decoded > 1] = 1

    return decoded

def find_changepoints(decoded, time):
    index = np.arange(len(decoded))

    # Find Bead Start Times
    find_start = decoded - np.roll(decoded, 1)
    temp = np.column_stack((time, index, find_start))
    temp = temp[temp[:, 2] == 1]
    bead_start_time = temp[:, 0]
    bead_start_index = temp[:, 1]

    # Find Bead Stop Times
    find_stop = decoded - np.roll(decoded, -1)
    temp = np.column_stack((time, index, find_stop))
    temp = temp[temp[:, 2] == 1]
    bead_stop_time = temp[:, 0]
    bead_stop_index = temp[:, 1]

    # Create arrays with: Index, & Direction
    direction = np.ones(len(bead_start_index))
    bead_start_time_index_direction = np.column_stack((bead_start_time, bead_start_index, direction))
    bead_stop_time_index_direction = np.column_stack((bead_stop_time, bead_stop_index, -1 * direction) )

    # Concatenate and sort
    time_index_direction = np.vstack((bead_start_time_index_direction, bead_stop_time_index_direction))
    time_index_direction = time_index_direction[time_index_direction[:, 0].argsort()]

    return time_index_direction


