import numpy as np
import pandas as pd
from numba import jit


#@jit(nopython=True)
def get_stuff(length1, length2):
    rate = np.zeros(length2)
    rate.astype(np.float64)
    number = np.zeros(length2)
    number.astype(np.int64)
    coeff = np.zeros(length2)
    coeff.astype(np.float64)

    time = 0
    interarrival_array = []
    for i in range(length2):
        stop = False
        while stop == False:
            interarrival = np.random.exponential(1)
            time = time + interarrival
            if time <= length1:
                interarrival_array.append(interarrival)
            else:
                interarrival_array = np.asarray(interarrival_array)
                rate[i] = 1/np.mean(interarrival_array)
                number[i] = len(interarrival_array)
                coeff[i] = np.std(interarrival_array)/np.mean(interarrival_array)
                time = 0
                interarrival_array = []
                stop = True

    return rate, number, coeff

def main():
    rate, number, coeff = get_stuff(100, 1000000)
    a = np.column_stack((rate, number))
    b = np.column_stack((a, coeff))

    df = pd.DataFrame(b)
    df.to_csv('/Users/jenniferhovis/documents/fluxus_josh/data/sample_size.csv')


main()