import numpy
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az


def problem1():
    np.random.seed(57)

    # x = stats.norm.rvs(0, 1, size=10000)  # Distributie normala cu media 0 si deviatie standard 1, 1000 samples y =
    # stats.uniform.rvs(-1, 2, size=10000)  # Distributie uniforma intre -1 si 1, 1000 samples . Primul parametru 
    # fiind limita inferioara a intervalului, al doilea parametru fiind "marimea" intervalului, aka [-1,-1+2] = [-1,
    # z = x + y  # Compunerea prin insumare a celor 2 distributii 

    m1 = stats.expon.rvs(0, 1 / 4, size=4000)
    m2 = stats.expon.rvs(0, 1 / 6, size=6000)
    avg = numpy.concatenate([m1, m2], axis=0)

    az.plot_posterior({'Mechanic 1': m1, 'Mechanic 2': m2, 'Average waiting time': avg})


# problem 2 ######################################################################
def problem2():
    latency = stats.expon.rvs(0, 1 / 4, size=10000)

    server1 = stats.gamma.rvs(4, scale=1 / 3, size=10000) + latency
    server2 = stats.gamma.rvs(4, scale=1 / 2, size=10000) + latency
    server3 = stats.gamma.rvs(5, scale=1 / 2, size=10000) + latency
    server4 = stats.gamma.rvs(5, scale=1 / 3, size=10000) + latency

    avg = 0.25 * server1 + 0.25 * server2 + 0.3 * server3 + 0.2 * server4

    az.plot_posterior({"Server 1": server1, "Server 2": server2, "Avg:": avg})
    plt.show()


problem1()
plt.show()
