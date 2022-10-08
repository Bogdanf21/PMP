
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az


def problem1():
    np.random.seed(1)

    # x = stats.norm.rvs(0, 1, size=10000)  # Distributie normala cu media 0 si deviatie standard 1, 1000 samples y =
    # stats.uniform.rvs(-1, 2, size=10000)  # Distributie uniforma intre -1 si 1, 1000 samples . Primul parametru
    # fiind limita inferioara a intervalului, al doilea parametru fiind "marimea" intervalului, aka [-1,-1+2] = [-1,
    # z = x + y  # Compunerea prin insumare a celor 2 distributii

    m1 = stats.expon.rvs(0, 1 / 4, size=4000)
    m2 = stats.expon.rvs(0, 1 / 6, size=6000)
    avg = (np.concatenate([m1, m2]))

    x = stats.norm.rvs(0, 1, size=10000)  # Distributie normala cu media 0 si deviatie standard 1, 1000 samples
    y = stats.uniform.rvs(-1, 2, size=10000)

    # az.plot_posterior({'Mechanic 1': m1})
    # az.plot_posterior({'Mechanic 2': m2})
    az.plot_posterior({'Average': avg})
    az.plot_density(avg,bw=0.001)  # bw is for the graph to be more accurate
    plt.show()


# problem 2 ######################################################################
def problem2():
    # latency = stats.expon.rvs(0, 1 / 4, size=SAMPLE_SIZE_FOR_EACH_SERVER)

    server1 = stats.gamma.rvs(4, scale=1 / 3, size=int(10000 * 0.25)) + stats.expon.rvs(0, 1 / 4, size = int(10000 * 0.25))
    server2 = stats.gamma.rvs(4, scale=1 / 2, size=int(10000 * 0.25)) + stats.expon.rvs(0, 1 / 4, size = int(10000 * 0.25))
    server3 = stats.gamma.rvs(5, scale=1 / 2, size=int(10000 * 0.3)) + stats.expon.rvs(0, 1 / 4, size = int(10000 * 0.3))
    server4 = stats.gamma.rvs(5, scale=1 / 3, size=int(10000 * 0.2)) + stats.expon.rvs(0, 1 / 4, size = int(10000 * 0.2))

    avg = np.concatenate([server1, server2, server3, server4])
    favorable = 0
    for i in avg:
        if i > 3:
            favorable += 1

    print("Probability: ", favorable/10000)

    az.plot_posterior({"Avg:": avg})
    plt.show()


def problem3():
    # toss 1 and toss 2 are independent => p(t1 ^ t2) = p(t1) * p(t2)
    # toss1 = np.random.choice(a=['s', 'b'], p=[0.5, 0.5], size=1)
    # toss2 = np.random.choice(a=['s', 'b'], p=[0.3, 0.7], size=1)
    # ss - 0, sb - 1, bs - 2, bb - 3
    chars = ['ss', 'sb', 'bs', 'bb']
    values = []
    for i in range(0,100):
        experiment = np.random.choice(a=[0, 1, 2, 3], p=[0.5*0.3, 0.5*0.7, 0.5*0.3, 0.5*0.7], size=10)
        values.append(experiment)
    values = np.array(values)

    plt.xticks([0, 1, 2, 3], chars)


    az.plot_density(values)


    plt.show()
#problem1()
#problem2()
problem3()

