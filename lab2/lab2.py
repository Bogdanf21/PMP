import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az
import _tkinter


def problem1():
    np.random.seed(1)
    x = []
    for i in range(0, 10000):
        if stats.binom.rvs(n=1, p=0.4, size=1) == 0:
            x.append(stats.expon.rvs(0, scale=1 / 4, size=1)[0])
        else:
            x.append(stats.expon.rvs(0, scale=1 / 6, size=1)[0])

    az.plot_posterior({'Average:': x})
    az.plot_density({'Density': np.array(x)}, bw=0.001)  # bw is for the graph to be more accurate
    print("Mean: ", np.mean(x))
    print("Stdev: ", np.std(x))
    plt.show()


# problem 2 ######################################################################
def problem2():

    x = []
    for i in range(0, 10000):
        latency = stats.expon.rvs(0, 1 / 4, size=1)[0]
        choice = np.random.choice(a=[1, 2, 3, 4], p=[0.25, 0.25, 0.3, 0.2], size=1)[0]
        if choice == 1:
            x.append(stats.gamma.rvs(4, scale=1 / 3, size=1)[0] + latency)
        elif choice == 2:
            x.append(stats.gamma.rvs(4, scale=1 / 2, size=1)[0] + latency)
        elif choice == 3:
            x.append(stats.gamma.rvs(5, scale=1 / 2, size=1)[0] + latency)
        elif choice == 4:
            x.append(stats.gamma.rvs(5, scale=1 / 3, size=1)[0] + latency)
    print("Probability: ", (np.asarray(x) > 3).sum() / 10000)
    az.plot_density({"Density:": x})
    plt.show()


def problem3():
    ss = []
    sb = []
    bs = []
    bb = []
    for i in range(0, 100):
        exp = []
        for j in range(0, 10):
            toss1 = np.random.choice(a=[0, 1], p=[0.5, 0.5], size=1)[0]
            toss2 = np.random.choice(a=[0, 1], p=[0.3, 0.7], size=1)[0]
            match toss1, toss2:
                case 0, 0:
                    exp.append(1)
                case 0, 1:
                    exp.append(2)
                case 1, 0:
                    exp.append(3)
                case 1, 1:
                    exp.append(4)
        ss.append(exp.count(1))
        sb.append(exp.count(2))
        bs.append(exp.count(3))
        bb.append(exp.count(4))
    az.plot_posterior({"ss": np.array(ss), "sb": np.array(sb), "bs" : np.array(bs), "bb" : np.array(bb)})
    plt.show()


#problem1()
#problem2()
problem3()
