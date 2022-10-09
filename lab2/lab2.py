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

    plt.show()


# problem 2 ######################################################################
def problem2():
    latency = stats.expon.rvs(0, 1 / 4, size=10000)
    x = []
    for i in range(0, 10000):
        match np.random.choice(a=[1, 2, 3, 4], p=[0.25, 0.25, 0.3, 0.2], size=1)[0]:
            case 1:
                x.append(stats.gamma.rvs(4, scale=1 / 3, size=1)[0])
            case 2:
                x.append(stats.gamma.rvs(4, scale=1 / 2, size=1)[0])
            case 3:
                x.append(stats.gamma.rvs(5, scale=1 / 2, size=1)[0])
            case 4:
                x.append(stats.gamma.rvs(5, scale=1 / 3, size=1)[0])
    x += latency
    favorable = 0
    for i in x:
        if i > 3:
            favorable += 1

    print("Probability: ", favorable / 10000)

    az.plot_density({"Avg:": x})
    plt.show()


def problem3():
    ss = []
    sb = []
    bs = []
    bb = []
    for i in range(0, 100):
        experiment = []
        for i in range(0, 10):
            toss1 = np.random.choice(a=['s', 'b'], p=[0.5, 0.5], size=1)[0]
            toss2 = np.random.choice(a=['s', 'b'], p=[0.3, 0.7], size=1)[0]
            match toss1, toss2:
                case 's', 's':
                    experiment.append('ss')
                case 's', 'b':
                    experiment.append('sb')
                case 'b', 's':
                    experiment.append('bs')
                case 'b', 'b':
                    experiment.append('bb')
        ss.append(experiment.count('ss'))
        sb.append(experiment.count('sb'))
        bs.append(experiment.count('bs'))
        bb.append(experiment.count('bb'))

    az.plot_posterior({"Stema, Stema": np.array(ss)})
    az.plot_posterior({"stema, ban": np.array(sb)})
    az.plot_posterior({"ban, stema" : np.array(bs)})
    az.plot_posterior({"ban, ban" : np.array(bb)})
    plt.show()


# problem1()
# problem2()
problem3()
