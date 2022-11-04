import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import statistics

def main():

    data = pd.read_csv("./data.csv")
    mom_age = data['momage'].tolist()
    ppvt = data['ppvt'].tolist()
    plt.scatter(ppvt, mom_age)
    plt.show()

    a = mom_age
    mom_age.sort()
    alpha_real = a[1]
    sample_size = 100
    for i in mom_age:
        beta_real = mom_age[i] / ppvt[i]
        eps_real = np.random.normal(0, 0.25, size=sample_size)
        x = np.random.normal(40, 1, sample_size)
        y_real = alpha_real + beta_real * x
        y = y_real + eps_real
    _, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(x, y, 'C0.')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].plot(x, y_real, 'k')
    az.plot_kde(y, ax=ax[1])
    ax[1].set_xlabel('y')
    plt.show()


if __name__ == "__main__":
    main()