import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import math
from scipy import stats

if __name__ == '__main__':
    data = pd.read_csv("./Prices.csv")
    price = data['Price'].values
    speed = data['Speed'].values
    hard_drive = data['HardDrive'].values
    ram = data['Ram'].values
    premium = data['Premium'].values
    the_model = pm.Model()
    # ex1
    with the_model:
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta1 = pm.Normal('beta1', mu=0, sd=10)
        beta2 = pm.Normal('beta2', mu=0, sd=10)
        sigma = pm.HalfNormal('sigma', sd=10)
        mu = pm.Deterministic('mu', alpha + beta1 * speed + beta2 * np.log(hard_drive))
        Price_obs = pm.Normal('Price_obs', mu=mu, sd=sigma, observed=price)
        trace = pm.sample(1000, tune=200, chains=1)

    dictionary = {
        'alpha': trace['alpha'].tolist(),
        'beta1': trace['beta1'].tolist(),
        'beta2': trace['beta2'].tolist(),
        'sigma': trace['sigma'].tolist(),
    }

    df = pd.DataFrame(dictionary)
    az.plot_posterior(
        {"alpha": trace['alpha'], "beta1": trace['beta1'], "beta2": trace['beta2'], "sigma": trace['sigma']},
        hdi_prob=0.95)
    plt.show()


    def print_plot():
        fig, axes = plt.subplots(2, 2, sharex=False, figsize=(10, 8))
        axes[0, 0].scatter(speed, price, alpha=0.6)
        axes[0, 1].scatter(hard_drive, price, alpha=0.6)
        axes[1, 0].scatter(ram, price, alpha=0.6)
        axes[1, 1].scatter(premium, price, alpha=0.6)
        axes[0, 0].set_ylabel("Price")
        axes[0, 0].set_xlabel("Speed")
        axes[0, 1].set_xlabel("HardDrive")
        axes[1, 0].set_xlabel("Ram")
        axes[1, 1].set_xlabel("Premium")
        plt.savefig('price_correlations.png')


    print_plot()

# 3. The processor has an influence on the price of the machine


# Bonus: The price of a PC is not influenced by the fact that a manufacturer is premium. From fig 4 in which the pc's
# are classified by price and whether are premium or not we can observe that both cases follow a distribution in the
# same manner