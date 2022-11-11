import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import math
from scipy import stats

if __name__ == '__main__':
    data = pd.read_csv("./Prices.csv")
    Price = data['Price'].values
    Speed = data['Speed'].values
    hard_drive = data['HardDrive'].values
    Ram = data['Ram'].values
    Premium = data['Premium'].values
    the_model = pm.Model()
    # ex1
    with the_model:
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta1 = pm.Normal('beta1', mu=0, sd=10)
        beta2 = pm.Normal('beta2', mu=0, sd=10)
        sigma = pm.HalfNormal('sigma', sd=10)
        mu = pm.Deterministic('mu', alpha + beta1 * Speed + beta2 * np.log(hard_drive))
        Price_obs = pm.Normal('Price_obs', mu=mu, sd=sigma, observed=Price)
        trace = pm.sample(1000, tune=1000, cores=4, return_inferencedata=True)

    list = trace['beta1'].tolist()
    az.plot_posterior({"beta1": trace['beta1'].tolist(), "beta2": trace['beta2'].tolist()}, hdi_prob=0.95)
    plt.show()
